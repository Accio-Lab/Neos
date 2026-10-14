# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Split this file into smaller files.

import copy
import dataclasses
import functools
import gc
import inspect
import logging
import math
import traceback
import warnings
from collections import defaultdict, namedtuple
from contextlib import ExitStack, nullcontext
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.distributed import _coalescing_manager
from torch.distributed.tensor import DTensor, Replicate, Shard

from .mixed_precision import (
    MixedPrecisionPolicy,
    fp8_discard_transpose_cache,
    fp8_get_raw_data,
    fp8_need_transpose_data,
    fp8_need_transpose_data_for_meta_device_init,
    fp8_quantize,
    fp8_set_raw_data,
    get_quantized_model_init_context_cls,
    is_blockwise_float8tensor,
    is_float8tensor,
    is_te_min_version,
)
from .uneven_dtensor import update_uneven_dtensor_chunk_metadata, validate_uneven_dtensor
from .utils import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    FSDPDistributedIndex,
    get_global_memory_buffer,
    get_mcore_tensor_parallel_partition_dim,
    is_mcore_tensor_parallel_duplicated,
    log_single_rank,
    using_tensor_parallel,
)

logger = logging.getLogger(__name__)


try:
    
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.tensor_parallel import get_cuda_rng_tracker
    from megatron.core.utils import is_submodule

    HAVE_MCORE = True
    logger.info("Detected Megatron Core, using Megatron-FSDP with Megatron.")

except ImportError:
    
    from .distributed_data_parallel_config import DistributedDataParallelConfig
    from .utils import get_cuda_rng_tracker, is_submodule

    HAVE_MCORE = False
    logger.info("Megatron Core is not installed, Megatron-FSDP will run without Megatron Core.")

try:
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    HAVE_TE = True
except Exception:
    HAVE_TE = False

NCCL_ALLOCATOR = None

try:
    
    
    import megatron.core.nccl_allocator as nccl_allocator

    NCCL_ALLOCATOR = "MCORE"
except ImportError:
    try:
        import apex.contrib.nccl_allocator as nccl_allocator

        NCCL_ALLOCATOR = "APEX"
    except ImportError:
        nccl_allocator = None

NCCL_MEMORY_POOL = None


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error
    message ``s`` since otherwise, it is swallowed.
    """
    if not cond:
        logger.error(s)
        logger.error(''.join(traceback.format_stack()))
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDEr",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                tensor._typed_storage()._resize_(0)


TensorItemIndex = namedtuple(
    "TensorItemIndex", ["global_data_index", "size", "item_id", "bucket_id", "shape"]
)
BucketIndex = namedtuple("BucketIndex", ["bucket_id", "global_data_index", "size", "items"])
ShardBucketIndex = namedtuple(
    "ShardBucketIndex",
    ["bucket_id", "global_data_index", "local_data_index", "bucket_data_index", "size"],
)


class MultiGroupUBRAllocator:
    """
    A custom allocator class that registers a single memory pool with multiple different
    communication groups, which is not natively supported by apex's nccl_allocator.

    This is particularly useful for Mixture of Experts (MoE) models where:
    - Non-expert parameters/gradients use the data-parallel + context-parallel group (dp_cp_group)
    - Expert parameters/gradients use the expert-parallel + data-parallel group (ep_dp_group)

    Since Megatron-Core FSDP uses a contiguous single tensor for the entire model's parameters, we
    need to register the same memory pool with both communication groups to enable nccl algorithms
    that is relying on the user buffer registration for both expert and non-expert parameters.

    Implementation:
        It uses apex nccl_allocator internally to create a Tensor using ncclMemAlloc
        and register to the `group` and then registers the Mempool also for the `additional_group`

    Example:
        ```
        import apex.contrib.nccl_allocator as nccl_allocator
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool()
        group_1 = torch.distributed.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7], backend="nccl")
        group_2 = torch.distributed.new_group(ranks=[0, 2, 4, 6], backend="nccl")
        with MultiGroupUBRAllocator(pool, groups=[group_1, group_2]):
            a = torch.zeros(1024, dtype=torch.float32, device="cuda")
            b = torch.zeros(1024, dtype=torch.float32, device="cuda")
        ```
    """

    def __init__(self, pool, groups):  
        self.pool = pool
        self.groups = groups
        self.mem_allocator = nccl_allocator.nccl_mem(self.pool, group=self.groups[0])
        assert len(self.groups) > 1, "MultiGroupUBRAllocator requires at least two groups"

    def __enter__(self):
        for group in self.groups[1:]:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            try:
                
                
                
                backend.deregister_mem_pool(self.pool)
            except RuntimeError:
                pass
        self.mem_allocator.__enter__()

    def __exit__(self, *args):
        self.mem_allocator.__exit__(*args)
        for group in self.groups[1:]:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            log_single_rank(
                logger,
                logging.INFO,
                f"[MultiGroupUBRAllocator] Registering mem pool to group {group}, "
                f"group.group_desc:{group.group_desc}",
            )
            backend.register_mem_pool(self.pool)


@dataclasses.dataclass
class BucketingPolicy:
    """
    A policy for bucketing in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        suggested_bucket_size (int): The suggested size of each bucket in num of elements.
        fsdp_unit_modules (list): A list of module classes that are treated as a
            single unit for FSDP bucketing.
        data_parallel_sharding_strategy (str): The strategy used for sharding
            data parallel modules.

    Note:
        This policy is used to configure the bucketing behavior in FSDP training.
    """

    suggested_bucket_size: Optional[int] = 40_000_000
    fsdp_unit_modules: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    data_parallel_sharding_strategy: str = "no_shard"


def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)


def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    ddp_config: DistributedDataParallelConfig,
    bucket_id: int = 0,
    chunk_size_factor: int = 1,
) -> Tuple[List[tuple], BucketIndex, ShardBucketIndex]:
    """
    Assuming that all input tensor elements contiguously compose a global
    buffer, give the index range of every tensor, the bucket in the buffer,
    and the (distributed) shard within the bucket. Note that the global bucket
    buffer is only temporarily allocated, but is abstractly tracked via indices
    deduced from the number of raw parameters assigned to this buffer / bucket.

    Args:
        elements (List[torch.Size]): List of input tensor.
        data_parallel_rank (int): Rank of the current process in the data parallel group.
        data_parallel_world_size (int): World size of the data parallel group.
        bucket_id (int, optional): The id of the bucket. Defaults to 0.

    Returns:
        Tuple[Dict[int, TensorItemIndex], BucketIndex, ShardBucketIndex]: The index
            range of every tensor, every bucket and every in bucket local buffer.
    """

    def _pad_if_needed(data_index: int) -> int:
        if ddp_config.data_parallel_sharding_strategy != "no_shard":
            return _pad(data_index, data_parallel_world_size * chunk_size_factor)
        return data_index

    def add_item(item_id, item, offset, item_index_map):
        
        
        item_index_map[item_id] = TensorItemIndex(
            
            
            global_data_index=offset,
            
            size=item.numel(),
            
            item_id=item_id,
            
            bucket_id=bucket_id,
            
            shape=item,
        )

    fragment_items = []
    regular_items = []
    for item_id, item in enumerate(elements):
        if item.numel() < chunk_size_factor:
            fragment_items.append((item_id, item))
        else:
            item[1:].numel()
            regular_items.append((item_id, item))

    
    
    sorted(fragment_items, key=lambda id_item: -id_item[1].numel())

    
    
    item_index_map = {}
    data_index = 0
    while len(regular_items) > 0:
        item_id, item = regular_items.pop(0)
        add_item(item_id, item, data_index, item_index_map)
        if item.numel() % chunk_size_factor == 0:
            data_index += item.numel()
            continue

        gap_offset = data_index + item.numel()
        data_index += (item.numel() // chunk_size_factor + 1) * chunk_size_factor
        remain = item.numel() % chunk_size_factor
        space = chunk_size_factor - remain
        found_rhs = False
        for id_rhs in regular_items[:]:
            rhs_id, rhs = id_rhs
            if rhs.numel() % chunk_size_factor == 0:
                continue
            rhs_remain = rhs.numel() % chunk_size_factor
            if remain + rhs_remain <= chunk_size_factor:
                found_rhs = True
                regular_items.remove(id_rhs)
                break

        
        
        if found_rhs:
            add_item(rhs_id, rhs, data_index - rhs_remain, item_index_map)
            space -= rhs_remain
            data_index += rhs.numel() // chunk_size_factor * chunk_size_factor

        
        for id_frag in fragment_items[:]:
            frag_id, frag = id_frag
            if frag.numel() > space:
                continue
            add_item(frag_id, frag, gap_offset, item_index_map)
            space -= frag.numel()
            gap_offset += frag.numel()
            fragment_items.remove(id_frag)

    for frag_id, frag in fragment_items:
        add_item(frag_id, frag, data_index, item_index_map)
        data_index += frag.numel()

    
    bucket_index = BucketIndex(
        bucket_id=bucket_id,
        global_data_index=0,
        size=_pad_if_needed(data_index),
        items=list(item_index_map.values()),
    )

    
    shard_bucket_index = _get_dp_buffer_shard_bucket_index(
        bucket_index, is_data_distributed, data_parallel_world_size, data_parallel_rank
    )

    
    
    
    return item_index_map, bucket_index, shard_bucket_index


def _get_dp_buffer_shard_bucket_index(
    bucket_index: BucketIndex,
    is_data_distributed: bool,
    data_parallel_world_size: int,
    data_parallel_rank: int,
) -> ShardBucketIndex:
    """
    Build the data parallel buffer shard bucket index from the bucket index.

    Args:
        bucket_index (BucketIndex): The bucket index containing information on the
            items in the bucket.
        is_data_distributed (bool): Whether the data is distributed across multiple
            processes.
        data_parallel_world_size (int): The world size of the data parallel group.
        data_parallel_rank (int): The rank of the current process in the data parallel group.

    Returns:
        ShardBucketIndex: The shard bucket index containing information on the
            location and size of the buffer shard in the global bucket.
    """
    
    
    
    shard_size = bucket_index.size // data_parallel_world_size
    bucket_data_index = shard_size * data_parallel_rank

    
    global_data_index = bucket_index.global_data_index + bucket_data_index

    if is_data_distributed:
        
        
        shard_bucket_index = ShardBucketIndex(
            bucket_id=bucket_index.bucket_id,
            
            global_data_index=global_data_index,
            
            local_data_index=0,
            
            bucket_data_index=bucket_data_index,
            size=shard_size,  
        )
    else:
        
        
        shard_bucket_index = ShardBucketIndex(
            bucket_id=bucket_index.bucket_id,
            global_data_index=global_data_index,
            
            
            
            local_data_index=global_data_index,
            bucket_data_index=bucket_data_index,
            size=shard_size,
        )
    return shard_bucket_index


@dataclasses.dataclass
class Bucket:
    """
    A container for holding data in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        data (torch.Tensor): A tensor containing the data elements
            grouped together in a bucket.
            used to synchronize data operations.

    Note:
        Buckets are used to optimize communication in FSDP training by
            grouping small tensors together.
    """

    data: torch.Tensor


class TemporaryBucketAllocator:
    """
    A utility class for managing temporary buckets (buffers) used in FSDP
    operations like parameters unshard and gradients reduction.

    This allocator handles the dynamic allocation and deallocation of temporary memory buffers
    needed during FSDP (Fully Sharded Data Parallel) operations, particularly for parameters
    unshard and gradients reduction. It helps optimize memory usage by allowing temporary
    buckets to be released when no longer needed.

    Key Features:
        - Dynamic allocation of temporary buckets for FSDP operations
        - Memory-efficient management of temporary buffers
        - Support for both parameters unshard and gradients reduction operations
        - Automatic cleanup of unused buckets to save memory

    Usage:
        ```python
        # Create an allocator instance
        allocator = TemporaryBucketAllocator(name="gpt_parameters")

        # Allocate a temporary bucket
        temp_bucket = allocator.allocate(size=1024, dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done
        allocator.free(temp_bucket)
        ```

    Note:
        It's important to release temporary buckets after use to prevent memory leaks
        and optimize memory usage during training.
    """

    def __init__(self):
        self.buckets = {}

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[bucket_id]

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)
            del self.buckets[bucket_id]


class StorageResizeBasedBucketAllocator(TemporaryBucketAllocator):
    """
    A specialized temporary bucket allocator that resizes the storage of temporary buckets
    based on the required size.
    """

    def __init__(self):
        super().__init__()

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        bucket = self.buckets[bucket_id]
        _alloc_storage(bucket.data, torch.Size([size]))
        return bucket

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)


class RotaryBucketAllocator(TemporaryBucketAllocator):
    """A specialized temporary bucket allocator that implements a circular buffer recycling strategy
    to minimize memory fragmentation in FSDP operations.

    RotaryBucketAllocator extends TemporaryBucketAllocator by maintaining a limited pool of
    pre-allocated buffers that are reused in a circular manner. This approach helps prevent
    memory fragmentation that typically occurs with frequent allocation and deallocation of
    temporary buffers during FSDP operations.

    Key Features:
        - Circular buffer recycling strategy for memory efficiency
        - Reduced memory fragmentation compared to dynamic allocation
        - Pre-allocated buffer pool for faster access
        - Automatic buffer reuse without explicit deallocation

    Usage:
        ```python
        # Create a rotary allocator
        allocator = RotaryBucketAllocator(name="gpt_parameters")

        # Get a temporary buffer from the pool
        temp_bucket = allocator.allocate(dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done, make it in idle buffer pool
        allocator.free(temp_bucket)
        ```
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.num_global_buffer = 0
        self.idle_buffer = []  
        self.using_buffer = {}  

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """

        def _get_global_buffer(buffer_id: int):
            return get_global_memory_buffer().get_tensor(
                [size],
                dtype=dtype,
                name=self._get_gbuf_name(buffer_id),
                mem_alloc_context=mem_alloc_context,
            )

        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer[bucket_id]
            return Bucket(data=_get_global_buffer(buffer_id))

        if len(self.idle_buffer) == 0:
            
            buffer_id = self.num_global_buffer
            self.num_global_buffer += 1
            self.idle_buffer.append(buffer_id)

        buffer_id = self.idle_buffer.pop(0)
        self.using_buffer[bucket_id] = buffer_id
        return Bucket(data=_get_global_buffer(buffer_id))

    def _get_gbuf_name(self, buffer_id: int):
        return f"{self.name}_{buffer_id}"

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer.pop(bucket_id)
            self.idle_buffer.append(buffer_id)


class FixedPoolAllocator(TemporaryBucketAllocator):
    """
    A specialized temporary bucket allocator that implements a buffer recycling strategy
    to minimize memory fragmentation in FSDP operations.

    This allocator maintains a fixed pool of pre-allocated buffers, reusing them
    to reduce the overhead and fragmentation caused by frequent allocation and
    deallocation of temporary buffers during FSDP operations.
    """

    def __init__(
        self,
        name: str,
        fsdp_param_groups: List["ParameterGroup"],
        size: int = 2,
        fallback_to_persistent_buffer: bool = False,
    ):
        self.name = name
        self.fsdp_param_groups = fsdp_param_groups
        self.size = size  
        self.allocation_tracker = {}  

        
        fsdp_unit_buckets = defaultdict(list)
        for bucket_id, param_group in enumerate(fsdp_param_groups):
            if param_group.fsdp_unit_id == -1 or param_group.fsdp_unit_id is None:
                continue
            fsdp_unit_buckets[param_group.fsdp_unit_id].append(bucket_id)
        self.fsdp_unit_buckets = fsdp_unit_buckets

        
        fsdp_units_to_double_buffer = []
        for fsdp_unit_id, bucket_ids in fsdp_unit_buckets.items():
            same_storage_fsdp_units = []
            for i in fsdp_unit_buckets:
                if self._is_two_bucket_group_equal(fsdp_unit_buckets[i], bucket_ids):
                    same_storage_fsdp_units.append(i)
            
            if len(same_storage_fsdp_units) > len(fsdp_units_to_double_buffer):
                fsdp_units_to_double_buffer = same_storage_fsdp_units

        
        
        
        assert (
            len(fsdp_units_to_double_buffer) > 0
        ), "Found no FSDP units to use fixed-size buffering"
        self.fsdp_double_buffer_units = fsdp_units_to_double_buffer

        if torch.distributed.get_rank() == 0:
            for bucket_id, param_group in enumerate(fsdp_param_groups):
                if (
                    param_group.fsdp_unit_id == -1
                    or param_group.fsdp_unit_id is None
                    or param_group.fsdp_unit_id not in self.fsdp_double_buffer_units
                ):
                    logging.info(
                        f"FSDP unit (id={param_group.fsdp_unit_id}) does not fit "
                        "in FixedPoolAllcator"
                    )
                    if fallback_to_persistent_buffer is False:
                        logging.info(
                            "It will fall back to dynamic memory allocator, NCCL user "
                            "buffer is not supported"
                        )
                    else:
                        logging.info(
                            "It will be allocated a persistent buffer. If the memory "
                            "budget is tight, set "
                            "trainer.strategy.ddp.fsdp_db_use_persist_buf_on_alloc_fail to False."
                        )

        
        
        self.idle_buffer = []  
        self.using_buffer = {}  

        
        for buf_group_id in range(self.size):  
            num_bucket = len(self.fsdp_unit_buckets[self.fsdp_double_buffer_units[0]])
            for bucket_offset in range(num_bucket):
                self.idle_buffer.append((buf_group_id, bucket_offset))

        
        self.fallback_to_persistent_buffer = fallback_to_persistent_buffer
        self.backup_allocator = TemporaryBucketAllocator()

    def _is_two_bucket_group_equal(self, group_a, group_b):
        
        if len(group_a) != len(group_b):
            return False

        for a, b in zip(group_a, group_b):
            pg_a = self.fsdp_param_groups[a]
            pg_b = self.fsdp_param_groups[b]
            a_size = sum(p.numel() for p in pg_a.params)
            b_size = sum(p.numel() for p in pg_b.params)
            if pg_a.dtype != pg_b.dtype or a_size != b_size:
                return False
        return True

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """
        fsdp_unit_id = self.fsdp_param_groups[bucket_id].fsdp_unit_id
        if fsdp_unit_id in self.fsdp_double_buffer_units:
            
            bucket_offset = self.fsdp_unit_buckets[fsdp_unit_id].index(bucket_id)
            buffer_name = None
            if bucket_id in self.using_buffer:
                
                buf_group_id, bucket_offset = self.using_buffer[bucket_id]
                buffer_name = self._get_gbuf_name(buf_group_id, bucket_offset)
            else:
                
                for buf_group_id in range(self.size):
                    if (buf_group_id, bucket_offset) in self.idle_buffer:
                        self.using_buffer[bucket_id] = (buf_group_id, bucket_offset)
                        buffer_name = self._get_gbuf_name(buf_group_id, bucket_offset)
                        self.idle_buffer.remove((buf_group_id, bucket_offset))
                        break

            assert buffer_name is not None, (
                f"[FSDP][Rank {torch.distributed.get_rank()}][{self.name}] "
                f"No buffer found for bucket_id: {bucket_id}, fsdp_unit_id: {fsdp_unit_id}, "
                f"bucket_offset: {bucket_offset} \n"
                f"current using_buffer: {self.using_buffer} \n"
                f"current idle_buffer: {self.idle_buffer}"
            )
        elif self.fallback_to_persistent_buffer is True:
            buffer_name = f"{self.name}_not_fit_in_fixed_pool_{bucket_id}_{size}_{dtype}_{device}"
        else:
            
            
            
            logging.debug(f"[FSDP] Using backup allocator for {bucket_id} {fsdp_unit_id}")
            return self.backup_allocator.allocate(
                bucket_id=bucket_id, size=size, dtype=dtype, device=device
            )

        
        if mem_alloc_context is not None and mem_alloc_context != nullcontext:
            
            if (
                self.allocation_tracker.get((buffer_name, dtype), None) is None
                or self.allocation_tracker[(buffer_name, dtype)] < size
            ):
                
                self.allocation_tracker[(buffer_name, dtype)] = size
                torch.cuda.synchronize()
        return Bucket(
            data=get_global_memory_buffer().get_tensor(
                [size], dtype=dtype, name=buffer_name, mem_alloc_context=mem_alloc_context
            )
        )

    def _get_gbuf_name(self, buf_group_id: int, bucket_index: int):
        return f"{self.name}_{buf_group_id}_{bucket_index}"

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        fsdp_unit_id = self.fsdp_param_groups[bucket_id].fsdp_unit_id
        if fsdp_unit_id in self.fsdp_double_buffer_units:
            if bucket_id not in self.using_buffer:
                
                return
            
            self.idle_buffer.append(self.using_buffer[bucket_id])
            del self.using_buffer[bucket_id]
            return
        if self.fallback_to_persistent_buffer is False:
            
            logging.debug(f"[FSDP] Free from the backup allocator for {bucket_id} {fsdp_unit_id}")
            self.backup_allocator.free(bucket_id)


class DataParallelBuffer:
    """
    A class that manages the data parallel buffer for Fully Sharded Data Parallel (FSDP) training.
    It has two operating modes given a bucket of module parameters:

        - Sharded: The bucket is sharded across the data parallel group, and each
            rank will manage a shard of the bucket that is persistently stored in this buffer.
        - Unsharded: The bucket is not sharded, and the entire bucket is persistently
            stored in this buffer. Virtual shards of this unsharded buffer can be
            retrieved from each rank when needed.

    This design supports interoperability of sharded and unsharded buffers,
    e.g. optim and optim_grads,
    where buffers associated with sharded parameters can be utilized with buffers
    associated with unsharded parameters through the use of "virtual" or rank-specific
    shards for the unsharded buffers.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        is_data_distributed: bool,
        bucket_id: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_rank: Optional[int] = None,
        temporary_bucket_allocator: Optional[TemporaryBucketAllocator] = None,
        is_transpose_buffer: bool = False,
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        mem_alloc_context: Optional[Callable] = None,
        item_index_map: Optional[Dict[int, TensorItemIndex]] = None,
        bucket_index: Optional[BucketIndex] = None,
        shard_bucket_index: Optional[ShardBucketIndex] = None,
    ) -> None:
        self.ddp_config = ddp_config
        self.params = params
        _param_dtype = {p.dtype for p in self.params}

        assert len(_param_dtype) == 1, f"params have different dtypes: {_param_dtype}"
        self.is_data_distributed = is_data_distributed
        self.bucket_id = bucket_id
        self.dtype = dtype if dtype else next(iter(_param_dtype))
        self.device = device
        self.data_parallel_group = data_parallel_group
        
        
        if dp_rank is not None:
            self.dp_rank = dp_rank
        else:
            self.dp_rank = torch.distributed.get_rank(data_parallel_group)
        self.dp_world_size = torch.distributed.get_world_size(data_parallel_group)
        self.temporary_bucket_allocator = (
            temporary_bucket_allocator if temporary_bucket_allocator else TemporaryBucketAllocator()
        )
        self.is_transpose_buffer = is_transpose_buffer
        self.gradient_scaling_factor = gradient_scaling_factor
        self.mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
        self.chunk_size_factor = chunk_size_factor

        
        
        if item_index_map or bucket_index or shard_bucket_index:
            assert (
                item_index_map is not None
            ), "item_index_map must be provided if bucket_index or shard_bucket_index is provided"
            assert (
                bucket_index is not None
            ), "bucket_index must be provided if item_index_map or shard_bucket_index is provided"
            assert (
                shard_bucket_index is not None
            ), "shard_bucket_index must be provided if item_index_map or bucket_index is provided"
            self.item_index_map = item_index_map
            self.bucket_index = bucket_index
            self.shard_bucket_index = shard_bucket_index
        else:
            
            
            
            (self.item_index_map, self.bucket_index, self.shard_bucket_index) = (
                build_data_parallel_buffer_index(
                    [to_local_if_dtensor(p).shape for p in self.params],
                    self.dp_rank,
                    self.dp_world_size,
                    is_data_distributed,
                    ddp_config,
                    bucket_id=bucket_id,
                    chunk_size_factor=chunk_size_factor,
                )
            )

        self.data_size = (
            self.bucket_index.size if not is_data_distributed else self.shard_bucket_index.size
        )

        
        self.param_idx = {p: i for i, p in enumerate(self.params)}

    def init_data(self, data: torch.Tensor):
        """Allocate a buffer Tensor to persistently store the data for this
        (shard of) the buffer.
        """
        assert data.dtype == self.dtype, f"Data type mismatch: {data.dtype} != {self.dtype}"
        assert (
            data.numel() == self.data_size
        ), f"Data size mismatch: {data.numel()} != {self.data_size}"
        self.data = data

    def fetch_bucket(
        self, dtype: Optional[torch.dtype] = None, set_param_data: bool = False
    ) -> Bucket:
        """
        Fetch a communication buffer for data-parallel operations. If the buffer
        is not distributed and doesn't have a custom data-type, the local un-sharded
        buffer data will be returned. Otherwise, an un-sharded bucket is allocated.

        The size of the bucket is defined by the `DataParallelBuffer` instance.

        Args:
            dtype (Optional[torch.dtype]): The data type of the tensor
                to fetch a buffer for. Defaults to None.

        Returns:
            Bucket: The communication buffer for the specified data type.
        """
        if dtype is None:
            dtype = self.dtype

        if not self.is_data_distributed and dtype == self.dtype:
            
            bucket_index = self.bucket_index
            bucket = Bucket(
                data=self.data[
                    bucket_index.global_data_index : bucket_index.global_data_index
                    + bucket_index.size
                ]
            )
        else:
            
            bucket = self.allocate_bucket_storage(dtype=dtype, device=self.device)

        
        if set_param_data:
            for p in self.params:
                item_id = self.param_idx[p]
                p = to_local_if_dtensor(p)
                data = self.get_item_from_bucket(bucket, item_id).view(p.shape)
                if is_float8tensor(p):
                    fp8_set_raw_data(p, data, self.is_transpose_buffer)
                else:
                    p.data = data
        return bucket

    def allocate_bucket_storage(
        self,
        shard: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        init_values: Optional[torch.Tensor] = None,
    ) -> Bucket:
        """
        Allocate a temporary flat communication buffer using the cached
        allocator associated with this DataParallelBuffer. If no bucket
        corresponding to the Bucket ID of this DataParallelBuffer has
        been allocated yet, an empty Tensor will be allocated.

        Can allocate sharded or un-sharded buckets.

        Optionally, if init_values is provided, the flattened values of
        the init_values will be copied into the newly-allocated storage.

        Args:
            shard (bool):
                Whether to allocate a sharded or un-sharded bucket with
                sizes defined by this DataParallelBuffer.
            dtype (Optional[torch.dtype]):
                Data-type of the allocated bucket.
            device (Optional[torch.device]):
                Device of the allocated bucket.
            init_values (Optional[torch.Tensor]):
                If provided, the allocated storage will be initialized
                to the values of this (flattened) Tensor.

        Returns:
            Bucket: The communication buffer for the specified data type.
        """
        
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        
        alloc_size = self.shard_bucket_index.size if shard else self.bucket_index.size
        bucket = self.temporary_bucket_allocator.allocate(
            bucket_id=self.bucket_index.bucket_id,
            size=alloc_size,
            dtype=dtype,
            device=device,
            mem_alloc_context=self.mem_alloc_context,
        )
        
        if init_values is not None:
            assert bucket.data.shape == init_values.flatten().shape, (
                "[allocate_bucket_storage] Size mismatch between allocated bucket "
                f"({bucket.data.shape}) and init_values ({init_values.shape})!"
            )
            bucket.data.detach().copy_(init_values.flatten())
        return bucket

    def free_bucket_storage(self):
        """
        Release the storage of a temporarily-allocated communication bucket.
        """
        self.temporary_bucket_allocator.free(self.bucket_index.bucket_id)

    def reset_param_main_grad(self):
        """
        Dereference param.main_grad for grad buckets managed by this buffer.
        """
        
        for param in self.params:
            if hasattr(param, "main_grad"):
                param.main_grad = None

    def _get_item_slice_in_shard(self, item_id: int) -> Tuple[int, int]:
        """
        Return the coordinates of the slice of the item that is contained
        in this buffer shard. In other words, this returns the coordinates
        of all of the data in this item that is stored in this shard.

        Maps to the global coordinates of the item in the bucket when added to
        the starting coordinate of the item in the bucket, and maps to the local
        coordinates of the item in the shard when added to the difference between
        the starting coordinate of the item and the starting coordinate of the
        shard in the global bucket (i.e. mapping from item coordinates to global
        coordinates to shard coordinates).
        """
        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index

        
        
        
        
        
        item_global_start = item_index.global_data_index
        item_global_end = item_index.global_data_index + item_index.size
        shard_bucket_start = shard_bucket_index.global_data_index
        shard_bucket_end = shard_bucket_index.global_data_index + shard_bucket_index.size

        
        if item_global_start > shard_bucket_end or item_global_end < shard_bucket_start:
            return (0, 0)

        
        
        
        
        
        
        start = max(item_global_start, shard_bucket_start) - item_global_start
        
        
        
        
        
        end = min(item_global_end, shard_bucket_end) - item_global_start

        
        
        return (start, end)

    def locate_item_in_global_item(self, item_id: int) -> Tuple[int, int]:
        """
        Return the coordinates of the slice of the item that is contained
        in this buffer shard. In other words, this returns the coordinates
        of all of the data in this item that is stored in this shard.

        Helper function that adds a shortcut when the buffer is not sharded,
        in which case we don't need to compute the item-shard intersection,
        and can simply return the coordinates of the entire item.
        """
        item_index = self.item_index_map[item_id]
        if not self.is_data_distributed:
            
            return (0, item_index.size)

        
        slice_start, slice_end = self._get_item_slice_in_shard(item_id)
        if slice_start == slice_end:
            return (0, 0)
        return (slice_start, slice_end)

    def _get_item_local_shard_index(self, item_id: int) -> Tuple[int, int]:
        """
        Return the local coordinates of the slice of this buffer's shard that
        contains the item with the given ID. In other words, this returns the
        coordinates of all of the data in this shard associated with the item.

        Maps to the global coordinates of the item in the bucket when added to
        the starting coordinate of the shard in the global bucket, and maps to
        the coordinates of the item contained in the shard when added to the
        difference between the starting coordinate of the shard and the starting
        coordinate of the item in the global bucket (i.e. mapping from shard
        coordinates to global coordinates to item coordinates).
        """
        
        slice_start, slice_end = self._get_item_slice_in_shard(item_id)
        if slice_start == slice_end:
            
            return (0, 0)

        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index

        """
        Compute the offset that maps the coordinates of the slice of the item in
        this shard to the local coordinates of the slice of this shard that contains
        the item, for retrieval of the item's data stored in this shard.
            - If distributed, then evaluates to item_start - shard_start
                (because shard_local_data_index = 0).
            - If not distributed, then evaluates to item_start (because
                shard_local_data_index = shard_global_data_index).
                This maps the coordinates of the slice of the item in this shard to
                the global coordinates of the slice of the item in the bucket because
                the unsharded buffer entirely backs the global bucket.
        """
        offset = (
            item_index.global_data_index
            - shard_bucket_index.global_data_index
            + shard_bucket_index.local_data_index
        )

        
        
        return (offset + slice_start, offset + slice_end)

    def _get_item_local_index(self, item_id: int) -> Tuple[int, int]:
        """
        Return the local coordinates of the slice of this buffer's data that
        contains the item with the given ID.
        """
        if not self.is_data_distributed:
            
            
            
            
            item_index = self.item_index_map[item_id]
            
            return (item_index.global_data_index, item_index.global_data_index + item_index.size)
        
        
        return self._get_item_local_shard_index(item_id)

    def set_item(self, item_id: int, item_data: torch.Tensor) -> None:
        """
        Update a Tensor item managed by the `DataParallelBuffer` instance,
        i.e. store (a shard of) the Tensor in this buffer's datastore.

        The storage of the item is mapped to the communication bucket.
        This method updates the item data and ensures consistency with the bucket.

        Args:
            item_id (int): The ID of the tensor item to update.
            item_data (torch.Tensor): The new data for the tensor item.

        Returns:
            None
        """
        
        
        
        if is_float8tensor(item_data):
            item_data = fp8_get_raw_data(item_data, self.is_transpose_buffer)

        if self.is_data_distributed:
            
            slice_start, slice_end = self._get_item_slice_in_shard(item_id)
            
            item_data = item_data.flatten()[slice_start:slice_end]
        
        
        local_index_start, local_index_end = self._get_item_local_index(item_id)
        
        
        shard = self.data[local_index_start:local_index_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, only_shard: bool = False) -> torch.Tensor:
        """
        Retrieve a tensor item managed by the `DataParallelBuffer` instance,
        i.e. get all the item data stored in this sharded or unsharded buffer.

        The storage of the item is mapped to the communication bucket.
        If `only_shard` is True, returns only the shard of the item corresponding
            to the current process / rank, a "virtual shard" for unsharded buffers.
        Otherwise, returns the entire item, which could be a bucket shard or bucket.

        Args:
            item_id (int): The ID of the tensor item to retrieve.
            only_shard (bool, optional): Whether to return only the shard of the
                item. Defaults to False.

        Returns:
            torch.Tensor: The retrieved tensor item.
        """
        if only_shard:
            
            
            
            
            
            start, end = self._get_item_local_shard_index(item_id)
        else:
            
            
            
            
            start, end = self._get_item_local_index(item_id)

        return self.data[start:end]

    def get_item_from_bucket(self, bucket: Bucket, item_id: int):
        """
        Get Tensor item data from the given bucket specified by the item ID.
        """
        item_index = self.item_index_map[item_id]
        bucket_index = self.bucket_index
        start_index = item_index.global_data_index - bucket_index.global_data_index
        end_index = start_index + item_index.size
        item = bucket.data[start_index:end_index]
        return item

    def get_shard_from_bucket(self, bucket: Bucket):
        """
        Get the shard from the provided bucket associated with the sharding strategy of this buffer.
        """
        shard_bucket_index = self.shard_bucket_index
        offset = shard_bucket_index.bucket_data_index
        shard_size = shard_bucket_index.size
        shard = bucket.data[offset : offset + shard_size]
        return shard

    def get_shard_from_local_buffer(self) -> torch.Tensor:
        """
        Get the shard or virtual shard of the data persistently stored in this buffer.
        """
        index = self.shard_bucket_index
        
        
        
        return self.data[index.local_data_index : index.local_data_index + index.size]


@dataclasses.dataclass
class ParameterGroup:
    """
    Represents a group of model parameters along with metadata for managing
    data-parallel training in PyTorch.

    This class encapsulates a list of parameters and associated information
    such as data type, gradient requirements, and references to buffers used
    in distributed training contexts.

    Attributes:
        params (List[torch.nn.Parameter]):
            The list of model parameters grouped together.
        dtype (Optional[torch.dtype]):
            The desired data type for the parameters.
        is_expert_param (bool):
            Indicates if this group contains expert parameters
            (e.g., in mixture-of-experts).
        requires_grad (Optional[bool]):
            Specifies if gradients should be computed for these parameters.
        fsdp_unit_id (Optional[int]):
            Identifier for Fully Sharded Data Parallel (FSDP) unit grouping.
        chunk_size_factor (int):
            Factor determining chunk size for grouped parameter processing.
        model_weight_buffer (Optional[DataParallelBuffer]):
            Buffer used to store model weights for data-parallel operations.
        transpose_weight_buffer (Optional[DataParallelBuffer]):
            Buffer used to store transpose weights for data-parallel operations.
        main_weight_buffer (Optional[DataParallelBuffer]):
            Buffer used to store main model weights for data-parallel operations.
        main_grad_buffer (Optional[DataParallelBuffer]):
            Buffer used to store main gradients for data-parallel operations.
        hfsdp_helper_wbuf (Optional[DataParallelBuffer]):
            Inner-DP helper buffer that owns persistent HFSDP parameter shards.
            Created only when Hybrid FSDP (optimizer-state) full sharding is enabled.
        hfsdp_helper_wtbuf (Optional[DataParallelBuffer]):
            Inner-DP helper buffer that stores transpose weights for FP8/MXFP8.
            Created only when Hybrid FSDP (optimizer-state) full sharding is enabled.
        hfsdp_helper_gbuf (Optional[DataParallelBuffer]):
            Inner-DP helper buffer that owns persistent HFSDP gradient shards.
            Created only when Hybrid FSDP (optimizer-state) full sharding is enabled.
        hsdp_comm_gbuf (Optional[DataParallelBuffer]):
            Extra buffer to allocate buffers that enable custom gradient
            communication data-types when using HSDP or HFSDP only.
            Only allocates memory when `MixedPrecisionPolicy.grad_comm_dtype`
            is set to a non-`None` `torch.dtype`. Contains no local data.
    """

    params: List[torch.nn.Parameter]
    dtype: Optional[torch.dtype] = None
    is_expert_param: bool = False
    requires_grad: Optional[bool] = None
    fsdp_unit_id: Optional[int] = None
    chunk_size_factor: int = 1
    model_weight_buffer: Optional[DataParallelBuffer] = None
    transpose_weight_buffer: Optional[DataParallelBuffer] = None
    main_weight_buffer: Optional[DataParallelBuffer] = None
    main_grad_buffer: Optional[DataParallelBuffer] = None
    hfsdp_helper_wbuf: Optional[DataParallelBuffer] = None
    hfsdp_helper_wtbuf: Optional[DataParallelBuffer] = None
    hfsdp_helper_gbuf: Optional[DataParallelBuffer] = None
    hsdp_comm_gbuf: Optional[DataParallelBuffer] = None


def _get_parameter_groups(
    module: torch.nn.Module,
    policy: BucketingPolicy,
    meta_device_init_fp8_params: dict,
    bucket_group_by_fsdp_unit: bool = True,
):
    """
    Get the parameter group for the given module and parameters.

    Args:
        module (torch.nn.Module): The module whose parameters are to be grouped
            and flattened.
        policy (BucketingPolicy): The bucketing policy.
        meta_device_init_fp8_params (dict): A dictionary mapping parameter names to
            a boolean indicating whether the parameter is initialized on the meta device.
        bucket_group_by_fsdp_unit (bool): Whether to group buckets by FSDP unit.

    Returns:
        Tuple[List[ParameterGroup], Dict[torch.nn.Parameter, int], Dict[int, List[int]]]:
            - The list of parameter groups.
            - The mapping from parameters to their bucket group ID.
            - The mapping from bucket ID to the full group of bucket IDs that are
                NCCL-aggregated with this bucket ID.
    """

    
    param_to_name = {p: name for name, p in module.named_parameters()}
    
    fsdp_units = []
    if policy.fsdp_unit_modules:
        fsdp_modules = []
        
        for m in module.modules():
            
            
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue
            
            
            if isinstance(m, tuple(policy.fsdp_unit_modules)):
                fsdp_units.append([param_to_name[p] for p in m.parameters()])
                fsdp_modules.append(m)

    def _does_param_require_new_bucket(param):
        """
        Split shared embedding parameters into separate bucket if using distributed
        optimizer that makes use of reduce-scatters instead of all-reduces.
        This ensures that the first and last pipeline stage partition optimizer state
        for the shared embedding parameters the same way across DP replicas, allowing
        the DP reduce-scatter to be before the embedding all-reduce.
        """
        return (
            getattr(param, "shared_embedding", False)
            and policy.data_parallel_sharding_strategy != "no_shard"
        )

    is_expert_parameter = lambda n, p: ".experts." in n

    
    
    
    parameter_groups = []
    for name, param in module.named_parameters():
        
        is_fp8 = is_float8tensor(param)
        is_fp8_meta_device_init = meta_device_init_fp8_params.get(name, (False, False))[0]
        param_attrs = dict(
            dtype="float8" if (is_fp8 or is_fp8_meta_device_init) else param.dtype,
            is_expert_param=is_expert_parameter(name, param),
            requires_grad=param.requires_grad,
            fsdp_unit_id=None,
        )

        
        
        for fsdp_unit_id, fsdp_unit in enumerate(fsdp_units):
            if name in fsdp_unit:
                param_attrs["fsdp_unit_id"] = fsdp_unit_id
                break

        found_group = False
        
        for param_group in parameter_groups:
            group_attrs = {
                key: value for key, value in param_group.__dict__.items() if key in param_attrs
            }
            
            if group_attrs == param_attrs:
                param_group.params.append(param)
                found_group = True
                break

        
        if not found_group:
            parameter_groups.append(ParameterGroup([param], **param_attrs))

    
    
    suggested_bucket_size = policy.suggested_bucket_size
    bucket_groups = []
    for group in parameter_groups:
        bucket = []

        
        basic_attrs = {
            key: value
            for key, value in group.__dict__.items()
            if key in ["dtype", "is_expert_param", "requires_grad", "fsdp_unit_id"]
        }
        for param in group.params:
            if _does_param_require_new_bucket(param):
                
                
                
                
                if len(bucket) > 0:
                    
                    bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                
                bucket_groups.append(ParameterGroup([param], **basic_attrs))
                bucket = []
                continue

            
            bucket.append(param)
            
            
            
            
            if (
                group.fsdp_unit_id is None
                and suggested_bucket_size
                and sum([to_local_if_dtensor(p).shape.numel() for p in bucket])
                >= suggested_bucket_size
            ):
                
                
                bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                bucket = []
                continue

        
        if bucket:
            bucket_groups.append(ParameterGroup(bucket, **basic_attrs))

    
    new_bucket_groups = []
    for group in bucket_groups:
        params = sorted(
            group.params, key=lambda p: to_local_if_dtensor(p).shape[1:].numel(), reverse=True
        )
        while len(params) > 0:
            chunk_size_factor = to_local_if_dtensor(params[0]).shape[1:].numel()
            same_factor_params = []
            remaining_params = []
            for param in params:
                param_shape = to_local_if_dtensor(param).shape
                if (
                    param_shape[1:].numel() == chunk_size_factor
                    or (
                        chunk_size_factor % param_shape[1:].numel() == 0
                        and param_shape.numel() % chunk_size_factor == 0
                    )
                    or (param_shape.numel() < chunk_size_factor)
                ):
                    same_factor_params.append(param)
                else:
                    lcm_chunk_size_factor = math.lcm(chunk_size_factor, param_shape[1:].numel())
                    chunk_size_factor = lcm_chunk_size_factor
                    same_factor_params.append(param)
            
            new_bucket_groups.append(
                ParameterGroup(
                    same_factor_params,
                    dtype=group.dtype,
                    is_expert_param=group.is_expert_param,
                    requires_grad=group.requires_grad,
                    fsdp_unit_id=group.fsdp_unit_id,
                    chunk_size_factor=chunk_size_factor,
                )
            )
            params = remaining_params
    bucket_groups = new_bucket_groups

    
    param_to_param_group = {}
    for group_id, group in enumerate(bucket_groups):
        for param in group.params:
            param_to_param_group[param] = group_id

    
    
    
    bucket_to_bucket_group = {}
    
    
    for bucket_id in range(len(bucket_groups)):
        
        bucket_to_bucket_group[bucket_id] = [bucket_id]

    
    
    
    
    if bucket_group_by_fsdp_unit:
        bucket_group_map = {}

        
        for bucket_id, param_group in enumerate(bucket_groups):
            if param_group.fsdp_unit_id is None:
                
                
                
                
                continue
            
            
            id = (param_group.fsdp_unit_id, param_group.is_expert_param)
            if id not in bucket_group_map:
                bucket_group_map[id] = []
            bucket_group_map[id].append(bucket_id)

        
        
        for bucket_group in bucket_group_map.values():
            for bucket_id in bucket_group:
                bucket_to_bucket_group[bucket_id] = bucket_group

    
    
    
    return (bucket_groups, param_to_param_group, bucket_to_bucket_group)


class ParamAndGradBuffer:
    """A class that manages parameter grouping, buffer allocation, and
    communication operations for data-parallel distributed training.

    This class provides functionality to:
    1.  Group parameters based on their data types and communication group sizes.
    2.  Create contiguous buffers for model weights, gradients, and high-precision
        main weights.
    3.  Handle parameter unsharding, gradient reduction, and weight
        synchronization operations.

    Key Features:
        - Efficient parameter grouping based on data types and communication patterns
        - Memory-efficient contiguous buffer allocation
        - Support for mixed-precision training with main weights
        - Distributed operations including parameters all-gather and gradients
            reduce-scatter/all-reduce
        - Synchronized weight updates between model and main weights

    Note:
        This class is designed for distributed training scenarios where efficient
        parameter management and communication are crucial for performance.

    Args:
        ddp_config (DistributedDataParallelConfig): The distributed data parallel
            configuration.
        module (torch.nn.Module): The module whose parameters are to be grouped
            and flatten.
        bucketing_policy (BucketingPolicy): The bucketing policy.
        dist_index (FSDPDistributedIndex): FSDPDistributedIndex object containing references
            to the process groups and device meshes used by Megatron-FSDP.
        mixed_precision_policy (megatron_fsdp.MixedPrecisionPolicy): Configuration for
            mixed-precision customization of compute and communications in Megatron-FSDP.
        gradient_scaling_factor (Optional[float]): The gradient scaling factor.
        expert_gradient_scaling_factor (Optional[float]): The expert gradient
            scaling factor.
        device (torch.device): The parameter and gradient buffer device.
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad (bool):
            Whether to only create the gradient buffer and main weight buffer
            for parameters that require gradients. Default is True.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        bucketing_policy: BucketingPolicy,
        dist_index: FSDPDistributedIndex,
        mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
        gradient_scaling_factor: Optional[float] = None,
        expert_gradient_scaling_factor: Optional[float] = None,
        device: torch.device = torch.device("cuda"),
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad: bool = True,
        reset_parameters_for_meta_device_init_module: bool = False,
    ):
        
        
        bucketing_policy.data_parallel_sharding_strategy = (
            ddp_config.data_parallel_sharding_strategy
        )

        self.ddp_config = ddp_config
        self.use_decoupled_grad = ddp_config.megatron_fsdp_use_decoupled_grad
        self.module = module
        self.bucketing_policy = bucketing_policy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.mp_policy = mixed_precision_policy
        self.dist_index = dist_index
        self.params = list(module.parameters())
        self.gradient_scaling_factor = gradient_scaling_factor
        self.expert_gradient_scaling_factor = expert_gradient_scaling_factor
        self.device = device
        self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad = (
            only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad
        )
        self.reset_parameters_for_meta_device_init_module = (
            reset_parameters_for_meta_device_init_module
        )
        self.ubr_groups = None
        self.already_registered = False
        
        if self.ddp_config.nccl_ub:
            assert nccl_allocator is not None, (
                "To use user buffer registration, "
                "either requires megatron.core.nccl_allocator or apex.contrib.nccl_allocator"
            )
            
            
            self.ddp_config.fsdp_double_buffer = True
            
            global NCCL_MEMORY_POOL
            
            nccl_allocator.init()
            NCCL_MEMORY_POOL = nccl_allocator.create_nccl_mem_pool(
                symmetric=not self.ddp_config.disable_symmetric_registration
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[Rank {torch.distributed.get_rank()}] Created NCCL memory pool for "
                "UserBuffer Registration",
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[Rank {torch.distributed.get_rank()}] FSDP double buffer is enabled.",
            )
            
            self.ubr_groups = [self.dist_index.get_fsdp_group(is_expert_parallel=False)]
            if self.dist_index.get_fsdp_group(is_expert_parallel=True) is not None:
                
                self.ubr_groups.append(self.dist_index.get_fsdp_group(is_expert_parallel=True))
            if (
                self.dist_index.get_fsdp_group(
                    is_expert_parallel=False, independent_all_gather=True
                )
                is not None
            ):
                
                self.ubr_groups.append(
                    self.dist_index.get_fsdp_group(
                        is_expert_parallel=False, independent_all_gather=True
                    )
                )
            if (
                self.dist_index.get_fsdp_group(is_expert_parallel=True, independent_all_gather=True)
                is not None
            ):
                
                self.ubr_groups.append(
                    self.dist_index.get_fsdp_group(
                        is_expert_parallel=True, independent_all_gather=True
                    )
                )
            if self.dist_index.get_outer_fsdp_group() is not None:
                
                self.ubr_groups.append(self.dist_index.get_outer_fsdp_group())

            log_single_rank(
                logger,
                logging.INFO,
                f"[ParamAndGradBuffer] FSDP UBRegistration Groups ({len(self.ubr_groups)}):",
            )
            
            for i, group in enumerate(self.ubr_groups):
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Group [{i+1}/{len(self.ubr_groups)}] "
                    f"group.group_desc: {group.group_desc}, group.size(): {group.size()}",
                )
                torch.distributed.barrier(group=group, async_op=False)
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Call Success with the group [{i+1}/{len(self.ubr_groups)}] "
                    f"group.group_desc: {group.group_desc}",
                )
            
            torch.distributed.barrier(async_op=False)
            log_single_rank(logger, logging.INFO, "Call Success with the global communicator group")

        
        
        
        self.mem_alloc_context = self.get_mem_alloc_context(
            groups=self.ubr_groups, symmetric=not self.ddp_config.disable_symmetric_registration
        )

        
        meta_device_init_fp8_params = {}
        if reset_parameters_for_meta_device_init_module and HAVE_TE:
            for m in module.modules():
                if not isinstance(m, TransformerEngineBaseModule):
                    continue
                for name, param in m.named_parameters(recurse=False):
                    
                    
                    
                    fp8_meta_index = m.param_init_meta[name].fp8_meta_index
                    if m.primary_weights_in_fp8 and fp8_meta_index is not None:
                        meta_device_init_fp8_params[self.param_to_name[param]] = (
                            True,
                            fp8_need_transpose_data_for_meta_device_init(m),
                        )

        
        (self.parameter_groups, self.param_to_param_group, self.bucket_to_bucket_group) = (
            _get_parameter_groups(module, bucketing_policy, meta_device_init_fp8_params)
        )
        self._init_each_parameter_group_buffers(meta_device_init_fp8_params)
        self._init_distributed_params()

        
        self.optimizer_named_parameters = self._init_optimizer_named_parameters()

        self._log_parameter_groups()

    def get_mem_alloc_context(self, groups=None, symmetric=True):
        """
        Get the memory allocation context for the parameter and gradient buffers.
        """
        if self.ddp_config.nccl_ub:
            assert nccl_allocator is not None, (
                "To use user buffer registration, "
                "either requires megatron.core.nccl_allocator or apex.contrib.nccl_allocator"
            )
            global NCCL_MEMORY_POOL
            if groups is None:
                
                groups = [self.dist_index.get_fsdp_group(is_expert_parallel=False)]

            if NCCL_ALLOCATOR == "MCORE":
                if self.ddp_config.fsdp_manual_registration:
                    return functools.partial(
                        nccl_allocator.MemPoolAllocatorWithoutRegistration, NCCL_MEMORY_POOL
                    )
                if len(groups) == 1:
                    
                    mem_alloc_context = functools.partial(
                        nccl_allocator.nccl_mem,
                        NCCL_MEMORY_POOL,
                        group=groups[0],
                        symmetric=symmetric,
                    )
                else:
                    mem_alloc_context = functools.partial(
                        nccl_allocator.MultiGroupMemPoolAllocator,
                        NCCL_MEMORY_POOL,
                        groups=groups,
                        symmetric=symmetric,
                    )
            elif NCCL_ALLOCATOR == "APEX":
                if self.ddp_config.fsdp_manual_registration:
                    logging.warning(
                        "FSDP manual registration is not supported for APEX NCCL allocator."
                        "falling back to default registration. "
                        "Please use Megatron Core NCCL allocator for manual registration."
                    )
                if symmetric:
                    logging.warning(
                        "Symmetric registration is not supported for APEX NCCL allocator."
                        "falling back to non-symmetric registration. "
                        "Please use Megatron Core NCCL allocator for symmetric registration."
                    )

                if len(groups) == 1:
                    
                    mem_alloc_context = functools.partial(
                        nccl_allocator.nccl_mem, NCCL_MEMORY_POOL, group=groups[0]
                    )
                else:
                    
                    mem_alloc_context = functools.partial(
                        MultiGroupUBRAllocator, NCCL_MEMORY_POOL, groups=groups
                    )
            else:
                raise ValueError(f"Invalid NCCL allocator: {NCCL_ALLOCATOR}")
            return mem_alloc_context
        else:
            return nullcontext

    def manual_buffer_registration(self):
        """
        Manually register the FSDP communication buffers to NCCL user buffer.
        """
        assert self.ddp_config.nccl_ub, "NCCL UBR is not enabled"
        assert self.ddp_config.fsdp_double_buffer, "FSDP double buffer is not enabled"
        assert self.ddp_config.fsdp_manual_registration, "FSDP manual registration is not enabled"
        assert not self.already_registered, "Mem pool is already registered"

        self.already_registered = True

        global NCCL_MEMORY_POOL
        torch.cuda.synchronize()
        torch.distributed.barrier(async_op=False)
        torch.cuda.synchronize()

        for group in self.ubr_groups:
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][FSDP][Manual REG] Registering mem pool to group {group},"
                f"group.group_desc:{group.group_desc}, group.size(): {group.size()}",
            )
            nccl_allocator.register_mem_pool(
                NCCL_MEMORY_POOL,
                group,
                symmetric=not self.ddp_config.disable_symmetric_registration,
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][FSDP][Manual REG] Registered mem pool to group {group},"
                f"group.group_desc:{group.group_desc}, group.size(): {group.size()}",
            )

    def _log_parameter_groups(self):
        """Compact log of FSDP parameter groups and their parameters."""

        def _bytes_to_mb(bytes_val: int) -> str:
            return f"{bytes_val / 1_000_000:.2f} MB"

        bucket_groups = self.parameter_groups
        param_to_name = self.param_to_name
        total_padded_bytes = 0
        total_comm_bytes = 0
        log_lines = [f"Number of FSDP Parameter Groups: {len(bucket_groups)}"]

        for idx, group in enumerate(bucket_groups):
            numel = sum(to_local_if_dtensor(p).shape.numel() for p in group.params)
            buffers = {
                "weight": group.model_weight_buffer,
                "transpose_weight": group.transpose_weight_buffer,
                "main_weight": group.main_weight_buffer,
                "grad": group.main_grad_buffer,
            }
            group_padded = 0
            group_comm = 0
            buf_flags = []
            for k, buf in buffers.items():
                if buf:
                    elem_size = _dtype_size(buf.dtype)
                    group_padded += (buf.bucket_index.size - numel) * elem_size
                    group_comm += buf.bucket_index.size * elem_size
                    buf_flags.append(k)
            total_padded_bytes += group_padded
            total_comm_bytes += group_comm

            
            log_lines.append(
                f"[FSDP_UNIT {group.fsdp_unit_id}] Group {idx}: elems={numel} dtype={group.dtype} "
                f"bufs={','.join(buf_flags) or 'None'} pad={_bytes_to_mb(group_padded)}"
            )
            
            for param in group.params:
                log_lines.append(f"\t{param_to_name[param]} {tuple(param.shape)}")

        
        log_lines.append(
            f"Total comm: {_bytes_to_mb(total_comm_bytes)}, "
            f"Total pad: {_bytes_to_mb(total_padded_bytes)}"
        )

        log_single_rank(logger, logging.INFO, "\n".join(log_lines))

    def _init_each_parameter_group_buffers(self, meta_device_init_fp8_params):
        """
        Initialize the buffers for each parameter group.
        """
        
        data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        if data_parallel_sharding_strategy == "no_shard":
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = False
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == "optim":
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == "optim_grads":
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        elif data_parallel_sharding_strategy == "optim_grads_params":
            is_model_weight_buffer_distributed = True
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        else:
            raise ValueError(
                f"Invalid data_parallel_sharding_strategy: {data_parallel_sharding_strategy}"
            )

        """
        Hybrid FSDP (HFSDP) helper buffers for outer-DP optimizer-state sharding.

        Design goal
        ==========
        This design extends Megatron-FSDP's Hybrid / Fully Sharded Data Parallelism
        to support *outer* data-parallel (DP) optimizer-state sharding, without
        complicating the per-rank model weight / grad views that are used by the
        forward and backward passes.

        Core idea
        ==========
        We introduce two(or three) persistent helper buffers:

        - `hfsdp_helper_wbuf`: stores the *true* persistent parameter payload
            in the inner-DP layout (Inner-DP param buffer).
        - `hfsdp_helper_gbuf`: stores the *true* persistent gradient payload
            in the inner-DP layout (Inner-DP grad buffer).
        - `hfsdp_helper_wtbuf`: (optional) stores transpose weights in the inner-DP
            layout for FP8 parameters that require transposition for efficient
            mixed-precision matmuls.

        These buffers own the real storage for parameters and gradients that
        participate in HFSDP sharding. The existing model `weight` buffer and
        `gradient` buffer are simplified to be pure "data-parallel buffers"
        defined only over the DP dimension, and can alias (view into) the
        helper buffers. In other words:

        - Helper buffers: inner-DP-aware, persistent, sharded storage.
        - Model buffers: outer-DP-oriented data-parallel views used for compute,
            and to give the optimizer access to the relevant shards of the helper
            buffers when needed.

        By separating **storage** (helper buffers) from **compute views**
        (model buffers), we can:

        - Keep the model-side DP buffers conceptually simple (they do not need to
            encode Inner-DP vs Outer-DP tiling).
        - Implement fully sharded optimizer states over the DP mesh by sharding
            optimizer states consistently with the model buffers and helper buffers.
        - Control when and how data is synchronized between inner and outer DP
            dimensions on each iteration.

        Data flow per iteration
        =======================
        Compared to the usual Hybrid FSDP data flow (where the last micro-batch
        backward issues a DP all-reduce on gradients), this design explicitly
        uses parameter all-gather and gradient reduce-scatter between the DP and
        inner-DP layouts:

        1. Parameters:
            - Persistent parameter shards live in `hfsdp_helper_wbuf` in inner-DP
                layout.
            - At the beginning of each iteration (before the first micro-batch
                forward), we all-gather DP-sharded parameters to form the inner-DP
                parameter shards in `hfsdp_helper_wbuf`, following the standard
                Hybrid FSDP pattern of making shards "bigger" for compute.
            - The model weight buffer is set up as a DP-only view on top of
                these inner-DP shards. It does not own persistent storage.

        2. Gradients:
            - During backward for each micro-batch, gradients are accumulated into
                the `hfsdp_helper_gbuf` in inner-DP layout.
            - On the last micro-batch backward of the iteration, instead of a DP
                all-reduce, we perform a reduce-scatter that maps the DP gradient
                layout back into outer-DP gradient shards.
            - Because the model gradient buffer is a view of `hfsdp_helper_gbuf`
                in the current design, the reduced / scattered results effectively
                update both the helper buffer and the model-gradient-buffer view.
            - The optimizer then reads gradients from model-gradient-buffer view
                (DP layout) to perform the update.

        3. Optimizer states:
            - Optimizer states are constructed and kept sharded in the same DP /
                inner-DP pattern as the helper buffers and their model-buffer views.
            - Because parameters and gradients are stored persistently in helper
                buffers and are sharded over DP, the optimizer only ever touches
                fully sharded tensors. This enables *fully sharded* optimizer
                states on the DP group (outer-DP sharding in HFSDP).
            - After the optimizer step, updated parameter shards remain in
                `hfsdp_helper_wbuf`. At the beginning of the next iteration, the
                usual all-gather path re-exposes these updated outer-DP shards
                through the model weight buffer.

        Implementation details
        ======================
        - `hfsdp_helper_wbuf` / `hfsdp_helper_gbuf` (and optionally
            `hfsdp_helper_wtbuf`) are allocated as the canonical storage for all
            HFSDP-managed parameters / gradients. They encode the inner-DP
            partitioning and are aligned with the DP device mesh used for HFSDP
            optimizer sharding.

        - The existing Megatron-FSDP weight and grad buffers are repurposed as
            *outer-DP data-parallel buffers*. They:
            - Have a shape / layout that only reflects the DP dimension.
            - Implemented as views into the helper buffers to avoid extra copies.
            - Serve as the interface tensors that the optimizer code reads from
                and writes to.

        - Synchronization between helper buffers and model buffers is explicit:
            - "param sync" path:
                - At initialization and at the beginning of each iteration, parameters
                are exposed to the model buffer by all-gathering DP-sharded
                parameters into inner-DP shards in `hfsdp_helper_wbuf` and then
                viewing them through the model weight buffer.
            - "grad sync" path:
                - At the last micro-batch backward, we reduce-scatter gradients from
                the inner-DP layout into model gradient buffer as DP shards. Because
                the model gradient buffer is a view of `hfsdp_helper_gbuf`, the
                reduced results are immediately visible through the model gradient buffer
                as well.

        - Outer-DP optimizer-state sharding:
            - Optimizer state tensors are allocated with the same sharding pattern
                as the model buffers along the DP dimension. Each rank only owns the
                local shard of:
                * its parameters (from `model weight buffer`),
                * its gradients (from `model gradient buffer`),
                * and the corresponding optimizer states.
            - This is conceptually similar to ZeRO-style optimizer sharding, but
                implemented on top of Megatron-FSDP's buffer / device-mesh abstractions,
                using the helper buffers as the single source of truth for persistent
                data.

        Notes for maintainers
        =====================
        - When adding new parameters to HFSDP, register them with the helper
            buffers first. The model weight / grad DP buffers should be treated as
            *views* for compute, not as owners of persistent storage.

        - When changing the DP mesh, inner/outer-DP dimension mapping, or the
            sharding strategy, verify that:
            - The helper buffer partitioning matches the intended HFSDP sharding
                (inner-DP).
            - The optimizer state partitioning is kept consistent with the model
                weight and gradient buffer sharding.
            - The synchronization paths correctly map between inner-DP layout
                (helper buffers) and DP-only layout (model buffers), including the
                parameter all-gather at the beginning of the iteration and the
                gradient reduce-scatter on the last micro-batch backward.

        - Any logic that assumes the model weight or gradient buffers own
            persistent data should be updated to read/write from the helper
            buffers instead. The model buffers are intentionally simplified to
            keep the HFSDP optimizer sharding logic centralized in the helper
            layer.
        """
        should_create_hfsdp_helper_buffers = (
            self.dist_index.use_hybrid_fsdp
            and self.ddp_config.outer_dp_sharding_strategy != "no_shard"
        )
        
        
        if (
            should_create_hfsdp_helper_buffers
            and self.ddp_config.data_parallel_sharding_strategy != "optim_grads_params"
        ):
            raise NotImplementedError(
                "[Megatron-FSDP] Optimizer fully-sharded HFSDP is only supported "
                "with full-sharding on DP-Shard.\nMegatron-FSDP DP-Shard Strategy: "
                f"{self.ddp_config.data_parallel_sharding_strategy}\nMegatron-FSDP "
                f"DP-Outer Strategy: {self.ddp_config.outer_dp_sharding_strategy}"
            )

        if self.ddp_config.nccl_ub:
            assert self.ddp_config.fsdp_double_buffer, (
                "NCCL UB is only supported with FSDP double buffer. "
                "Please set fsdp_double_buffer=True in the ddp config."
            )
        if self.ddp_config.fsdp_double_buffer and len(self.bucketing_policy.fsdp_unit_modules) > 0:
            UB_BUFFER_NUM = 2
            self.weight_alloc = FixedPoolAllocator(
                name="fsdp_params",
                fsdp_param_groups=self.parameter_groups,
                size=UB_BUFFER_NUM,
                fallback_to_persistent_buffer=self.ddp_config.fsdp_db_use_persist_buf_on_alloc_fail,
            )
            self.transpose_weight_alloc = FixedPoolAllocator(
                name="fsdp_fp8_transpose_params",
                fsdp_param_groups=self.parameter_groups,
                size=UB_BUFFER_NUM,
                fallback_to_persistent_buffer=self.ddp_config.fsdp_db_use_persist_buf_on_alloc_fail,
            )
            self.main_grad_alloc = FixedPoolAllocator(
                name="fsdp_grads",
                fsdp_param_groups=self.parameter_groups,
                size=UB_BUFFER_NUM,
                fallback_to_persistent_buffer=(
                    self.ddp_config.fsdp_db_use_persist_buf_on_alloc_fail
                ),
            )
            if self.dist_index.use_hybrid_fsdp:
                
                
                
                
                self.hsdp_grad_comm_alloc = FixedPoolAllocator(
                    name="hsdp_grad_comm",
                    fsdp_param_groups=self.parameter_groups,
                    size=UB_BUFFER_NUM,
                    fallback_to_persistent_buffer=(
                        self.ddp_config.fsdp_db_use_persist_buf_on_alloc_fail
                    ),
                )
            self.double_buf_units = self.weight_alloc.fsdp_double_buffer_units
        else:
            self.weight_alloc = StorageResizeBasedBucketAllocator()
            self.transpose_weight_alloc = StorageResizeBasedBucketAllocator()
            self.main_grad_alloc = None
            if self.dist_index.use_hybrid_fsdp:
                
                
                
                self.hsdp_grad_comm_alloc = None
            self.double_buf_units = []

        self.buffer_all_in_one = True
        buffer_size = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}

        
        for group_id, group in enumerate(self.parameter_groups):
            main_buf_extra_kwargs = {}
            if should_create_hfsdp_helper_buffers:
                
                main_buf_dp_group = self.dist_index.get_dp_group(
                    is_expert_parallel=group.is_expert_param
                )
                
                inner_dp_group = self.dist_index.get_fsdp_group(
                    is_expert_parallel=group.is_expert_param
                )
                main_buf_extra_kwargs["dp_rank"] = self.dist_index.get_logical_hybrid_fsdp_rank(
                    is_expert_parallel=group.is_expert_param
                )
            else:
                
                main_buf_dp_group = self.dist_index.get_fsdp_group(
                    is_expert_parallel=group.is_expert_param
                )

            
            
            
            
            model_wbuf_dp_group = main_buf_dp_group
            if not should_create_hfsdp_helper_buffers:
                ag_group = self.dist_index.get_fsdp_group(
                    is_expert_parallel=group.is_expert_param, independent_all_gather=True
                )
                if ag_group is not None:
                    model_wbuf_dp_group = ag_group

            gradient_scaling_factor = (
                self.gradient_scaling_factor
                if not group.is_expert_param
                else self.expert_gradient_scaling_factor
            )
            
            one_param = group.params[0]
            is_dtype_float8 = (
                is_float8tensor(one_param)
                or meta_device_init_fp8_params.get(self.param_to_name[one_param], (False, False))[0]
            )

            
            if is_dtype_float8:
                param_dtype = torch.uint8
                main_grads_dtype = torch.bfloat16
            else:
                param_dtype = group.params[0].dtype
                main_grads_dtype = param_dtype
            
            if self.mp_policy.main_grads_dtype is not None:
                main_grads_dtype = self.mp_policy.main_grads_dtype

            
            
            need_transpose_data = is_float8tensor(one_param) and fp8_need_transpose_data(one_param)
            need_transpose_data_for_meta_device_init = meta_device_init_fp8_params.get(
                self.param_to_name[one_param], (False, False)
            )[1]
            should_create_transpose_weight_buffer = (
                need_transpose_data or need_transpose_data_for_meta_device_init
            )

            
            should_create_grad_buffer_or_main_weight_buffer = (
                not self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad
                or group.requires_grad
            )

            
            if data_parallel_sharding_strategy != "no_shard":
                group.model_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_model_weight_buffer_distributed
                    and model_wbuf_dp_group.size() > 1,
                    dtype=param_dtype,
                    device=self.device,
                    
                    
                    
                    data_parallel_group=model_wbuf_dp_group,
                    is_transpose_buffer=False,
                    temporary_bucket_allocator=self.weight_alloc,
                    bucket_id=group_id,
                    chunk_size_factor=group.chunk_size_factor,
                    mem_alloc_context=self.mem_alloc_context,
                    **main_buf_extra_kwargs,
                )
                if should_create_transpose_weight_buffer:
                    group.transpose_weight_buffer = DataParallelBuffer(
                        self.ddp_config,
                        group.params,
                        is_data_distributed=is_model_weight_buffer_distributed
                        and main_buf_dp_group.size() > 1,
                        dtype=param_dtype,
                        device=self.device,
                        data_parallel_group=main_buf_dp_group,
                        is_transpose_buffer=True,
                        temporary_bucket_allocator=self.transpose_weight_alloc,
                        bucket_id=group_id,
                        chunk_size_factor=group.chunk_size_factor,
                        mem_alloc_context=self.mem_alloc_context,
                        **main_buf_extra_kwargs,
                    )

            
            
            if (
                should_create_grad_buffer_or_main_weight_buffer
                and self.mp_policy.main_params_dtype is not None
            ):
                group.main_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_main_weight_buffer_distributed
                    and main_buf_dp_group.size() > 1,
                    dtype=self.mp_policy.main_params_dtype,
                    device=self.device,
                    data_parallel_group=main_buf_dp_group,
                    bucket_id=group_id,
                    chunk_size_factor=group.chunk_size_factor,
                    mem_alloc_context=self.mem_alloc_context,
                    **main_buf_extra_kwargs,
                )

            
            if should_create_grad_buffer_or_main_weight_buffer:
                assert (
                    main_grads_dtype.is_floating_point
                ), f"Main gradient dtype ({main_grads_dtype}) must be Float."
                group.main_grad_buffer = DataParallelBuffer(
                    self.ddp_config,
                    
                    
                    group.params,
                    is_data_distributed=is_grad_buffer_distributed and main_buf_dp_group.size() > 1,
                    dtype=main_grads_dtype,
                    device=self.device,
                    
                    
                    
                    data_parallel_group=main_buf_dp_group,
                    is_transpose_buffer=False,
                    temporary_bucket_allocator=self.main_grad_alloc,
                    gradient_scaling_factor=gradient_scaling_factor,
                    bucket_id=group_id,
                    chunk_size_factor=group.chunk_size_factor,
                    mem_alloc_context=self.mem_alloc_context,
                    **main_buf_extra_kwargs,
                )

                
                buffer_size[group.main_grad_buffer.dtype] += group.main_grad_buffer.data_size

            
            if should_create_hfsdp_helper_buffers:
                
                wbuf = group.model_weight_buffer
                group.hfsdp_helper_wbuf = _create_hfsdp_helper_buffer(
                    group.model_weight_buffer,
                    inner_dp_group=inner_dp_group,
                    is_data_distributed=is_main_weight_buffer_distributed
                    and inner_dp_group.size() > 1,
                )

                if group.transpose_weight_buffer is not None:
                    group.hfsdp_helper_wtbuf = _create_hfsdp_helper_buffer(
                        group.transpose_weight_buffer,
                        inner_dp_group=inner_dp_group,
                        is_data_distributed=is_main_weight_buffer_distributed
                        and inner_dp_group.size() > 1,
                    )

                if should_create_grad_buffer_or_main_weight_buffer:
                    group.hfsdp_helper_gbuf = _create_hfsdp_helper_buffer(
                        group.main_grad_buffer,
                        inner_dp_group=inner_dp_group,
                        is_data_distributed=is_grad_buffer_distributed
                        and inner_dp_group.size() > 1,
                    )
                    buffer_size[group.main_grad_buffer.dtype] -= group.main_grad_buffer.data_size
                    buffer_size[group.main_grad_buffer.dtype] += group.hfsdp_helper_gbuf.data_size

            
            if should_create_grad_buffer_or_main_weight_buffer and self.dist_index.use_hybrid_fsdp:
                
                
                
                
                
                gbuf = group.main_grad_buffer
                fsdp_group = self.dist_index.get_fsdp_group(
                    is_expert_parallel=group.is_expert_param
                )
                hfsdp_kwargs = {}
                if should_create_hfsdp_helper_buffers:
                    hfsdp_kwargs["item_index_map"] = gbuf.item_index_map
                    hfsdp_kwargs["bucket_index"] = gbuf.bucket_index
                    hfsdp_kwargs["shard_bucket_index"] = _get_dp_buffer_shard_bucket_index(
                        gbuf.bucket_index,
                        is_data_distributed=is_grad_buffer_distributed and fsdp_group.size() > 1,
                        data_parallel_world_size=fsdp_group.size(),
                        data_parallel_rank=fsdp_group.rank(),
                    )
                group.hsdp_comm_gbuf = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_grad_buffer_distributed and fsdp_group.size() > 1,
                    
                    dtype=self.mp_policy.grad_comm_dtype,
                    device=gbuf.device,
                    data_parallel_group=fsdp_group,
                    is_transpose_buffer=False,
                    
                    temporary_bucket_allocator=self.hsdp_grad_comm_alloc,
                    bucket_id=group_id,
                    chunk_size_factor=group.chunk_size_factor,
                    mem_alloc_context=self.mem_alloc_context,
                    **hfsdp_kwargs,
                )

        reset_context_args = {"init_param_with_fp8": self.ddp_config.fp8_param_gather}
        module_reset_flag = {}
        if self.reset_parameters_for_meta_device_init_module:
            self.param_to_direct_module = {}
            for name, m in self.module.named_modules():
                for p in m.parameters(recurse=False):
                    self.param_to_direct_module[p] = (name, m)

            meta_params_numel = 0
            cuda_params_numel = 0
            cpu_params_numel = 0
            for group in self.parameter_groups:
                for p in group.params:
                    p_numel = to_local_if_dtensor(p).shape.numel()
                    if p.is_meta:
                        meta_params_numel += p_numel
                    elif p.device.type == "cuda":
                        cuda_params_numel += p_numel
                    else:
                        cpu_params_numel += p_numel
            log_str = (
                f"Meta params numel: {meta_params_numel / 1_000_000:.2f} M, "
                f"CUDA params numel: {cuda_params_numel / 1_000_000:.2f} M, "
                f"CPU params numel: {cpu_params_numel / 1_000_000:.2f} M"
            )
            log_single_rank(logger, logging.INFO, log_str)

        
        
        
        
        for group in self.parameter_groups:
            wbuf = group.model_weight_buffer
            if wbuf:
                with self.mem_alloc_context():
                    if group.hfsdp_helper_wbuf:
                        _init_hfsdp_helper_and_dp_buffer_data(
                            group.hfsdp_helper_wbuf,
                            wbuf,
                            mem_alloc=lambda size, dtype: torch.empty(
                                size, dtype=dtype, device=self.device
                            ),
                            outer_dp_group=self.dist_index.get_outer_fsdp_group(
                                is_expert_parallel=group.is_expert_param
                            ),
                        )
                    else:
                        
                        wbuf.init_data(
                            torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device)
                        )
                bucket = wbuf.fetch_bucket()

            tbuf = group.transpose_weight_buffer
            if tbuf:
                with self.mem_alloc_context():
                    if group.hfsdp_helper_wbuf:
                        _init_hfsdp_helper_and_dp_buffer_data(
                            group.hfsdp_helper_wtbuf,
                            tbuf,
                            mem_alloc=lambda size, dtype: torch.empty(
                                size, dtype=dtype, device=self.device
                            ),
                            outer_dp_group=self.dist_index.get_outer_fsdp_group(
                                is_expert_parallel=group.is_expert_param
                            ),
                        )
                    else:
                        
                        tbuf.init_data(
                            torch.empty(tbuf.data_size, dtype=tbuf.dtype, device=self.device)
                        )
                transpose_bucket = tbuf.fetch_bucket()

            mbuf = group.main_weight_buffer
            if mbuf:
                
                mbuf.init_data(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
            for item_id, p in enumerate(group.params):
                
                if wbuf:
                    if self.reset_parameters_for_meta_device_init_module and p.is_meta:
                        m_name, m = self.param_to_direct_module[p]
                        
                        
                        
                        if not hasattr(m, "reset_parameters") and not hasattr(
                            m, "_reset_parameters"
                        ):
                            raise ValueError(
                                f"[init_model_with_meta_device=True] Module {m_name} "
                                "does not have a reset_parameters or _reset_parameters method."
                            )
                        if not module_reset_flag.get(m_name, False):
                            old_params = list(m.parameters(recurse=False))

                            
                            
                            
                            if check_gpu_memory(threshold=0.5):
                                gc.collect()
                                torch.cuda.empty_cache()

                            m.to_empty(device=self.device, recurse=False)
                            if (
                                HAVE_TE
                                and is_te_min_version("0.9.0")
                                and not isinstance(m, TransformerEngineBaseModule)
                            ):
                                reset_context_args["with_cuda_rng_tracker"] = True
                            with ResetParametersContext(**reset_context_args):
                                
                                if hasattr(m, "reset_parameters"):
                                    m.reset_parameters()
                                elif hasattr(m, "_reset_parameters"):
                                    m._reset_parameters()
                            module_reset_flag[m_name] = True
                            new_params = list(m.parameters(recurse=False))

                            self._reset_parameters(old_params, new_params)
                            p = group.params[item_id]

                            
                            
                            if not self.ddp_config.keep_fp8_transpose_cache:
                                for _param in m.parameters(recurse=False):
                                    if is_float8tensor(_param):
                                        fp8_discard_transpose_cache(_param)
                    
                    assert not p.is_meta, (self.param_to_name[p], module_reset_flag)

                    p_local = to_local_if_dtensor(p)

                    
                    
                    wbuf.set_item(item_id, p_local)
                    if tbuf:
                        tbuf.set_item(item_id, p_local)

                    
                    
                    
                    new_param_data = wbuf.get_item_from_bucket(bucket, item_id).view(p_local.shape)
                    if tbuf:
                        new_transpose_data = tbuf.get_item_from_bucket(
                            transpose_bucket, item_id
                        ).view(p_local.shape)
                    else:
                        new_transpose_data = None

                    if is_float8tensor(p_local):
                        
                        
                        old_param_data = fp8_get_raw_data(p_local)
                        assert old_param_data._base is None
                        new_param_data.detach().copy_(old_param_data)
                        fp8_set_raw_data(p_local, new_param_data)
                        del old_param_data
                        if new_transpose_data is not None:
                            
                            
                            old_transpose_data = fp8_get_raw_data(p_local, True)
                            assert old_transpose_data._base is None
                            new_transpose_data.detach().copy_(old_transpose_data)
                            fp8_set_raw_data(p_local, new_transpose_data, True)
                            del old_transpose_data
                    elif isinstance(p, DTensor):
                        
                        
                        old_param_data = p._local_tensor.data
                        p._local_tensor.data = new_param_data
                        assert old_param_data._base is None
                        p._local_tensor.data.detach().copy_(old_param_data)
                        del old_param_data
                    else:
                        
                        
                        

                        
                        
                        
                        
                        old_param_data = p.data
                        p.data = new_param_data
                        assert old_param_data._base is None
                        p.data.detach().copy_(old_param_data)
                        del old_param_data

                
                if mbuf:
                    if hasattr(p, "get_high_precision_init_val"):
                        assert not isinstance(p, DTensor), (
                            self.param_to_name[p],
                            "not support fp8 DTensor.",
                        )
                        
                        
                        mbuf.set_item(item_id, p.get_high_precision_init_val())
                        p.clear_high_precision_init_val()
                    else:
                        
                        
                        
                        
                        p_local = to_local_if_dtensor(p)
                        assert not is_float8tensor(p_local), (
                            self.param_to_name[p],
                            "fp8 param should use get_high_precision_init_val method.",
                        )
                        mbuf.set_item(item_id, p_local)

            if wbuf and wbuf.is_data_distributed:
                
                
                
                
                
                
                
                wbuf.free_bucket_storage()

            if tbuf and tbuf.is_data_distributed:
                tbuf.free_bucket_storage()

        
        if self.buffer_all_in_one:
            with self.mem_alloc_context():
                self.buffer = {
                    torch.float32: torch.empty(
                        buffer_size[torch.float32], dtype=torch.float32, device=self.device
                    ),
                    torch.float16: torch.empty(
                        buffer_size[torch.float16], dtype=torch.float16, device=self.device
                    ),
                    torch.bfloat16: torch.empty(
                        buffer_size[torch.bfloat16], dtype=torch.bfloat16, device=self.device
                    ),
                    "float8": torch.empty(
                        buffer_size["float8"], dtype=torch.uint8, device=self.device
                    ),
                }
            offset = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}

        def _alloc(dtype, size):
            """
            If using a single buffer for main model weights and gradients,
            allocate memory per dtype buffer with size at the current offset.
            Return the allocated slice of the buffer data Tensor.

            If not using a single buffer, then return an empty Tensor on this device.
            """
            if self.buffer_all_in_one:
                if dtype == torch.uint8:
                    dtype = "float8"
                data = self.buffer[dtype][offset[dtype] : offset[dtype] + size]
                offset[dtype] += size
                return data
            return torch.empty(size, dtype=dtype, device=self.device)

        
        for group in self.parameter_groups:
            gbuf = group.main_grad_buffer
            if not gbuf:
                
                continue
            
            with self.mem_alloc_context():
                if group.hfsdp_helper_gbuf:
                    _init_hfsdp_helper_and_dp_buffer_data(
                        group.hfsdp_helper_gbuf,
                        gbuf,
                        mem_alloc=_alloc,
                        outer_dp_group=self.dist_index.get_outer_fsdp_group(
                            is_expert_parallel=group.is_expert_param
                        ),
                    )
                    group.hfsdp_helper_gbuf.data.zero_()
                else:
                    
                    gbuf.init_data(_alloc(gbuf.dtype, gbuf.data_size))
                    gbuf.data.zero_()
            for item_id, p in enumerate(group.params):
                
                p._gbuf = group.hfsdp_helper_gbuf if group.hfsdp_helper_gbuf else gbuf
                p._item_id = item_id

                def main_grad_getter(p):
                    
                    
                    
                    
                    
                    bucket = p._gbuf.fetch_bucket(
                        dtype=(
                            self.mp_policy.grad_comm_dtype if p._gbuf.is_data_distributed else None
                        )
                    )
                    gbuf = p._gbuf
                    item_id = p._item_id
                    
                    
                    return gbuf.get_item_from_bucket(bucket, item_id).view(
                        to_local_if_dtensor(p).shape
                    )

                
                
                
                
                
                p.get_main_grad = main_grad_getter.__get__(p)

        
        gc.collect()
        torch.cuda.empty_cache()

    def _reset_parameters(self, old_params, new_params):
        assert len(old_params) == len(new_params)
        param_map = {}
        for old_param, new_param in zip(old_params, new_params):
            param_map[old_param] = new_param
            self.param_to_name[new_param] = self.param_to_name[old_param]
            del self.param_to_name[old_param]

            self.param_to_param_group[new_param] = self.param_to_param_group[old_param]
            del self.param_to_param_group[old_param]

            self.param_to_direct_module[new_param] = self.param_to_direct_module[old_param]
            del self.param_to_direct_module[old_param]

            new_param.requires_grad_(old_param.requires_grad)

            for tp_attr in ["_tensor_parallel_mode"]:
                if getattr(old_param, tp_attr, None) is not None:
                    setattr(new_param, tp_attr, getattr(old_param, tp_attr))

            
            
            setattr(
                new_param,
                'skip_backward_post_hook',
                getattr(old_param, 'skip_backward_post_hook', False),
            )
        for item_id, p in enumerate(self.params):
            if p in param_map:
                new_p = param_map[p]
                self.params[item_id] = new_p

        for group in self.parameter_groups:
            for item_id, p in enumerate(group.params):
                if p not in param_map:
                    continue
                new_p = param_map[p]
                group.params[item_id] = new_p
                for buf in [
                    group.model_weight_buffer,
                    group.transpose_weight_buffer,
                    group.main_weight_buffer,
                    group.main_grad_buffer,
                    group.hfsdp_helper_wbuf,
                    group.hfsdp_helper_wtbuf,
                    group.hfsdp_helper_gbuf,
                ]:
                    if buf is None:
                        continue
                    buf.param_idx[new_p] = buf.param_idx[p]
                    del buf.param_idx[p]

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        for group in self.parameter_groups:
            if group.main_grad_buffer is None:
                continue
            group.main_grad_buffer.data *= scaling_factor
        self.update_main_grads()

    def zero_grad(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation
        for the next iteration of training.

        Gradient shards are dereferenced to free memory. However, dereferencing is
        not compatible with (FWD-BWD / full-iteration) CUDA graph-ability, because
        we need to preserve this reference to the sharded gradient generated during
        CUDA graph replay (`setattr` in `update_main_grads` not executed during
        CUDA graph replay, as it is not a CUDA kernel).

        If the gradient is decoupled (precision-aware) or is equivalent to the
        distributed optimizer parameter precision, the gradient shard is a view of
        the Megatron-FSDP sharded gradient buffer. If not, then not dereferencing
        this gradient shard will increase memory utilization as this gradient is a
        persistent casted-copy of the accumulated gradient.
        """
        if not self.ddp_config.megatron_fsdp_cuda_graph_mode:
            
            
            for name, param in self.optimizer_named_parameters:
                param.grad = None
                if hasattr(param, "decoupled_grad"):
                    param.decoupled_grad = None
                if name in self.dist_main_grad:
                    self.dist_main_grad[name]._local_tensor = None

        
        
        for group in self.parameter_groups:
            if group.main_grad_buffer:
                group.main_grad_buffer.data.zero_()
            if group.hfsdp_helper_gbuf:
                group.hfsdp_helper_gbuf.data.zero_()

    def _init_distributed_params(self):
        """
        Register model training and high-precision parameters as optimizer
        named parameters and DTensor(s). Specifically, we utilize the highest
        precision weights available for optimization using fall-back logic
        on mbuf -> wbuf -> orig_param depending on if main_params_dtype is
        specified or "no_shard" is utilized.
        """
        dist_main_weight = {}
        for pg in self.parameter_groups:
            wbuf = pg.model_weight_buffer
            tbuf = pg.transpose_weight_buffer
            mbuf = pg.main_weight_buffer
            for item_id, orig_param in enumerate(pg.params):
                param_name = self.param_to_name[orig_param]

                
                
                
                
                
                sharded_optimizer_state = (
                    self.bucketing_policy.data_parallel_sharding_strategy != "no_shard"
                )

                
                if mbuf:
                    dist_param = make_fsdp_dtensor(
                        local_tensor=mbuf.get_item(item_id, only_shard=sharded_optimizer_state),
                        param=orig_param,
                        dist_index=self.dist_index,
                        is_sharded_param=sharded_optimizer_state,
                        is_expert_param=pg.is_expert_param,
                        run_check=True,
                        update_uneven_dtensor_chunk_meta=True,
                        force_sync_tp_duplicated_param=True,
                    )
                    dist_main_weight[param_name] = dist_param
                elif wbuf:
                    assert tbuf is None, "Transpose buffer should only exist when main params exist"
                    dist_param = make_fsdp_dtensor(
                        local_tensor=wbuf.get_item(item_id, only_shard=sharded_optimizer_state),
                        param=orig_param,
                        dist_index=self.dist_index,
                        is_sharded_param=sharded_optimizer_state,
                        is_expert_param=pg.is_expert_param,
                        run_check=True,
                        update_uneven_dtensor_chunk_meta=True,
                        force_sync_tp_duplicated_param=True,
                    )
                    dist_main_weight[param_name] = dist_param
                else:
                    
                    
                    
                    dist_param = make_fsdp_dtensor(
                        local_tensor=orig_param.data,
                        param=orig_param,
                        dist_index=self.dist_index,
                        is_sharded_param=False,
                        is_expert_param=pg.is_expert_param,
                        run_check=True,
                        update_uneven_dtensor_chunk_meta=False,
                        force_sync_tp_duplicated_param=True,
                    )
                    dist_main_weight[param_name] = dist_param

        
        self.dist_main_weight = dist_main_weight
        self.dist_main_grad = {}

    def _init_optimizer_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        named_parameters = []
        for pg in self.parameter_groups:
            for item_id, orig_param in enumerate(pg.params):
                param_name = self.param_to_name[orig_param]

                
                if param_name in self.dist_main_weight:
                    param_data = self.dist_main_weight[param_name]
                else:
                    
                    raise ValueError(
                        f"Parameter {param_name} not found in dist model weight "
                        "or dist main weight."
                    )
                dist_param = torch.nn.Parameter(param_data)

                def set_param_attribute_closure(param, orig_param):
                    def set_param_attribute():
                        for attr_name in [
                            "requires_grad",
                            "sequence_parallel",
                            "shared",
                            "tensor_model_parallel",
                            "partition_dim",
                            "partition_stride",
                            "is_embedding_or_output_parameter",
                            "is_embedding_parameter",
                            "_tensor_parallel_mode",
                        ]:
                            if hasattr(orig_param, attr_name):
                                setattr(param, attr_name, getattr(orig_param, attr_name))

                    return set_param_attribute

                setattr(
                    dist_param,
                    "reset_attribute",
                    set_param_attribute_closure(dist_param, orig_param),
                )
                setattr(dist_param, "orig_param", orig_param)
                setattr(dist_param, "megatron_fsdp_dist_index", self.dist_index)

                
                
                mbuf = pg.model_weight_buffer
                if mbuf:
                    _start, _end = mbuf._get_item_slice_in_shard(item_id)
                    setattr(dist_param, "megatron_fsdp_slice", slice(_start, _end))

                dist_param.reset_attribute()
                named_parameters.append((param_name, dist_param))

        return named_parameters

    def update_main_grads(self):
        """
        Update the gradients in the model parameters with the main gradients
        from the main gradient buffer. If the model parameters are sharded,
        we only need to update the gradient shard associated with the model
        parameter shard, as both are sharded symmetrically.
        """
        for name, param in self.optimizer_named_parameters:
            orig_param = param.orig_param
            group = self.parameter_groups[self.param_to_param_group[orig_param]]
            gbuf = group.main_grad_buffer
            if gbuf is None:
                
                continue

            
            
            sharded_optimizer_state = (
                self.bucketing_policy.data_parallel_sharding_strategy != "no_shard"
            )

            
            item_id = group.main_grad_buffer.param_idx[orig_param]
            optimizer_grad = group.main_grad_buffer.get_item(
                item_id, only_shard=sharded_optimizer_state
            )
            if group.main_weight_buffer is not None and not self.use_decoupled_grad:
                
                
                
                optimizer_grad = optimizer_grad.to(param.dtype)

            if name not in self.dist_main_grad:
                
                self.dist_main_grad[name] = make_fsdp_dtensor(
                    local_tensor=optimizer_grad,
                    param=orig_param,
                    dist_index=self.dist_index,
                    is_sharded_param=sharded_optimizer_state,
                    is_expert_param=group.is_expert_param,
                )
            else:
                
                if len(orig_param.shape) > 1:
                    local_shape = (-1, *orig_param.shape[1:])
                else:
                    local_shape = (-1,)
                self.dist_main_grad[name]._local_tensor = optimizer_grad.view(local_shape)
            grad = self.dist_main_grad[name]

            if optimizer_grad.numel() == 0:
                grad = None

            
            
            if self.use_decoupled_grad:
                setattr(param, "decoupled_grad", grad)
            else:
                
                setattr(param, "grad", grad)

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return len(self.parameter_groups)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """
        Update the model weights from the main weights.

        If FP8 parameters are utilized, this function will quantize the high-precision
        main weights prior to installation into the model compute weight buffers.
        """
        dense_param_quantize_kwargs = {
            "model_params": [],
            "main_params": [],
            "start_offsets": [],
            "fsdp_shard_model_params": [],
        }
        expert_param_quantize_kwargs = copy.deepcopy(dense_param_quantize_kwargs)
        data_parallel_group = None
        expert_data_parallel_group = None
        clear_quantize_kwargs = lambda kwargs: [d.clear() for d in kwargs.values()]

        def _fp8_quantize_params(dense_param_quantize_kwargs, expert_param_quantize_kwargs):
            if len(dense_param_quantize_kwargs["model_params"]) > 0:
                
                fp8_quantize(data_parallel_group=data_parallel_group, **dense_param_quantize_kwargs)

            if len(expert_param_quantize_kwargs["model_params"]) > 0:
                
                fp8_quantize(
                    data_parallel_group=expert_data_parallel_group, **expert_param_quantize_kwargs
                )

            clear_quantize_kwargs(dense_param_quantize_kwargs)
            clear_quantize_kwargs(expert_param_quantize_kwargs)

        
        BATCH_QUANT_MEMORY_LIMIT_BYTES = 5 * 1024**3  
        blockwise_fp8_weight_buffers = []
        blockwise_fp8_param_buffers = []

        def _batch_quantize_blockwise_fp8_params(
            dense_param_quantize_kwargs, expert_param_quantize_kwargs, blockwise_fp8_param_buffers
        ):
            if len(blockwise_fp8_param_buffers) == 0:
                return

            
            for bufs in blockwise_fp8_param_buffers:
                bufs["bucket_param"].copy_(bufs["param"])

            
            _fp8_quantize_params(dense_param_quantize_kwargs, expert_param_quantize_kwargs)

            
            for bufs in blockwise_fp8_param_buffers:
                bufs["param"].copy_(bufs["bucket_param"])
            blockwise_fp8_param_buffers.clear()

            
            for wbuf in blockwise_fp8_weight_buffers:
                wbuf.free_bucket_storage()
            blockwise_fp8_weight_buffers.clear()

        for pg in self.parameter_groups:
            mbuf = pg.main_weight_buffer
            wbuf = pg.model_weight_buffer
            tbuf = pg.transpose_weight_buffer
            if mbuf is None:
                continue

            if pg.is_expert_param:
                quantize_func_kwargs = expert_param_quantize_kwargs
                expert_data_parallel_group = mbuf.data_parallel_group
            else:
                quantize_func_kwargs = dense_param_quantize_kwargs
                data_parallel_group = mbuf.data_parallel_group

            fp8_params = quantize_func_kwargs["model_params"]
            shard_fp32_from_fp8 = quantize_func_kwargs["main_params"]
            shard_offsets_in_fp8 = quantize_func_kwargs["start_offsets"]
            shard_model_params = quantize_func_kwargs["fsdp_shard_model_params"]

            has_blockwise_fp8_param = False
            for param in pg.params:
                item_id = mbuf.param_idx[param]
                if wbuf:
                    if wbuf.is_data_distributed or mbuf.is_data_distributed:
                        model_param = wbuf.get_item(item_id, only_shard=True)
                        if tbuf:
                            transpose_param = tbuf.get_item(item_id, only_shard=True)
                        else:
                            transpose_param = None
                        main_weight = mbuf.get_item(item_id, only_shard=True)
                    else:
                        model_param = wbuf.get_item(item_id)
                        if tbuf:
                            transpose_param = tbuf.get_item(item_id)
                        else:
                            transpose_param = None
                        main_weight = mbuf.get_item(item_id)
                else:
                    assert not mbuf.is_data_distributed
                    model_param = to_local_if_dtensor(param)
                    main_weight = mbuf.get_item(item_id)

                
                
                
                
                

                if is_blockwise_float8tensor(param):
                    fp8_params.append(param)
                    if model_param.numel() == 0:
                        
                        shard_fp32_from_fp8.append(None)
                        shard_offsets_in_fp8.append(None)
                        shard_model_params.append([None, None])
                    else:
                        shard_fp32_from_fp8.append(main_weight)
                        shard_offsets_in_fp8.append(wbuf.locate_item_in_global_item(item_id)[0])
                        bucket = wbuf.fetch_bucket()
                        b_model_param = wbuf.get_item_from_bucket(bucket, item_id)[
                            slice(*wbuf.locate_item_in_global_item(item_id))
                        ]
                        assert (
                            transpose_param is None
                        ), "Blockwise FP8 does not support transpose param."
                        shard_model_params.append([b_model_param, None])
                        assert b_model_param.numel() == model_param.numel(), (
                            f"Blockwise FP8 bucket param numel {b_model_param.numel()} does"
                            f" not match model param numel {model_param.numel()}"
                            f" name: {self.param_to_name[param]}"
                        )
                        blockwise_fp8_param_buffers.append(
                            {"bucket_param": b_model_param, "param": model_param}
                        )
                        has_blockwise_fp8_param = True
                    continue

                if is_float8tensor(param):
                    fp8_params.append(param)
                    if model_param.numel() == 0:
                        
                        shard_fp32_from_fp8.append(None)
                        shard_offsets_in_fp8.append(None)
                        shard_model_params.append([None, None])
                    else:
                        shard_fp32_from_fp8.append(main_weight)
                        shard_offsets_in_fp8.append(wbuf.locate_item_in_global_item(item_id)[0])
                        shard_model_params.append([model_param, transpose_param])
                    continue

                if model_param.numel() > 0:
                    model_param.data.copy_(main_weight.view(model_param.shape))

            if has_blockwise_fp8_param:
                blockwise_fp8_weight_buffers.append(wbuf)
                if (
                    sum([wbuf.bucket_index.size for wbuf in blockwise_fp8_weight_buffers])
                    > BATCH_QUANT_MEMORY_LIMIT_BYTES
                ):
                    _batch_quantize_blockwise_fp8_params(
                        dense_param_quantize_kwargs,
                        expert_param_quantize_kwargs,
                        blockwise_fp8_param_buffers,
                    )

        _batch_quantize_blockwise_fp8_params(
            dense_param_quantize_kwargs, expert_param_quantize_kwargs, blockwise_fp8_param_buffers
        )
        _fp8_quantize_params(dense_param_quantize_kwargs, expert_param_quantize_kwargs)

    def all_gather_parameters(self, async_op: bool = True):
        """All gather the parameters.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.model_weight_buffer.is_data_distributed for g in self.parameter_groups]
        ), "all_gather_parameters() should only be called when parameters are not sharded."
        assert (
            self.ddp_config.outer_dp_sharding_strategy == "no_shard"
        ), "all_gather_parameters() should not be called when outer-DP sharding is enabled."

        all_gather_ops = []
        for g in self.parameter_groups:
            for buf in [g.model_weight_buffer, g.transpose_weight_buffer]:
                if buf is None:
                    continue
                shard = buf.get_shard_from_local_buffer()
                all_gather_handler = torch.distributed.all_gather_into_tensor(
                    output_tensor=buf.data,
                    input_tensor=shard,
                    group=buf.data_parallel_group,
                    async_op=async_op,
                )
                if async_op:
                    all_gather_ops.append(all_gather_handler)

        for op in all_gather_ops:
            op.wait()

    def reduce_scatter_gradients(self, async_op: bool = True):
        """Reduce scatter the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups]
        ), "reduce_scatter_gradients() should only be called when gradients are not sharded."
        assert (
            self.ddp_config.outer_dp_sharding_strategy == "no_shard"
        ), "reduce_scatter_gradients() should not be called when outer-DP sharding is enabled."

        reduce_scatter_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            if self.ddp_config.check_for_nan_in_grad:
                _check_nan_in_grad(gbuf.data)
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            reduce_scatter_handler = torch.distributed.reduce_scatter_tensor(
                output=gbuf.get_shard_from_local_buffer(),
                input=gbuf.data,
                op=reduce_op,
                group=g.main_grad_buffer.data_parallel_group,
                async_op=async_op,
            )

            if async_op:
                reduce_scatter_ops.append(reduce_scatter_handler)

        for op in reduce_scatter_ops:
            op.wait()

    def all_reduce_gradients(self, async_op: bool = False):
        """All reduce the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [
                not g.main_grad_buffer.is_data_distributed
                for g in self.parameter_groups
                if g.main_grad_buffer
            ]
        ), "all_reduce_gradients() should only be called when gradients are not sharded."
        assert (
            self.ddp_config.outer_dp_sharding_strategy == "no_shard"
        ), "all_reduce_gradients() should not be called when outer-DP sharding is enabled."

        all_reduce_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            if self.ddp_config.check_for_nan_in_grad:
                _check_nan_in_grad(gbuf.data)
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            all_reduce_handler = torch.distributed.all_reduce(
                gbuf.data, op=reduce_op, group=gbuf.data_parallel_group, async_op=async_op
            )
            if async_op:
                all_reduce_ops.append(all_reduce_handler)

        for op in all_reduce_ops:
            op.wait()


def _create_hfsdp_helper_buffer(
    dp_buffer: DataParallelBuffer,
    inner_dp_group: torch.distributed.ProcessGroup,
    is_data_distributed: bool,
) -> DataParallelBuffer:
    """
    Create a Hybrid-FSDP helper DataParallelBuffer on the inner-DP group.

    This helper buffer mirrors the metadata of the original fully
    `dp_buffer` (bucket config, params, allocator, etc.), but binds it to
    the `inner_dp_group` and computes a per-rank `shard_bucket_index`
    appropriate for that group. The resulting buffer is used as the
    HFSDP helper buffer that owns the persistent inner-DP shard of the
    global bucket, while still sharing the same logical bucket indexing
    (`bucket_index`) with the fully DP buffer.

    Parameters
    ==========
    dp_buffer : DataParallelBuffer
        The existing fully data-parallel buffer whose configuration
        and bucket layout should be mirrored.
    inner_dp_group : torch.distributed.ProcessGroup
        The process group representing the inner-DP (HFSDP) data-parallel
        group for this helper buffer.
    is_data_distributed : bool
        Whether the underlying data in this helper buffer is sharded
        across ranks in `inner_dp_group`.

    Returns
    =======
    DataParallelBuffer
        A new DataParallelBuffer configured as the HFSDP helper buffer
        for the given `inner_dp_group`, sharing the same bucket index
        as `dp_buffer` but with an inner-DP `shard_bucket_index`.
    """
    helper_buffer = DataParallelBuffer(
        dp_buffer.ddp_config,
        dp_buffer.params,
        is_data_distributed=is_data_distributed,
        dtype=dp_buffer.dtype,
        device=dp_buffer.device,
        data_parallel_group=inner_dp_group,
        is_transpose_buffer=dp_buffer.is_transpose_buffer,
        temporary_bucket_allocator=dp_buffer.temporary_bucket_allocator,
        bucket_id=dp_buffer.bucket_id,
        chunk_size_factor=dp_buffer.chunk_size_factor,
        mem_alloc_context=dp_buffer.mem_alloc_context,
        item_index_map=dp_buffer.item_index_map,
        bucket_index=dp_buffer.bucket_index,
        
        
        
        shard_bucket_index=_get_dp_buffer_shard_bucket_index(
            bucket_index=dp_buffer.bucket_index,
            is_data_distributed=is_data_distributed,
            data_parallel_world_size=inner_dp_group.size(),
            data_parallel_rank=inner_dp_group.rank(),
        ),
    )

    return helper_buffer


def _init_hfsdp_helper_and_dp_buffer_data(
    hfsdp_helper_buffer: DataParallelBuffer,
    dp_buffer: DataParallelBuffer,
    mem_alloc: Callable[[torch.dtype, int], torch.Tensor],
    outer_dp_group: torch.distributed.ProcessGroup,
) -> None:
    """
    Initialize storage for the HFSDP helper buffer and its corresponding
    fully-DP DataParallelBuffer view.

    The helper buffer is allocated as a single contiguous tensor that
    stores all DP shards for the given bucket. Each rank in the outer-DP
    group then takes its local slice of this storage and exposes it
    through `dp_buffer`, so the fully-DP buffer becomes a view into the helper
    buffer rather than owning separate storage.

    Parameters
    ==========
    hfsdp_helper_buffer : DataParallelBuffer
        The HFSDP helper buffer that owns the full inner-/outer-DP bucket
        storage.
    dp_buffer : DataParallelBuffer
        The fully-DP DataParallelBuffer that should view its local shard
        from `hfsdp_helper_buffer`.
    mem_alloc : Callable[[torch.dtype, int], torch.Tensor]
        Allocation function used to create the backing tensor for the
        helper buffer (dtype, numel).
    outer_dp_group : torch.distributed.ProcessGroup
        Process group for the outer data-parallel dimension. Its rank and
        world size determine which slice of the helper buffer this rank
        sees through `dp_buffer`.
    """
    
    hfsdp_helper_buffer.init_data(
        mem_alloc(dtype=hfsdp_helper_buffer.dtype, size=hfsdp_helper_buffer.data_size)
    )

    rank = outer_dp_group.rank()
    shard_size = dp_buffer.data_size
    start = shard_size * rank
    end = shard_size * (rank + 1)

    
    
    
    dp_buffer_data = hfsdp_helper_buffer.data[start:end]
    dp_buffer.init_data(dp_buffer_data)


class BucketStatus(Enum):
    """
    An enumeration of possible statuses for a data-parallel communication bucket.

    Attributes:
        EMPTY (int): The bucket is empty and not in use.
        PRESERVED (int): The bucket storage is retained but not ready for use.
        COMMUNICATING (int): The bucket is currently being used for communication.
        READY_TO_USE (int): The bucket is filled with data and ready for use.
    """

    EMPTY = 1
    PRESERVED = 2
    COMMUNICATING = 3
    READY_TO_USE = 4


class GradReducePipeline:
    """
    Pipeline for reducing gradients.
    """

    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        rs_stream: Optional[torch.cuda.Stream] = None,
        check_nans: bool = False,
    ) -> None:
        self.buffer = param_and_grad_buffer
        
        self.grad_reduce_queue = []
        self.bucket_status = {
            
            
            i: BucketStatus.EMPTY
            for i in range(self.buffer.num_buckets)
            if self.buffer.parameter_groups[i].main_grad_buffer
        }
        
        self.bucket_grad_ready_params = [set() for _ in range(self.buffer.num_buckets)]
        self.rs_stream = rs_stream
        self.check_nans = check_nans

        
        dist_index = self.buffer.dist_index
        if dist_index.use_hybrid_fsdp:
            
            self.outer_fsdp_group_grad_reduce = True
            self.outer_fsdp_group_grad_reduce_stream = torch.cuda.Stream()
        else:
            self.outer_fsdp_group_grad_reduce = False

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self):
        """Handle the processing tasks and reset the pipeline."""
        self.wait_for_previous_grad_reduce(0)
        for bucket_id, grad_ready_params in enumerate(self.bucket_grad_ready_params):
            param_list = self.buffer.parameter_groups[bucket_id].params
            n_params = len(param_list)
            param_to_name = self.buffer.param_to_name
            assert len(grad_ready_params) == 0, (
                f"Found {len(grad_ready_params)} out of {n_params} parameters that are ready for "
                f"reduce-scatter/all-reduce, but the pipeline is being reset. "
                f"grad_ready_params: {[param_to_name[p] for p in grad_ready_params]} "
                f"param_list: {[param_to_name[p] for p in param_list]}"
            )

        for bucket_id, _ in self.bucket_status.items():
            gbuf = self.get_fsdp_buffer(bucket_id)
            gbuf.free_bucket_storage()
            gbuf.reset_param_main_grad()
            self.bucket_status[bucket_id] = BucketStatus.EMPTY

    def reduce_gradients(
        self,
        params: List[torch.Tensor],
        suggested_queue_capacity: Optional[int] = None,
        outer_fsdp_group_grad_reduce: bool = False,
    ):
        """Reduce the gradients for the given parameters.
        Args:
            params (List[torch.Tensor]): The parameters.
            suggested_queue_capacity (int, optional): The suggested queue capacity.
                Defaults to None.
            outer_fsdp_group_grad_reduce (bool, optional): Whether to reduce gradients
                across outer-DP groups. Defaults to False.
        """
        
        
        params = sorted(list(params), key=lambda x: self.buffer.param_to_param_group[x])
        for param in params:
            bucket_id = self.buffer.param_to_param_group[param]
            param_group = self.buffer.parameter_groups[bucket_id]
            if not param.requires_grad:
                assert param_group.requires_grad is False, (
                    f"Param {self.buffer.param_to_name[param]} has requires_grad=False, "
                    f"but it is in a parameter group with requires_grad=True."
                )
                continue
            assert param_group.requires_grad, (
                f"Param {self.buffer.param_to_name[param]} has requires_grad=True, "
                f"but it is in a parameter group with requires_grad=False."
            )

            
            self.bucket_grad_ready_params[bucket_id].add(param)
            bucket_group = self.get_ready_bucket_group_for_reduction(bucket_id)
            if bucket_group:
                self.wait_for_previous_grad_reduce(
                    suggested_queue_capacity=suggested_queue_capacity
                )
                self._bucket_group_gradient_reduce(
                    bucket_group,
                    async_op=True,
                    outer_fsdp_group_grad_reduce=outer_fsdp_group_grad_reduce,
                )

    def wait_for_previous_grad_reduce(
        self, suggested_queue_size: int = 1, suggested_queue_capacity: Optional[int] = None
    ):
        """
        Wait for the previous reduce-scatter/all-reduce to finish.
        Args:
            suggested_queue_size (int, optional): The recommended queue size in
                buckets. Defaults to 1.
            suggested_queue_capacity (Optional[int], optional): The recommended queue capacity
                in number of parameters in all buckets in the reduction queue. Defaults to None.
        """
        if suggested_queue_capacity is not None:
            queue_space = sum(
                [
                    self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size
                    for _, _, bucket_id in self.grad_reduce_queue
                ]
            )
            while queue_space > suggested_queue_capacity:
                grad_reduce_event, free_up_grad_bucket, bucket_id = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()
                queue_space -= self.buffer.parameter_groups[
                    bucket_id
                ].main_grad_buffer.bucket_index.size
        else:
            suggested_queue_size = max(0, min(suggested_queue_size, self.buffer.num_buckets - 1))
            while len(self.grad_reduce_queue) > suggested_queue_size:
                grad_reduce_event, free_up_grad_bucket, _ = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()

    def _enforce_double_buffer_limit(self, add_buckets):
        if not self.buffer.ddp_config.fsdp_double_buffer:
            return

        param_groups = self.buffer.parameter_groups
        double_buf_units = set()
        for bucket_id in add_buckets:
            fsdp_unit_id = param_groups[bucket_id].fsdp_unit_id
            if fsdp_unit_id in self.buffer.double_buf_units:
                double_buf_units.add(fsdp_unit_id)
        assert (
            len(double_buf_units) <= 2
        ), f"Double buffer limit exceeded. Current double_buf_units: {double_buf_units}."

        keep_n = len(self.grad_reduce_queue)
        for _, _, bucket_id in reversed(self.grad_reduce_queue):
            fsdp_unit_id = param_groups[bucket_id].fsdp_unit_id
            double_buf_units.add(fsdp_unit_id)
            if len(double_buf_units) > 1:
                keep_n -= 1

        with torch.cuda.stream(self.rs_stream):
            self.wait_for_previous_grad_reduce(keep_n)

    def get_ready_bucket_group_for_reduction(self, bucket_id: int) -> Optional[List[int]]:
        """Checks if all buckets in the bucket group containing the given bucket_id
        are ready for gradient reduction.
        If so, returns the list of ready bucket IDs for reduction; otherwise, returns None.

        Args:
            bucket_id (int): The bucket to mark as ready for reduce-scatter or all-reduce.

        Returns:
            Optional[List[int]]: The bucket group ready for gradient reduction,
            or None if not all buckets are ready.
        """
        
        
        
        bucket_group = self.buffer.bucket_to_bucket_group[bucket_id]
        bucket_group = [i for i in bucket_group if self.buffer.parameter_groups[i].main_grad_buffer]
        
        
        for bucket_id in bucket_group:
            param_group = self.buffer.parameter_groups[bucket_id]
            if len(self.bucket_grad_ready_params[bucket_id]) != len(param_group.params):
                return None

        return bucket_group

    def get_fsdp_buffer(self, bucket_id: int) -> DataParallelBuffer:
        """Get the FSDP buffer for the given bucket ID."""
        param_group = self.buffer.parameter_groups[bucket_id]
        if self.buffer.ddp_config.outer_dp_sharding_strategy != "no_shard":
            return param_group.hfsdp_helper_gbuf
        return param_group.main_grad_buffer

    def _bucket_group_gradient_reduce(
        self,
        bucket_group: List[int],
        async_op: bool = False,
        outer_fsdp_group_grad_reduce: bool = False,
    ) -> bool:
        """Mark the bucket ready for reduce-scatter/all-reduce, if all bucket in
        the bucket group are ready, then do the reduce-scatter/all-reduce.
        Args:
            bucket_id (int): The bucket to be marked.
            bucket_group (List[int]): The bucket group to be reduced.
            async_op (bool, optional): Whether to do the reduce-scatter/all-reduce
                asynchronously. Defaults to False.
        Returns:
            bool: True if the bucket is go for reduce-scatter/all-reduce.
        """
        
        
        
        ddp_config = self.buffer.ddp_config
        mp_policy = self.buffer.mp_policy
        if ddp_config.fsdp_double_buffer:
            self._enforce_double_buffer_limit(bucket_group)

        current_stream = torch.cuda.current_stream()
        reduce_scatter_stream = (
            self.rs_stream if self.rs_stream is not None else torch.cuda.current_stream()
        )
        reduce_scatter_stream.wait_stream(current_stream)

        
        dp_group = self.get_fsdp_buffer(bucket_group[0]).data_parallel_group
        with torch.cuda.stream(reduce_scatter_stream):
            with _coalescing_manager(dp_group):
                
                
                grad_accum_closure = []
                for bucket_id in bucket_group:

                    
                    gbuf = self.get_fsdp_buffer(bucket_id)

                    
                    unreduced_grad_bucket = gbuf.fetch_bucket(
                        dtype=mp_policy.grad_comm_dtype if gbuf.is_data_distributed else None
                    )
                    
                    
                    
                    
                    custom_grad_comm_dtype = (
                        mp_policy.grad_comm_dtype is not None
                        and unreduced_grad_bucket.data.dtype != mp_policy.grad_comm_dtype
                    )
                    if not gbuf.is_data_distributed and custom_grad_comm_dtype:
                        
                        
                        unreduced_grad_bucket = gbuf.allocate_bucket_storage(
                            dtype=mp_policy.grad_comm_dtype,
                            device=unreduced_grad_bucket.data.device,
                            init_values=unreduced_grad_bucket.data,
                        )
                    unreduced_grad = unreduced_grad_bucket.data

                    
                    scaling_factor = gbuf.gradient_scaling_factor
                    reduce_op = gradient_reduce_preprocessing(
                        unreduced_grad, scaling_factor, ddp_config
                    )

                    
                    if ddp_config.data_parallel_sharding_strategy == "no_shard":
                        
                        torch.distributed.all_reduce(
                            unreduced_grad, op=reduce_op, group=gbuf.data_parallel_group
                        )
                        if custom_grad_comm_dtype:
                            
                            grad_accum_closure.append(
                                
                                (gbuf.data, unreduced_grad)
                            )
                    else:
                        
                        grad_shard = gbuf.get_shard_from_bucket(unreduced_grad_bucket)

                        
                        torch.distributed.reduce_scatter_tensor(
                            output=grad_shard,
                            input=unreduced_grad,
                            op=reduce_op,
                            group=gbuf.data_parallel_group,
                        )

                        
                        
                        
                        
                        
                        if gbuf.is_data_distributed or custom_grad_comm_dtype:
                            grad_accum_closure.append(
                                
                                (gbuf.get_shard_from_local_buffer(), grad_shard)
                            )

                    
                    self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING

            for local_grad, reduced_grad in grad_accum_closure:
                if ddp_config.data_parallel_sharding_strategy in ["no_shard", "optim"]:
                    
                    local_grad.copy_(reduced_grad)
                else:
                    
                    
                    
                    local_grad += reduced_grad

            
            reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

        
        if outer_fsdp_group_grad_reduce:
            
            self.outer_fsdp_group_grad_reduce_stream.wait_stream(reduce_scatter_stream)
            outer_fsdp_group = self.buffer.dist_index.get_outer_fsdp_group()
            with torch.cuda.stream(self.outer_fsdp_group_grad_reduce_stream):
                with _coalescing_manager(outer_fsdp_group):
                    
                    
                    grad_accum_closure = []
                    for bucket_id in bucket_group:
                        
                        
                        if ddp_config.average_in_collective:
                            reduce_op = torch.distributed.ReduceOp.AVG
                        else:
                            reduce_op = torch.distributed.ReduceOp.SUM

                        
                        main_grad_buffer = self.buffer.parameter_groups[bucket_id].main_grad_buffer

                        
                        
                        
                        fsdp_grad_buffer = self.get_fsdp_buffer(bucket_id)
                        unreduced_grad = fsdp_grad_buffer.data
                        assert (
                            main_grad_buffer.dtype == fsdp_grad_buffer.dtype
                        ), "Main and DP-Shard gradient buffer must share the exact same dtype."

                        
                        custom_grad_comm_dtype = (
                            mp_policy.grad_comm_dtype is not None
                            and unreduced_grad.dtype != mp_policy.grad_comm_dtype
                        )
                        if custom_grad_comm_dtype:
                            
                            
                            hsdp_comm_gbuf = self.buffer.parameter_groups[bucket_id].hsdp_comm_gbuf
                            unreduced_grad = hsdp_comm_gbuf.allocate_bucket_storage(
                                
                                
                                shard=fsdp_grad_buffer.is_data_distributed,
                                dtype=mp_policy.grad_comm_dtype,
                                device=unreduced_grad.device,
                                init_values=unreduced_grad,
                            ).data

                        
                        if ddp_config.outer_dp_sharding_strategy != "no_shard":
                            
                            
                            
                            main_grad_shard = main_grad_buffer.get_shard_from_local_buffer()
                            if custom_grad_comm_dtype:
                                
                                dp_outer_rank = outer_fsdp_group.rank()
                                output_buffer = unreduced_grad[
                                    dp_outer_rank
                                    * main_grad_shard.numel() : (dp_outer_rank + 1)
                                    * main_grad_shard.numel()
                                ]
                            else:
                                
                                output_buffer = main_grad_shard
                            
                            
                            torch.distributed.reduce_scatter_tensor(
                                output=output_buffer,
                                input=unreduced_grad,
                                op=reduce_op,
                                group=outer_fsdp_group,
                            )
                            if custom_grad_comm_dtype:
                                
                                grad_accum_closure.append((main_grad_shard, output_buffer))
                        else:  
                            
                            
                            torch.distributed.all_reduce(
                                unreduced_grad, group=outer_fsdp_group, op=reduce_op
                            )
                            if custom_grad_comm_dtype:
                                
                                grad_accum_closure.append((main_grad_buffer.data, unreduced_grad))

                for main_grad_buffer, reduced_grad in grad_accum_closure:
                    
                    
                    main_grad_buffer.copy_(reduced_grad)

            reduce_scatter_view_out_event = self.outer_fsdp_group_grad_reduce_stream.record_event()

        free_up_grad_bucket_func = {}
        for bucket_id in bucket_group:

            def get_closure(bucket_id):
                def free_up_grad_bucket():
                    
                    self.bucket_grad_ready_params[bucket_id] = set()
                    gbuf = self.get_fsdp_buffer(bucket_id)
                    
                    
                    
                    gbuf.free_bucket_storage()
                    
                    gbuf.reset_param_main_grad()
                    hsdp_comm_gbuf = self.buffer.parameter_groups[bucket_id].hsdp_comm_gbuf
                    if hsdp_comm_gbuf is not None:
                        
                        hsdp_comm_gbuf.free_bucket_storage()
                    
                    self.bucket_status[bucket_id] = BucketStatus.EMPTY

                return free_up_grad_bucket

            free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

        if async_op:
            for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
                self.grad_reduce_queue.append(
                    (reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id)
                )
            return True

        reduce_scatter_view_out_event.wait()
        for free_up_grad_bucket in free_up_grad_bucket_func.values():
            free_up_grad_bucket()
        return True


class PrefetchOrder(Enum):
    """
    An enumeration of possible prefetch orders for data-parallel operations.

    Attributes:
        FORWARD_PASS_ORDER (int): Prefetch in the order of forward pass computation.
        BACKWARD_PASS_ORDER (int): Prefetch in the order of backward pass computation.
    """

    FORWARD_PASS_ORDER = 0
    BACKWARD_PASS_ORDER = 1


class AllGatherPipeline:
    """
    Pipeline for all-gathering parameters.
    """

    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        ag_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.buffer = param_and_grad_buffer
        self.ag_stream = ag_stream
        
        self.param_gather_event_map = {}
        
        self.bucket_status = {}
        for i in range(self.buffer.num_buckets):
            for bwd in [False, True]:
                self.bucket_status[self.get_bucket_key(i, bwd)] = BucketStatus.EMPTY

        
        self.bucket_can_be_released = {}
        for i in range(self.buffer.num_buckets):
            for bwd in [False, True]:
                self.bucket_can_be_released[self.get_bucket_key(i, bwd)] = False

        
        
        self.bucket_to_bucket_group = {}
        group_id = 0
        for bucket_group in self.buffer.bucket_to_bucket_group.values():
            new_group = False
            for bucket_id in bucket_group:
                if bucket_id not in self.bucket_to_bucket_group:
                    new_group = True
                    break
            if new_group:
                group_id += 1
                for bucket_id in bucket_group:
                    self.bucket_to_bucket_group[bucket_id] = group_id

        if (
            self.buffer.dist_index.use_hybrid_fsdp
            and self.buffer.ddp_config.outer_dp_sharding_strategy != "no_shard"
        ):
            
            
            self.outer_fsdp_group_param_gather_stream = torch.cuda.Stream()

    def get_bucket_key(self, bucket_id, bwd):
        """Get the key for the bucket."""
        has_transpose_buffer = (
            self.buffer.parameter_groups[bucket_id].transpose_weight_buffer is not None
        )
        return (bucket_id, has_transpose_buffer and bwd)

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self, preserve_non_fsdp_units: bool = True):
        """Reset the pipeline state.

        Non-FSDP-unit buckets are preserved by default because their params may
        be read across module boundaries. Setting preserve_non_fsdp_units=False
        releases all bucket storage and is intended only for debugging when the
        model will not be reused.
        """
        if len(self.param_gather_event_map) > 0:
            warnings.warn(
                (
                    f"There are still pending all-gather tasks, process them. "
                    f"Bucket status: {self.bucket_status}."
                ),
                UserWarning,
            )
            while len(self.param_gather_event_map) > 0:
                (bucket_id, bwd) = next(iter(self.param_gather_event_map))
                self.wait_bucket_ready(bucket_id, bwd)

        for bucket_id in range(self.num_buckets):
            is_unit_bucket = self.buffer.parameter_groups[bucket_id].fsdp_unit_id is not None
            for bwd in [False, True]:
                bucket_key = self.get_bucket_key(bucket_id, bwd)
                
                
                
                
                if preserve_non_fsdp_units and not is_unit_bucket:
                    self.bucket_status[bucket_key] = BucketStatus.PRESERVED
                else:
                    self.bucket_can_be_released[bucket_key] = True
        self.recycle_unused_buckets()

        expected_statuses = (BucketStatus.EMPTY,)
        if preserve_non_fsdp_units:
            expected_statuses += (BucketStatus.PRESERVED,)

        assert all(status in expected_statuses for status in self.bucket_status.values()), (
            f"There are still working buckets, it is not safe to reset. "
            f"bucket_status: {self.bucket_status}."
        )
        assert all(
            [not can_be_released for can_be_released in self.bucket_can_be_released.values()]
        ), (
            f"The bucket can be released table is in an abnormal state, not safe to reset. "
            f"bucket_can_be_released: {self.bucket_can_be_released}."
        )

    def all_gather_params(
        self,
        params: List[torch.Tensor],
        prefetch: bool = False,
        prefetch_order: PrefetchOrder = PrefetchOrder.FORWARD_PASS_ORDER,
        suggested_AG_prefetch_size: Optional[int] = None,
        async_param_gather: bool = True,
        outer_fsdp_group_param_gather: bool = False,
        bwd: bool = False,
    ):
        """All-gather the params. If prefetch is enabled, prefetch next buckets
        in the order of `prefetch_order`.

        Args:
            params (List[torch.Tensor]): The list of params to be all-gathered.
            prefetch (bool, optional): Whether to prefetch the next bucket. Defaults to False.
            prefetch_order (PrefetchOrder, optional): The order of prefetching.
                Defaults to PrefetchOrder.FORWARD_PASS_ORDER.
            suggested_AG_prefetch_size (Optional[int], optional):
                The suggested prefetch size for all-gathering. Defaults to None.
            outer_fsdp_group_param_gather (bool, optional):
                Whether to all-gather parameters across DP-Outer. Defaults to False.
            bwd (bool, optional):
                Whether to all-gather column-wise parameters instead of row-wise parameters
                for the backward pass for formats that require a transpose buffer like MXFP8.
        """
        if len(params) == 0:
            return

        ag_buckets = [self.buffer.param_to_param_group[item] for item in params]
        ag_buckets = list(sorted(set(ag_buckets)))  
        parameter_groups = self.buffer.parameter_groups
        if self.buffer.ddp_config.fsdp_double_buffer:
            double_buf_units = set()
            for bucket_id in ag_buckets:
                fsdp_unit_id = parameter_groups[bucket_id].fsdp_unit_id
                if fsdp_unit_id in self.buffer.double_buf_units:
                    double_buf_units.add(fsdp_unit_id)
            if len(double_buf_units) > 2:
                raise ValueError(
                    f"{double_buf_units} FSDP units were requested, "
                    "but double buffers can support no more than 2 FSDP units."
                )

        
        for bucket_id in ag_buckets:
            self.bucket_can_be_released[self.get_bucket_key(bucket_id, bwd)] = False

        
        if prefetch:

            def next_bucket_id(ag_buckets):
                """
                Search for the next bucket ID that is not in the list of all-gather buckets.
                """
                if prefetch_order == PrefetchOrder.FORWARD_PASS_ORDER:
                    
                    bucket_id = ag_buckets[0] + 1
                    for i in ag_buckets[1:]:
                        if i != bucket_id:
                            break
                        bucket_id += 1
                else:
                    
                    bucket_id = ag_buckets[-1] - 1
                    for i in reversed(ag_buckets[:-1]):
                        if i != bucket_id:
                            break
                        bucket_id -= 1
                if bucket_id < 0 or bucket_id >= self.buffer.num_buckets:
                    
                    return None
                return bucket_id

            def need_skip_prefetch(bucket_id):
                
                
                if self.buffer.ddp_config.fsdp_double_buffer:
                    fsdp_unit_id = parameter_groups[bucket_id].fsdp_unit_id
                    double_buf_units.add(fsdp_unit_id)
                    if len(double_buf_units) > 2:
                        
                        
                        return True
                return False

            if suggested_AG_prefetch_size is None:
                
                suggested_AG_prefetch_size = 500_000_000

            base_all_gather_size = sum(
                [parameter_groups[i].model_weight_buffer.bucket_index.size for i in ag_buckets]
            )
            bucket_id = next_bucket_id(ag_buckets)
            while bucket_id is not None:
                prefetch_all_gather_size = (
                    sum(
                        [
                            parameter_groups[i].model_weight_buffer.bucket_index.size
                            for i in ag_buckets
                        ]
                    )
                    - base_all_gather_size
                )
                if prefetch_all_gather_size >= suggested_AG_prefetch_size:
                    
                    break

                if need_skip_prefetch(bucket_id):
                    break

                
                ag_buckets.extend(self.buffer.bucket_to_bucket_group[bucket_id])
                
                ag_buckets = list(sorted(set(ag_buckets)))
                bucket_id = next_bucket_id(ag_buckets)

        
        
        ag_buckets = [
            bucket_id
            for bucket_id in ag_buckets
            if self.bucket_status[self.get_bucket_key(bucket_id, bwd)]
            in (BucketStatus.EMPTY, BucketStatus.PRESERVED)
        ]
        if len(ag_buckets) == 0:
            return

        
        
        bucket_group_to_buckets = {}
        for bucket_id in ag_buckets:
            group_id = self.bucket_to_bucket_group[bucket_id]
            if group_id not in bucket_group_to_buckets:
                bucket_group_to_buckets[group_id] = []
            bucket_group_to_buckets[group_id].append(bucket_id)

        
        for _, buckets in bucket_group_to_buckets.items():
            all_gather_stream = (
                self.ag_stream if self.ag_stream is not None else torch.cuda.current_stream()
            )
            if outer_fsdp_group_param_gather:
                self.outer_fsdp_group_param_gather_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.outer_fsdp_group_param_gather_stream):
                    is_expert_parallel = parameter_groups[buckets[0]].is_expert_param
                    outer_fsdp_group = self.buffer.dist_index.get_outer_fsdp_group(
                        is_expert_parallel=is_expert_parallel
                    )
                    with _coalescing_manager(outer_fsdp_group, async_ops=False):
                        for bucket_id in buckets:
                            inner_dp_wbuf = self.get_fsdp_buffer(bucket_id, bwd=bwd)
                            shard_size = inner_dp_wbuf.data_size // outer_fsdp_group.size()
                            rank = outer_fsdp_group.rank()
                            torch.distributed.all_gather_into_tensor(
                                output_tensor=inner_dp_wbuf.data,
                                input_tensor=inner_dp_wbuf.data[
                                    rank * shard_size : (rank + 1) * shard_size
                                ],
                                group=outer_fsdp_group,
                            )
                
                all_gather_stream.wait_stream(self.outer_fsdp_group_param_gather_stream)

            
            all_gather_stream.wait_stream(torch.cuda.current_stream())
            dp_group = self.get_fsdp_buffer(buckets[0]).data_parallel_group
            with torch.cuda.stream(all_gather_stream):
                with _coalescing_manager(
                    dp_group, async_ops=async_param_gather
                ) as coalescing_event:
                    for bucket_id in buckets:
                        
                        
                        self.async_bucket_gather(bucket_id, bwd)

            
            for bucket_id in buckets:
                bucket_key = self.get_bucket_key(bucket_id, bwd)
                _, mark_bucket_ready_to_use = self.param_gather_event_map[bucket_key]
                self.param_gather_event_map[bucket_key] = (
                    coalescing_event,
                    mark_bucket_ready_to_use,
                )

        
        if not async_param_gather:
            for bucket_id in buckets:
                self.wait_bucket_ready(bucket_id, bwd)

    def wait_bucket_ready(self, bucket_id, bwd, empty_ok=False):
        """Wait for the bucket to be ready."""
        bucket_key = self.get_bucket_key(bucket_id, bwd)

        if self.bucket_status[bucket_key] == BucketStatus.READY_TO_USE:
            
            return
        if self.bucket_status[bucket_key] in (BucketStatus.EMPTY, BucketStatus.PRESERVED):
            if empty_ok:
                return
            
            
            
            raise ValueError(f"Bucket {bucket_id} is {self.bucket_status[bucket_key].name}.")

        
        param_gather_event, mark_bucket_ready_to_use = self.param_gather_event_map.pop(bucket_key)
        param_gather_event.wait()
        mark_bucket_ready_to_use()

    @torch.no_grad()
    def release_bucket(self, bucket_id, bwd, lazy: bool = False):
        """
        Release the specified parameter bucket, freeing its associated buffer storage.

        This function marks or frees the memory of a parameter bucket depending on
        whether lazy release is enabled. It ensures that buckets are not released
        while still being communicated or in use by the pipeline.

        Args:
            bucket_id (int): Identifier of the bucket to be released.
            bwd (bool): Indicates if the release is triggered during the backward pass.
            lazy (bool, optional): Determines when the parameter buffer (bucket) is released.
                - If False, the buffer is released immediately.
                - If True, the release is deferred until just before the all-gather pipeline
                requests a new buffer. The delayed release is performed by invoking
                `recycle_unused_buckets`.

        Raises:
            ValueError: If the specified bucket is currently in communication and
                cannot be safely released.

        Notes:
            - Buckets marked as lazy will be released later when the pipeline determines
            they are no longer needed.
            - If the bucket has a transpose weight buffer (used in FP8 backward passes),
            this buffer is freed; otherwise, the model weight buffer is released.
            - This function should NOT be invoked on buckets associated with modules not
            identified as FSDP unit modules, even when weights are sharded in the case of
            `optim_grads_params`. Non-unit modules should remain persistently allocated
            because they do not satisfy FSDP unit module state requirements, e.g. their
            parameters are simultaneously modified or shared with other modules.
        """
        bucket_key = self.get_bucket_key(bucket_id, bwd)
        if self.bucket_status[bucket_key] == BucketStatus.EMPTY:
            return

        if lazy:
            
            self.bucket_can_be_released[bucket_key] = True
            return

        self.wait_bucket_ready(bucket_id, bwd, empty_ok=True)
        if self.bucket_status[bucket_key] == BucketStatus.COMMUNICATING:
            raise ValueError(f"Bucket {bucket_id} is communicating.")

        if bwd and self.buffer.parameter_groups[bucket_id].transpose_weight_buffer is not None:
            buf = self.buffer.parameter_groups[bucket_id].transpose_weight_buffer
        else:
            buf = self.buffer.parameter_groups[bucket_id].model_weight_buffer

        buf.free_bucket_storage()
        self.bucket_status[bucket_key] = BucketStatus.EMPTY

    def recycle_unused_buckets(self):
        """Recycle the unused buckets."""
        for bucket_key, can_be_released in self.bucket_can_be_released.items():
            if can_be_released:
                bucket_id, is_transpose_weight = bucket_key[0], bucket_key[1]
                self.release_bucket(bucket_id, is_transpose_weight)
                self.bucket_can_be_released[bucket_key] = False

    def get_fsdp_buffer(self, bucket_id: int, bwd=False) -> DataParallelBuffer:
        """
        Get the FSDP / DP-Shard buffer with the given bucket ID.
        If bwd=True, return the FSDP transpose buffer instead.
        """
        param_group = self.buffer.parameter_groups[bucket_id]
        if self.buffer.ddp_config.outer_dp_sharding_strategy != "no_shard":
            if bwd and param_group.transpose_weight_buffer is not None:
                return param_group.hfsdp_helper_wtbuf
            else:
                return param_group.hfsdp_helper_wbuf
        if bwd and param_group.transpose_weight_buffer is not None:
            return param_group.transpose_weight_buffer
        else:
            return param_group.model_weight_buffer

    @torch.no_grad()
    def async_bucket_gather(self, bucket_id, bwd) -> None:
        """All-gather the bucket and set the items."""
        bucket_key = self.get_bucket_key(bucket_id, bwd)

        self.bucket_can_be_released[bucket_key] = False
        if self.bucket_status[bucket_key] in (
            BucketStatus.COMMUNICATING,
            BucketStatus.READY_TO_USE,
        ):
            return

        self.bucket_status[bucket_key] = BucketStatus.COMMUNICATING

        
        
        wbuf = self.get_fsdp_buffer(bucket_id, bwd)

        
        self.recycle_unused_buckets()

        
        bucket = wbuf.fetch_bucket(set_param_data=True)

        
        
        param_gather_event = torch.distributed.all_gather_into_tensor(
            output_tensor=bucket.data,
            input_tensor=wbuf.get_shard_from_local_buffer(),
            group=wbuf.data_parallel_group,
            async_op=True,
        )

        def get_closure(bucket_id, bwd):
            @torch.no_grad()
            def mark_bucket_ready_to_use():
                
                self.bucket_status[self.get_bucket_key(bucket_id, bwd)] = BucketStatus.READY_TO_USE

            return mark_bucket_ready_to_use

        mark_bucket_ready_to_use = get_closure(bucket_id, bwd)

        
        self.param_gather_event_map[self.get_bucket_key(bucket_id, bwd)] = (
            param_gather_event,
            mark_bucket_ready_to_use,
        )


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """
    Gradient reduce preprocessing for gradient averaging and gradient scaling.
    """

    
    
    
    if scaling_factor is None:
        
        reduce_op = torch.distributed.ReduceOp.SUM
    elif ddp_config.average_in_collective:
        
        reduce_op = torch.distributed.ReduceOp.AVG
    elif ddp_config.gradient_reduce_div_fusion and grad_data.dtype != torch.bfloat16:
        
        reduce_op = torch.distributed._make_nccl_premul_sum(scaling_factor)
    else:
        
        grad_data.mul_(scaling_factor)
        reduce_op = torch.distributed.ReduceOp.SUM

    return reduce_op


def _check_nan_in_grad(grad: torch.Tensor):
    """
    Check if there are any NaN or Inf in grad.
    """
    
    grad_norm = torch.linalg.norm(grad)
    if torch.isnan(grad_norm) or not torch.isfinite(grad_norm):
        raise ValueError(
            f"[Megatron-FSDP](check_for_nan_in_grad=True) Detected NaN or Inf in wgrad: {grad}"
        )


def check_gpu_memory(threshold=0.9):
    """
    Check if the GPU memory is over the threshold.
    Args:
        threshold (float, optional): The threshold to check if the GPU memory is over.
            Defaults to 0.9.
    Returns:
        bool: True if the GPU memory is over the threshold.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    allocated_ratio = allocated / total
    reserved_ratio = reserved / total

    near_full = allocated_ratio >= threshold or reserved_ratio >= threshold

    if near_full:
        log_single_rank(
            logger,
            logging.INFO,
            f"GPU Memory: Allocated: {allocated_ratio:.2%}, Reserved: {reserved_ratio:.2%}",
        )
    return near_full


class ResetParametersContext:
    """
    Context manager for resetting parameters for meta device initialization module.
    """

    def __init__(self, init_param_with_fp8=False, with_cuda_rng_tracker=False):
        self.init_param_with_fp8 = init_param_with_fp8
        self.with_cuda_rng_tracker = with_cuda_rng_tracker

    def __enter__(self):
        self.stack = ExitStack()
        if self.init_param_with_fp8:
            
            
            
            
            assert (
                HAVE_TE
            ), "TransformerEngine is required for using FP8 parameters with Megatron-FSDP."
            
            
            te_quantized_model_init_cls = get_quantized_model_init_context_cls()
            if te_quantized_model_init_cls is not nullcontext:
                
                args = {"enabled": True}
                if (
                    "preserve_high_precision_init_val"
                    in inspect.signature(te_quantized_model_init_cls).parameters
                ):
                    
                    args["preserve_high_precision_init_val"] = True
                self.stack.enter_context(te_quantized_model_init_cls(**args))

        if self.with_cuda_rng_tracker:
            
            
            cuda_rng_tracker = get_cuda_rng_tracker()
            if _MODEL_PARALLEL_RNG_TRACKER_NAME in cuda_rng_tracker.states_:
                
                self.stack.enter_context(get_cuda_rng_tracker().fork())

        return self

    def __exit__(self, *exc_details):
        self.stack.__exit__(*exc_details)


def override_sharded_param_methods_with_safety_checks(params, all_gather_pipeline):
    """
    Override the methods of the parameters to prevent undefined behavior.
    Args:
        params (List[torch.Tensor]): The parameters to add hint on shard to functions.
        all_gather_pipeline (AllGatherPipeline): The all-gather pipeline.
    """
    for p in params:
        to_function = p.to
        cpu_function = p.cpu

        def override_sharded_param_to_function_closure(p, to_function):
            def override_sharded_param_to_function(*args, **kwargs):
                if p._typed_storage()._size() == 0:
                    warnings.warn(
                        "The parameter may be sharded by Megatron-FSDP, "
                        "no actual 'to' operation is performed."
                    )
                    return torch.empty([])
                return to_function(*args, **kwargs)

            return override_sharded_param_to_function

        setattr(p, "to", override_sharded_param_to_function_closure(p, to_function))

        def override_sharded_param_cpu_function_closure(p, cpu_function):
            def override_sharded_param_cpu_function(*args, **kwargs):
                if p._typed_storage()._size() == 0:
                    warnings.warn(
                        "The parameter may be sharded by Megatron-FSDP, "
                        "no actual 'cpu' operation is performed."
                    )
                    return torch.empty([], device="cpu")
                return cpu_function(*args, **kwargs)

            return override_sharded_param_cpu_function

        setattr(p, "cpu", override_sharded_param_cpu_function_closure(p, cpu_function))


def _dtype_size(dtype: torch.dtype) -> int:
    """
    Get the size of the dtype. Note that many data-types un-common to ML
    or not supported by NCCL communication (e.g. CFloat) are listed here
    for mixed-precision coverage and to avoid allocating a dummy Tensor.

    Args:
        dtype (torch.dtype): The dtype to get the size of.
    Returns:
        int: The size of the dtype.
    """
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.int16:
        return 2
    elif dtype == torch.float32 or dtype == torch.int32 or torch.complex32:
        return 4
    elif dtype == torch.float64 or dtype == torch.int64 or torch.complex64:
        return 8
    elif dtype == torch.uint8 or dtype == torch.int8:
        return 1
    elif dtype == "float8":
        return 1
    else:
        try:
            
            
            
            return torch.empty((), dtype=dtype).element_size()
        except:
            raise ValueError(f"Unsupported dtype: {dtype}")


def to_local_if_dtensor(tensor):
    """
    Convert a DTensor to a local tensor.
    Args:
        tensor (torch.Tensor): The tensor to convert.
    Returns:
        torch.Tensor: The local tensor.
    """
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def _get_fsdp_tensor_spec(
    param, dist_index: FSDPDistributedIndex, is_sharded_param, is_expert_param
):
    """
    Get the DeviceMesh for the parameter and modify the placement for Megatron-FSDP.
    """
    
    if isinstance(param, DTensor) and cast(DTensor, param)._spec.num_shards > 1:
        
        dtensor_spec = cast(DTensor, param)._spec

        
        assert len(dtensor_spec.placements) == 1, (
            "When using DTensor with Megatron-FSDP, the DTensorSpec should have only one placement."
            f"Current placements: {dtensor_spec.placements}"
        )
        dtensor_placement = dtensor_spec.placements[0]

        if dist_index.use_hybrid_fsdp:
            mesh_dim_names = (dist_index.dp_outer_dim, dist_index.dp_shard_dim, dist_index.tp_dim)
        else:
            mesh_dim_names = (dist_index.dp_shard_dim, dist_index.tp_dim)

        
        
        
        shard_order = None
        if not is_sharded_param:
            if dist_index.use_hybrid_fsdp:
                placements = [Replicate(), Replicate(), dtensor_placement]
            else:
                placements = [Replicate(), dtensor_placement]
        elif dist_index.use_hybrid_fsdp:
            if dist_index.hsdp_outer_dp_shard:
                
                placements = [Shard(0), Shard(0), dtensor_placement]
                shard_order = [2, 1, 0]
            else:
                placements = [Replicate(), Shard(0), dtensor_placement]
                shard_order = [2, 1, 0]
        else:
            placements = [Shard(0), dtensor_placement]
            shard_order = [1, 0]

        device_mesh = dist_index.get_submesh(mesh_dim_names, is_expert_parallel=is_expert_param)
        if shard_order is not None:
            setattr(device_mesh, "_shard_order", shard_order)

        return device_mesh, placements

    shard_order = None

    if dist_index.use_hybrid_fsdp:
        mesh_dim_names = (dist_index.dp_outer_dim, dist_index.dp_shard_dim)
    else:
        mesh_dim_names = (dist_index.dp_shard_dim,)

    if not is_sharded_param:
        placements = [Replicate()]
    elif dist_index.use_hybrid_fsdp:
        
        if dist_index.hsdp_outer_dp_shard:
            placements = [Shard(0), Shard(0)]
            shard_order = [1, 0]
        else:
            placements = [Replicate(), Shard(0)]
    else:
        placements = [Shard(0)]

    device_mesh = dist_index.get_submesh(mesh_dim_names, is_expert_parallel=is_expert_param)
    if shard_order is not None:
        setattr(device_mesh, "_shard_order", shard_order)

    return device_mesh, placements


def make_fsdp_dtensor(
    local_tensor: torch.Tensor,
    param: torch.nn.Parameter,
    dist_index: FSDPDistributedIndex,
    is_sharded_param: bool = True,
    is_expert_param: bool = False,
    run_check: bool = False,
    update_uneven_dtensor_chunk_meta: bool = False,
    force_sync_tp_duplicated_param: bool = False,
):
    """
    Creates a distributed tensor (DTensor) from a local tensor with support for
    Megatron-FSDP and Tensor Parallel scenarios.

    This function is typically used in a FSDP setup where tensor data needs to be converted
    into sharded DTensors across a device mesh. It also supports model configurations
    involving tensor model parallelism such as Megatron-Core.

    Args:
        local_tensor (torch.Tensor): The local tensor data to be converted to a DTensor.
        param (nn.Parameter): Template parameter used to infer shape, stride,
            and partition attributes.
        dist_index (FSDPDistributedIndex): Metadata object providing the distributed device mesh.
        is_sharded_param (bool, optional): Whether the parameter is sharded across
            devices. Defaults to True.
        is_expert_param (bool, optional): Indicates if the tensor corresponds to
            Megatron-Core expert (Mixture-of-Experts) parameters. Defaults to False.
        run_check (bool, optional): Enables additional internal validation for
            DTensor. Defaults to False.
        update_uneven_dtensor_chunk_meta (bool, optional): Whether to update metadata
            for uneven chunk distributions. Defaults to False.

    Returns:
        DTensor: A DTensor object sharded appropriately across devices.

    Example:
        >>> import torch
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torch.distributed._tensor import DeviceMesh
        >>> from my_fsdp_utils import FSDPDistributedIndex  # assumed utility
        >>>
        >>> # Initialize device mesh (4 GPUs)
        >>> device_mesh = DeviceMesh("cuda", (2, 2), dim_names=("tp", "dp"))
        >>> dist_index = FSDPDistributedIndex(
        ...     device_mesh=device_mesh,
        ...     dp_mesh_dim_name="dp",
        ...     tp_mesh_dim_name="tp"
        ... )
        >>>
        >>> # Dummy local tensor and parameter
        >>> local_tensor = torch.randn(8, 16, device="cuda")
        >>> param = torch.nn.Parameter(torch.empty(32, 32))
        >>>
        >>> # Attach partition metadata for tensor model parallelism
        >>> param.tensor_model_parallel = True
        >>> param.partition_dim = 0
        >>> param.partition_stride = 1
        >>>
        >>> # Convert to DTensor
        >>> dtensor = make_fsdp_dtensor(
        ...     local_tensor=local_tensor,
        ...     param=param,
        ...     dist_index=dist_index,
        ...     is_sharded_param=True,
        ...     run_check=True
        ... )
        >>> print(dtensor)
        DTensor(sharded(...))

    Note:
        - For tensor model parallel use cases, the `param` object must either:
            * Be a tensor-parallel (TP) DTensor, or
            * Include all of these attributes: `tensor_model_parallel`, `partition_dim`,
                and `partition_stride`.
    """
    
    if not is_sharded_param and param.numel() != local_tensor.numel():
        raise ValueError(
            f"[Megatron-FSDP] Mismatch between param shape {param.shape} and local tensor "
            f"shape {local_tensor.shape}. "
            "If the parameter is not sharded, they must match exactly."
        )

    
    orig_param = param

    
    if not isinstance(param, DTensor) and using_tensor_parallel(
        dist_index, is_expert_parallel=is_expert_param
    ):
        
        assert not isinstance(param, DTensor), (
            "[Megatron-FSDP] Parameter is already a DTensor, yet tensor_model_parallel " "is True."
        )
        
        assert dist_index.tp_dim is not None, (
            "[Megatron-FSDP] TP dimension is missing from DeviceMesh / FSDPDistributedIndex! "
            "Required for Megatron-Core or TransformerEngine modules that use TP. "
            "If TP=1, a trivial TP dimension of size 1 should be provided."
        )
        tp_mesh = dist_index.get_submesh(dist_index.tp_dim, is_expert_parallel=is_expert_param)
        global_shape = list(param.shape)
        if is_mcore_tensor_parallel_duplicated(param):
            placements = [Replicate()]
            if force_sync_tp_duplicated_param:
                if local_tensor.numel() > 0:
                    torch.distributed.broadcast(
                        local_tensor, group=tp_mesh.get_group(), group_src=0
                    )
            elif run_check:
                
                pass
        else:
            tp_dim = get_mcore_tensor_parallel_partition_dim(param)
            assert tp_dim is not None, (
                "[Megatron-FSDP] Parameter is not tensor model parallel, "
                "yet tensor_model_parallel is True."
            )
            placements = [Shard(tp_dim)]
            global_shape[tp_dim] *= tp_mesh.mesh.numel()

        
        param = DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=tp_mesh,
            placements=placements,
            run_check=run_check,
            shape=tuple(global_shape),
            stride=torch.empty(global_shape).stride(),
        )

    
    device_mesh, placements = _get_fsdp_tensor_spec(
        param, dist_index, is_sharded_param=is_sharded_param, is_expert_param=is_expert_param
    )

    
    if len(orig_param.shape) > 1:
        local_shape = (-1, *orig_param.shape[1:])
    else:
        local_shape = (-1,)

    
    fsdp_tensor = DTensor.from_local(
        local_tensor=local_tensor.view(local_shape),
        device_mesh=device_mesh,
        placements=placements,
        run_check=False,
        shape=param.shape,
        stride=param.stride(),
    )

    if run_check:
        validate_uneven_dtensor(fsdp_tensor)

    
    if update_uneven_dtensor_chunk_meta:
        update_uneven_dtensor_chunk_metadata(fsdp_tensor)

    return fsdp_tensor
