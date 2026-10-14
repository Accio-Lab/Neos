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
# limitations under the License.import functools

import functools
import importlib
import logging
from contextlib import contextmanager
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .mixed_precision import (
    MixedPrecisionPolicy,
    fp8_create_transpose_cache,
    fp8_discard_transpose_cache,
    is_float8tensor,
)
from .param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    PrefetchOrder,
    _check_nan_in_grad,
    override_sharded_param_methods_with_safety_checks,
    to_local_if_dtensor,
)
from .utils import FSDPDistributedIndex, log_single_rank

logger = logging.getLogger(__name__)


try:
    
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.utils import is_submodule

    logger.info("Detected Megatron Core, using Megatron-FSDP with Megatron.")
except ImportError:
    
    logger.info("Megatron Core is not installed, Megatron-FSDP will run without Megatron Core.")
    from .distributed_data_parallel_config import DistributedDataParallelConfig
    from .utils import is_submodule


class TrainingState(Enum):
    """States of a FSDP parameter group, which are coupled with
    the sharding activity of parameters and gradients during training."""

    
    FORWARD = auto()
    
    PRE_BACKWARD = auto()
    
    POST_BACKWARD = auto()
    
    
    IDLE = auto()


def setup_delayed_wgrad_acc_hook(module, grad_acc_func):
    """Configure delayed wgrad gradient processing for MoE expert parameters.

    When ``overlap_dispatch_backward_with_experts_wgrad`` is enabled on a TransformerLayer,
    this function:
      1. Marks expert parameters so the normal post-accumulate-grad hook is skipped.
      2. Registers a callback on the MoE layer that invokes FSDP's gradient
         reduce-scatter after the delayed wgrad computation completes.

    Args:
        module: The module being processed in the forward pre-hook. Only
            ``TransformerLayer`` instances with the delayed wgrad config flag
            enabled are affected; all other modules are no-ops.
        process_post_backward_gradients_fn: The FSDP gradient processing function
            (``_process_post_backward_gradients``) to be called after the delayed
            wgrad computation finishes.
    """
    from functools import partial

    need_backward_dw = getattr(module, "need_backward_dw", lambda: False)
    if not need_backward_dw():
        return

    for param in module.parameters():
        if getattr(param, 'skip_backward_post_hook', False):
            param.post_wgrad_grad_acc_hook = partial(grad_acc_func, [param])


class MegatronFSDP(torch.nn.Module):
    """Fully Sharded Data Parallel training.

    A distributed training wrapper that shards model parameters, gradients and optimizer
    states across data parallel workers. Integrates seamlessly with MCore's tensor
    and expert parallelism features, and in native PyTorch.

    We supports following modes:
    - no_shard: Traditional data parallel training without parameter sharding.
    - optim: Shards optimizer states, this is conceptually close to "ZeRO-1", and
        main weights for mixed precision training, meanwhile the following `optim_grads`
        and `optim_grads_params` will also sharding main weights
        during mixed-precision training, omitted without detailed notation.
    - optim_grads: Shards gradients and optimizer states, this is conceptually close to "ZeRO-2".
    - optim_grads_params: Shards parameters, gradients and optimizer states, this
        is conceptually close to "ZeRO-3".

    Key Features:
    - Compatible with MCore's tensor, context and expert parallelism
    - Compatible with Native PyTorch's tensor and context parallelism with DTensor
    - Automatic mixed precision training (BF16/FP8)
    - Gradient accumulation and bucketing
    - Optimized activation recompute with shard-aware communication: When recomputing
        a whole Transformer layer, gather parameters once for both the recomputation
        and backward computation
    - Compatible with MCore's distributed checkpointing, and native PyTorch.

    Args:
        module (torch.nn.Module): Underlying Torch Module.
        dist_index (FSDPDistributedIndex): FSDPDistributedIndex object containing references to the
            process groups and device meshes used by Megatron-FSDP.
        ddp_config (DistributedDataParallelConfig): FullyShardedDataParallel configuration
            dataclass containing a variety of Megatron-derived parameters that control the
            behavior of Megatron-FSDP.
        mixed_precision_policy (megatron_fsdp.MixedPrecisionPolicy): Configuration for
            mixed-precision customization of compute and communications in Megatron-FSDP.
        fsdp_unit_modules (List[torch.nn.Module] | List[str]): List of modules that
            should be treated as an FSDP Unit, i.e. the minimum releasable model unit.
            It affects the granularity of the communication parameter grouping and
            triggers aggregate collective communication in FP8 mixed precision training.
        device (torch.device): Target device for the sharded model. Used to migrate
            all model parameters to an expected device. If init_model_with_meta_device=True,
            this argument is ignored.
        init_model_with_meta_device (bool): Whether to initialize model parameters
            in shards across all devices of the fsdp_group. Utilized to initialize
            large models that do not fit on a single device.
        sync_model_each_microbatch (bool): Whether to sync parameters and install gradients on
            each training step. When disabled, Megatron-FSDP will overlap reduce-scatter with
            subsequent compute and delay HSDP gather and reduce operations per optimization cycle,
            which improves performance and throughput when using delayed optimization strategies
            such as gradient accumulation. Defaults to True, can be modified before the model
            forward / backward pass via MegatronFSDP.set_model_auto_sync(bool) or controlled
            with the (no_)sync context managers or microbatch_count and is_last_microbatch.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket.
        keep_fp8_transpose_cache (bool): Whether to keep
            the fp8 transpose cache when using Megatron-FSDP. It will use significantly
            more GPU memory but can improve performance.
        nccl_ub (bool): Whether to use NCCL userbuffer for the FSDP communication
            operands, which uses less number of SMs, resulting better overlapped
            computation performance.
            This flag automatically sets fsdp_double_buffer to True, which uses
            additional GPU memory.
        fsdp_double_buffer (bool): Whether to use persistently allocated double buffers
            for the temporary memory needed in the FSDP communication. This flag is
            automatically set to True when nccl_ub is True.
        fsdp_db_use_persist_buf_on_alloc_fail (bool): Whether to fall back to persistent buffer
            allocator when a bucket does not fit FSDP double buffer size.
        disable_symmetric_registration (bool): Whether to disable symmetric (window) registration
            for NCCL userbuffer registration. This option will force to use conventional (local)
            userbuffer registration when nccl_ub is set.
        enable_fine_grained_param_gather (bool): Whether to enable "fine-grained" param all-gather,
            which can improve performance when using MXFP8 parameters with activation recomputation.
        report_nan_in_param_grad (bool): Whether to enable precise NaN-checking for parameter wgrad.
            Can significantly degrade performance. Defaults to False.

    Examples:
        >>> model = GPTModel(config)
        >>> model = MegatronFSDP(
        ...     model,
        ...     dist_index,
        ...     ddp_config,
        ...     fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding],
        ...     device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        ...     init_model_with_meta_device=False,
        ...     disable_bucketing=False,
        ...     keep_fp8_transpose_cache=False,
        ...     nccl_ub=False,
        ...     fsdp_double_buffer=False,
        ...     fsdp_db_use_persist_buf_on_alloc_fail=False,
        ...     disable_symmetric_registration=False,
        ... )
    """

    def __init__(
        self,
        module: torch.nn.Module,
        dist_index: FSDPDistributedIndex,
        ddp_config: DistributedDataParallelConfig = None,
        mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
        fsdp_unit_modules: Optional[List[torch.nn.Module] | List[str]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        calculate_per_token_loss: bool = False,
        init_model_with_meta_device: bool = False,
        sync_model_each_microbatch: bool = False,
        keep_fp8_transpose_cache: bool = False,
        nccl_ub: bool = False,
        fsdp_double_buffer: bool = False,
        fsdp_db_use_persist_buf_on_alloc_fail: bool = False,
        disable_symmetric_registration: bool = False,
        enable_fine_grained_param_gather_hook: bool = False,
        enable_fine_grained_param_gather_backward_hook: bool = False,
        report_nan_in_param_grad: bool = False,
    ):
        super().__init__()
        
        self.device = (
            device if device is not None else torch.device(f"cuda:{torch.cuda.current_device()}")
        )
        if self.device != torch.device(f"cuda:{torch.cuda.current_device()}"):
            logger.warning(
                f"[Rank {torch.distributed.get_rank()}] Megatron-FSDP is "
                f"using device {self.device} instead of the current device "
                f"{torch.device(f'cuda:{torch.cuda.current_device()}')}, "
                "which may cause process-to-device mapping issues or "
                "cross-device Tensor operation errors. If necessary, "
                "send all Tensors in the module to the Megatron-FSDP "
                f"device ({self.device}) during initialization or to "
                "the device used by corresponding Tensors during "
                "operations of the module forward pass."
            )
        
        
        
        self.module = (
            
            module.to(self.device)
            if device is not None and not init_model_with_meta_device
            else module
        )

        
        
        if ddp_config is None:
            self.ddp_config = DistributedDataParallelConfig(
                data_parallel_sharding_strategy="optim_grads_params",
                outer_dp_sharding_strategy="no_shard",
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=False,
                keep_fp8_transpose_cache=keep_fp8_transpose_cache,  
                nccl_ub=nccl_ub,
                fsdp_double_buffer=fsdp_double_buffer or nccl_ub,
                fsdp_db_use_persist_buf_on_alloc_fail=fsdp_db_use_persist_buf_on_alloc_fail,
                disable_symmetric_registration=disable_symmetric_registration,
                check_for_nan_in_grad=False,
            )
        else:
            self.ddp_config = ddp_config
        self.data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        self.mp_policy = mixed_precision_policy
        self.calculate_per_token_loss = calculate_per_token_loss
        self.init_model_with_meta_device = init_model_with_meta_device
        self.enable_fine_grained_param_gather_hook = enable_fine_grained_param_gather_hook
        self.enable_fine_grained_param_gather_backward_hook = (
            enable_fine_grained_param_gather_backward_hook
        )
        self.report_nan_in_param_grad = report_nan_in_param_grad

        
        
        
        self.dist_index = dist_index

        
        
        
        
        
        
        self.set_model_auto_sync(sync_model_each_microbatch)

        
        has_expert_parameters = self._check_module_parameter_types()

        
        if (
            has_expert_parameters
            and self.dist_index.get_fsdp_group(is_expert_parallel=True) is None
        ):
            raise ValueError(
                "[Megatron-FSDP] Megatron Expert Parallelism is enabled, but no expt_dp_group is"
                "provided."
            )

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None

        
        self.fsdp_unit_modules = (
            [
                (self._import_class_from_path(cls_path) if isinstance(cls_path, str) else cls_path)
                for cls_path in fsdp_unit_modules
            ]
            if fsdp_unit_modules is not None
            else []
        )

        
        self.is_delay_grad_reduce = self.data_parallel_sharding_strategy in ["no_shard", "optim"]
        if self.data_parallel_sharding_strategy == "optim_grads_params":
            
            self.ddp_config.overlap_param_gather = True
        if self.ddp_config.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            
            self.ddp_config.overlap_grad_reduce = True
        if not self.is_delay_grad_reduce:
            
            
            assert self.ddp_config.overlap_grad_reduce

        for param in self.module.parameters():
            if not hasattr(param, "grad_added_to_main_grad"):
                
                
                param.grad_added_to_main_grad = False
            if not hasattr(param, "__fsdp_param__"):
                
                
                param.__fsdp_param__ = True

        self._init_fsdp_param_and_grad_buffer()
        self._register_fsdp_hooks(self.module)
        self.microbatch_count = 0

        
        
        
        self.is_param_fsdp_distributed = False
        self._replace_param_with_distributed_if_needed()
        for param in self.module.parameters():
            
            setattr(param, "_megatron_fsdp_model", self)

    def _check_module_parameter_types(self):
        """
        Check if the module parameters include special parameters
        such as Megatron-Core Expert Parallel (EP/EXPT) parameters.
        """
        expert_params = False
        for _, param in self.module.named_parameters():
            if not getattr(param, "allreduce", True):
                expert_params = True
            if expert_params:
                
                break
        return expert_params

    def _init_fsdp_param_and_grad_buffer(self):
        if self.calculate_per_token_loss:
            
            gradient_scaling_factor = None
            expert_gradient_scaling_factor = None
        else:
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = (
                    self.dist_index.get_dp_group(is_expert_parallel=True).size()
                    / self.dist_index.get_dp_group().size()
                )
            else:
                data_parallel_world_size = self.dist_index.get_dp_group().size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        
        self.param_and_grad_buffer = ParamAndGradBuffer(
            self.ddp_config,
            self.module,
            bucketing_policy=BucketingPolicy(
                suggested_bucket_size=self.bucket_size,
                fsdp_unit_modules=self.fsdp_unit_modules,
                data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            ),
            dist_index=self.dist_index,
            mixed_precision_policy=self.mp_policy,
            gradient_scaling_factor=gradient_scaling_factor,
            expert_gradient_scaling_factor=expert_gradient_scaling_factor,
            device=self.device,
            reset_parameters_for_meta_device_init_module=self.init_model_with_meta_device,
        )
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.raw_param = dict(self.module.named_parameters())

        
        self.side_stream_for_buffer_copy_and_grad_accum = torch.cuda.Stream()
        self.side_stream_for_param_gather = torch.cuda.Stream()

        
        self.grad_reduce_pipeline = GradReducePipeline(
            self.param_and_grad_buffer, rs_stream=self.side_stream_for_buffer_copy_and_grad_accum
        )

        
        self.all_gather_pipeline = AllGatherPipeline(
            self.param_and_grad_buffer, ag_stream=self.side_stream_for_param_gather
        )

        
        suggested_communication_unit_size = self.ddp_config.suggested_communication_unit_size
        if suggested_communication_unit_size is None:
            if self.data_parallel_sharding_strategy == "optim_grads_params":
                total_param_elements = 0
                total_fsdp_module = 0
                for module in self.module.modules():
                    if isinstance(module, tuple(self.fsdp_unit_modules)):
                        total_fsdp_module += 1
                        total_param_elements += sum(p.numel() for p in module.parameters())
                
                
                
                suggested_communication_unit_size = total_param_elements // total_fsdp_module * 2
            elif self.bucket_size is not None:
                suggested_communication_unit_size = self.bucket_size
            else:
                suggested_communication_unit_size = 1_000_000_000

            
            suggested_communication_unit_size = max(
                1_000_000_000, suggested_communication_unit_size
            )

        self.suggested_RS_queue_capacity = suggested_communication_unit_size
        self.suggested_AG_prefetch_size = suggested_communication_unit_size // 2

        if self.data_parallel_sharding_strategy == "optim_grads_params":
            override_sharded_param_methods_with_safety_checks(
                self.module.parameters(), self.all_gather_pipeline
            )

    def _import_class_from_path(self, class_path: str):
        """Helper function to import classes from string paths."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls

    def all_gather_and_wait_parameters_ready(
        self,
        params,
        prefetch=True,
        prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
        wait_bucket_ready=True,
        bwd=False,
    ):
        """
        All-gather parameters across the data parallel group and wait for
        the all-gather operation to complete.
        """
        if self.data_parallel_sharding_strategy == "no_shard":
            return

        ag_pipeline = self.all_gather_pipeline
        
        
        
        ag_pipeline.all_gather_params(
            params=params,
            prefetch=prefetch,
            prefetch_order=prefetch_order,
            suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
            outer_fsdp_group_param_gather=(
                
                
                
                
                self.dist_index.use_hybrid_fsdp
                and self.ddp_config.outer_dp_sharding_strategy != "no_shard"
                and (self.microbatch_count == 0 or self.model_auto_sync)
            ),
            bwd=bwd,
        )
        if wait_bucket_ready:
            for param in params:
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                ag_pipeline.wait_bucket_ready(bucket_id, bwd)
                if bwd and is_float8tensor(param):
                    fp8_create_transpose_cache(param)

        for param in params:
            
            
            param.grad_added_to_main_grad = False
            
            
            param.__fsdp_param__ = True
            
            
            
            
            param.overwrite_main_grad = True

    def _register_fsdp_hooks(self, root_module):
        """Register necessary hooks for Fully Sharded Data Parallel (FSDP) execution on the model.

        This function sets up various hooks required for FSDP operations, including parameter
        resharding/unsharding and gradient handling. The registered hooks are:
            - Pre-forward hook: Unshards parameters before forward pass
            - Post-forward hook: Reshards parameters after forward pass
            - Pre-backward hook: Unshards parameters before backward pass
            - Post-backward hook: Reshards parameters and reduces gradients after backward pass

        Args:
            root_module: The PyTorch module to register FSDP hooks on

        Note:
            These hooks are essential for FSDP's memory efficiency as they manage:
            1. Dynamic parameter sharding/unsharding to reduce memory footprint
            2. Proper gradient synchronization across distributed processes
            3. Gradient accumulation for large batch training

        Returns:
            None
        """

        
        for m in root_module.modules():
            setattr(m, "_training_state", TrainingState.IDLE)

        self.forward_pre_hooks = {}
        self.forward_hooks = {}
        self.backward_pre_hooks = {}
        self.grad_acc_hooks = {}

        """
        An FSDP unit is a module designed to manage the lifecycle of model parameters
        in Fully Sharded Data Parallel (FSDP) training. It ensures that parameters
        are only used within the module and are released immediately after
        the forward and backward computations are completed.
        This approach is crucial for efficient memory management, as releasing
        parameters too early can lead to issues if other computations depend on them.

        `optim` and `optim_grads` do not require FSDP units because they do not
        shard model parameters.
        """
        fsdp_unit_modules = self.fsdp_unit_modules

        def release_module_parameters(module, bwd, lazy=False, *unused):
            """
            Release the parameters of a given module after completing the forward
            and backward passes.

            Args:
                module: The module whose parameters should be released.
                bwd (bool): Indicates if the release is triggered during the backward pass.
                lazy (bool, optional): Determines when the parameter buffer (bucket) is released.
                    - If False, the buffer is released immediately.
                    - If True, the release is deferred until just before the all-gather pipeline
                    requests a new buffer. The delayed release is performed by invoking
                    `recycle_unused_buckets`.
                *unused: Placeholder for any unused arguments.

            Notes:
                - The function maps each parameter to its corresponding buffer group,
                then releases the associated bucket through the all-gather pipeline.
                - If `ddp_config.keep_fp8_transpose_cache` is False, it also clears
                the FP8 transpose cache associated with the module’s parameters.
            """
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                self.all_gather_pipeline.release_bucket(bucket_id, bwd, lazy=lazy)

            if not self.ddp_config.keep_fp8_transpose_cache:
                release_params_fp8_transpose_cache(module.parameters())

        def release_params_fp8_transpose_cache(params):
            for param in params:
                if is_float8tensor(param):
                    fp8_discard_transpose_cache(param)

        def _grad_acc(param):
            """
            Accumulate the gradient in the main_grad buffer.

            Utilizes the patched main_grad property of the parameter to allocate
            or fetch the main gradient bucket for the parameter.
            """
            group_id = self.param_and_grad_buffer.param_to_param_group[param]
            group = self.param_and_grad_buffer.parameter_groups[group_id]
            if not group.requires_grad:
                return

            
            gbuf = group.hfsdp_helper_gbuf if group.hfsdp_helper_gbuf else group.main_grad_buffer
            if gbuf.is_data_distributed:
                
                
                if not param.grad_added_to_main_grad:
                    
                    
                    
                    if self.ddp_config.fsdp_double_buffer:
                        self.grad_reduce_pipeline._enforce_double_buffer_limit([group_id])

                    param.main_grad = param.get_main_grad()
                    if param.grad is not None:
                        if self.report_nan_in_param_grad:
                            _check_nan_in_grad(to_local_if_dtensor(param.grad))
                        
                        
                        param.main_grad.copy_(to_local_if_dtensor(param.grad))
                        del param.grad
                    else:
                        param.main_grad.zero_()
            
            else:
                if not param.grad_added_to_main_grad:
                    if param.grad is not None:
                        if self.report_nan_in_param_grad:
                            _check_nan_in_grad(to_local_if_dtensor(param.grad))
                        
                        
                        param.main_grad = param.get_main_grad()
                        param.main_grad.add_(to_local_if_dtensor(param.grad))
                        del param.grad

            if param.grad_added_to_main_grad and param.grad is not None:
                del param.grad

            
            param.grad_added_to_main_grad = False

        self._params_require_handle_grad = set()

        def _post_backward_release_module(module, *unused):
            """
            Post-backward hook for an FSDP unit to release parameters and process
            its gradients after the backward pass.

            This hook:
            - Validates that the module is an FSDP unit and that the data-parallel
            sharding strategy is ``"optim_grads_params"``.
            - Releases the module's parameters for the backward phase to free memory.
            - Marks the module as IDLE in the training state machine.
            """
            assert isinstance(module, tuple(fsdp_unit_modules))
            assert self.data_parallel_sharding_strategy == "optim_grads_params"

            
            release_module_parameters(module, bwd=True)
            release_module_parameters(module, bwd=False)

            
            for sub_module in module.modules():
                sub_module._training_state = TrainingState.IDLE

        @torch.compiler.disable
        def _process_post_backward_gradients(param_list):
            """
            Process gradients for a list of parameters after the backward pass.

            This helper accumulates gradients into the main_grad buffer and, when
            appropriate, launches asynchronous reduce-scatter operations according
            to the data-parallel sharding strategy and training phase.

            Args:
                param_list (List[torch.nn.Parameter]): Parameters whose gradients
                    should be processed.

            Behavior:
                - Skips processing for shared parameters (those with ``_is_shared=True``),
                since their gradients are handled by the root post-backward hook.
                - Determines whether to reduce gradients based on:
                    * Data-parallel sharding strategy (``"optim_grads"`` or
                        ``"optim_grads_params"``).
                    * Whether this is the last microbatch of the iteration.
                    * Whether ``model_auto_sync`` is enabled.
                - When reduction conditions are met, performs an asynchronous
                reduce-scatter of gradients prior to the optimizer step, which
                requires a subsequent call to ``finish_grad_sync()`` to complete.
                - Marks parameters as processed by adding them to
                    ``_params_require_handle_grad``.

            Notes:
                - With gradient-sharding strategies, gradient reduction occurs on
                every backward propagation.
                - Without gradient sharding, gradient reduction is deferred until
                the last microbatch or when auto-sync is enabled.
                - In hybrid FSDP configurations, an outer FSDP group gradient reduction
                may be triggered.
            """
            
            param_list = [p for p in param_list if not getattr(p, "_is_shared", False)]

            
            for p in param_list:
                if getattr(p, 'skip_backward_post_hook', False):
                    assert hasattr(
                        p, 'post_wgrad_grad_acc_hook'
                    ), "Missing grad accumulation hook for delayed_wgrad_compute param."

            if not param_list:
                return

            for param in param_list:
                _grad_acc(param)

            grad_reduce_every_bprop = self.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            is_last_microbatch = getattr(self, "is_last_microbatch", False)

            if grad_reduce_every_bprop or is_last_microbatch or self.model_auto_sync:
                
                
                
                self.grad_reduce_pipeline.reduce_gradients(
                    param_list,
                    suggested_queue_capacity=self.suggested_RS_queue_capacity,
                    outer_fsdp_group_grad_reduce=(
                        
                        self.dist_index.use_hybrid_fsdp
                        and (is_last_microbatch or self.model_auto_sync)
                    ),
                )

            
            for param in param_list:
                self._params_require_handle_grad.discard(param)

        @torch.compiler.disable
        def _pre_forward_param_unshard(module: nn.Module, *unused):
            
            input_training_state = module._training_state
            fsdp_forward_prefetch = True
            if input_training_state == TrainingState.PRE_BACKWARD:
                
                fsdp_forward_prefetch = False
            else:
                module._training_state = TrainingState.FORWARD

            if isinstance(module, tuple(fsdp_unit_modules)):
                param_list = list(module.parameters())
            else:
                
                
                
                param_list = list(module.parameters(recurse=False))

            if self.enable_fine_grained_param_gather_hook:
                param_list = list(module.parameters(recurse=False))

            
            self.all_gather_and_wait_parameters_ready(
                params=param_list,
                prefetch=fsdp_forward_prefetch,
                prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
            )
            return None

        @torch.compiler.disable
        def _register_post_backward_hook(
            post_backward_hook: callable,
            module: nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            """
            Register a post-backward hook for the given module by inserting an autograd
            Function in front of it. Note that a post-backward hook implemented in this
            way is not compatible with in-place modifications of the module's inputs,
            since such operations can trigger an autograd error that
            "the output is a view and is being modified in-place".
            """
            if not torch.is_grad_enabled():
                
                return args, kwargs

            
            args_list, args_spec = tree_flatten(args)
            kwargs_list, kwargs_spec = tree_flatten(kwargs)
            args_kwargs_list = list(args_list) + list(kwargs_list)
            inp_tensor_indices: List[int] = []
            inp_tensors: List[torch.Tensor] = []
            for i, obj in enumerate(args_kwargs_list):
                if torch.is_tensor(obj) and obj.requires_grad:
                    inp_tensor_indices.append(i)
                    inp_tensors.append(obj)

            if len(inp_tensors) == 0:
                return args, kwargs

            """
            Identity autograd Function that attaches a post-backward "hook" to the
            module, triggering parameter deallocation immediately after the module's
            backward pass has completed in order to shard this layer's model memory
            once the current backward stage is done.
            """
            inp_tensors = RegisterFSDPBackwardFunction.apply(
                functools.partial(post_backward_hook, module), *inp_tensors
            )

            
            for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
                args_kwargs_list[inp_tensor_idx] = inp_tensor
            args_list = args_kwargs_list[: len(args_list)]
            kwargs_list = args_kwargs_list[len(args_list) :]
            args = tree_unflatten(args_list, args_spec)
            kwargs = tree_unflatten(kwargs_list, kwargs_spec)

            
            return args, kwargs

        def _root_post_backward(*unused):
            
            ordered_params = sorted(
                list(self._params_require_handle_grad), key=lambda p: self.param_to_name[p]
            )
            for param in ordered_params:
                _grad_acc(param)

            
            grad_reduce_every_bprop = self.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            
            
            
            
            is_last_microbatch = getattr(self, "is_last_microbatch", False)
            if grad_reduce_every_bprop or is_last_microbatch or self.model_auto_sync:
                self.grad_reduce_pipeline.reduce_gradients(
                    ordered_params,
                    suggested_queue_capacity=self.suggested_RS_queue_capacity,
                    outer_fsdp_group_grad_reduce=(
                        self.dist_index.use_hybrid_fsdp
                        and (is_last_microbatch or self.model_auto_sync)
                    ),
                )
                self.grad_reduce_pipeline.reset()

            
            self._root_pre_backward_hook_issued = False
            self.microbatch_count += 1

            
            
            
            
            
            
            
            if self.model_auto_sync:
                self.finish_grad_sync()

        @torch.compiler.disable
        def _pre_backward_param_unshard(module: nn.Module, *unused):
            """
            Sub-module pre-backward hook to all-gather the module parameters
            before the backward pass.
            """
            
            for sub_module in module.modules():
                sub_module._training_state = TrainingState.PRE_BACKWARD

            if isinstance(module, tuple(fsdp_unit_modules)):
                param_list = list(module.parameters())
            else:
                param_list = list(module.parameters(recurse=False))

            
            self.all_gather_and_wait_parameters_ready(
                param_list, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER, bwd=True
            )

        self._root_pre_backward_hook_issued = False

        def _root_pre_backward(module: nn.Module, *unused, skip_backward_hook: bool = False):
            """Marks the module's training state as PRE_BACKWARD before the
            backprop, this function is registered on the root module.

            This root pre-backward hook informs all modules to skip forward
            pre-fetching in the pre-forward hooks (for activation recomputation)
            and skip weight deallocation / resharding in the post-forward hooks
            during the backward pass, which are instead performed by backward hooks.
            """
            if self._root_pre_backward_hook_issued:
                return
            self._root_pre_backward_hook_issued = True

            if self.data_parallel_sharding_strategy == "optim_grads_params":
                for sub_module in root_module.modules():
                    
                    
                    sub_module._training_state = TrainingState.PRE_BACKWARD
                
                ag_pipeline = self.all_gather_pipeline
                for bucket_id in range(ag_pipeline.num_buckets):
                    group = self.param_and_grad_buffer.parameter_groups[bucket_id]
                    if group.fsdp_unit_id is not None:
                        ag_pipeline.bucket_can_be_released[
                            ag_pipeline.get_bucket_key(bucket_id, bwd=False)
                        ] = True
            
            self._params_require_handle_grad = set()
            for param_group in self.param_and_grad_buffer.parameter_groups:
                if not param_group.requires_grad:
                    continue
                self._params_require_handle_grad |= set(param_group.params)
                for param in param_group.params:
                    param.grad_added_to_main_grad = False
            
            
            if skip_backward_hook:
                return
            torch.autograd.Variable._execution_engine.queue_callback(_root_post_backward)

        @torch.compiler.disable
        def _post_forward(module: nn.Module, input: Any, output: Any):
            
            
            
            if module._training_state == TrainingState.PRE_BACKWARD:
                
                
                
                
                lazy_release = True
            else:
                lazy_release = False
                module._training_state = TrainingState.IDLE

            assert isinstance(
                module, tuple(fsdp_unit_modules)
            ), "_post_forward hook should only be registered on FSDP unit modules."

            
            release_module_parameters(module, bwd=False, lazy=lazy_release)

            return output

        @torch.compiler.disable
        def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
            release_params_fp8_transpose_cache(module.parameters(recurse=False))

        def create_custom_backward_hook(module, custom_backward_handler):
            """
            Creates a custom backward hook via attaching a gradient-triggered hook
            to the output tensor(s) of a module during a post-forward hook.
            """

            @torch.compiler.disable
            def forward_hook(_module, inputs, output):
                
                
                
                
                output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)

                output_list = []

                
                if isinstance(output, torch.Tensor):
                    output_list = [output]
                elif isinstance(output, (tuple, list)):
                    output_list = [t for t in output if isinstance(t, torch.Tensor)]

                
                
                
                torch.autograd.graph.register_multi_grad_hook(
                    output_list, lambda grads: custom_backward_handler(_module, grads), mode="any"
                )
                return output

            
            
            return module.register_forward_hook(forward_hook)

        def _register_pre_forward_param_unshard_hook(module):
            """
            Register the forward pre-hook to unshard parameters before the forward pass.
            If we are not sharding anything, we do not have a model weight buffer and thus
            have nothing to all-gather / un-shard.
            """
            if self.data_parallel_sharding_strategy != "no_shard":
                self.forward_pre_hooks[f"{module._get_name()} parameter unshard"] = (
                    module.register_forward_pre_hook(
                        _pre_forward_param_unshard, prepend=True, with_kwargs=True
                    )
                )

        def _register_pre_backward_param_unshard_hook(module):
            """
            Register the backward pre-hook to unshard FSDP unit module parameters
            immediately before the backward pass via attaching a gradient-triggered
            hook to the output tensor(s) of a module during a post-forward hook.
            """
            self.backward_pre_hooks[f"all-gather {module._get_name()} parameters"] = (
                create_custom_backward_hook(module, _pre_backward_param_unshard)
            )

        
        
        
        
        
        self.post_forward_release_module = partial(_post_forward, input=None, output=None)
        self.post_backward_release_module = _post_backward_release_module
        self.pre_backward = partial(_root_pre_backward, module=None, skip_backward_hook=True)
        self.post_backward = _root_post_backward

        fsdp_modules = []
        for name, module in root_module.named_modules():
            
            setup_delayed_wgrad_acc_hook(module, _process_post_backward_gradients)
            if self.enable_fine_grained_param_gather_hook:
                _register_pre_forward_param_unshard_hook(module)
            if self.enable_fine_grained_param_gather_backward_hook:
                _register_pre_backward_param_unshard_hook(module)

            
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue

            if not self.enable_fine_grained_param_gather_hook:
                _register_pre_forward_param_unshard_hook(module)

            if isinstance(module, tuple(fsdp_unit_modules)):
                fsdp_modules.append(module)
                
                
                
                self.forward_hooks[f"release module {name} parameters"] = (
                    module.register_forward_hook(_post_forward, prepend=False)
                )

                _register_pre_backward_param_unshard_hook(module)
            elif (
                not self.ddp_config.keep_fp8_transpose_cache
                and self.data_parallel_sharding_strategy == "optim_grads_params"
            ):
                
                
                
                
                self.forward_hooks[f"remove module {name} fp8 transpose cache"] = (
                    module.register_forward_hook(_release_module_fp8_transpose_cache, prepend=False)
                )

            
            
            if isinstance(module, tuple(fsdp_unit_modules)):
                if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                    self.forward_pre_hooks[f"module {name} register post-backward hook"] = (
                        module.register_forward_pre_hook(
                            functools.partial(
                                _register_post_backward_hook, _post_backward_release_module
                            ),
                            with_kwargs=True,
                        )
                    )
                grad_acc_param_list = [p for p in module.parameters() if p.requires_grad]
            else:
                grad_acc_param_list = [
                    p for p in module.parameters(recurse=False) if p.requires_grad
                ]

            for param in grad_acc_param_list:
                
                if not param.requires_grad:
                    continue
                self.grad_acc_hooks[f"grad_acc and reduce for {self.param_to_name[param]}"] = (
                    param.register_post_accumulate_grad_hook(
                        lambda p: (
                            None
                            if getattr(p, 'skip_backward_post_hook', False)
                            else _process_post_backward_gradients([p])
                        )
                    )
                )

        
        
        
        for name, module in root_module.named_modules():
            if len(list(module.parameters())) != len(list(root_module.parameters())):
                
                continue
            
            self.backward_pre_hooks[f"{name} _root_pre_backward"] = create_custom_backward_hook(
                module, _root_pre_backward
            )
        self._root_pre_backward_hook_handle = create_custom_backward_hook(
            module, _root_pre_backward
        )

        
        
        
        self._load_state_dict_post_hook = self.module.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: self.install_optimized_model_weights()
        )

        
        
        for name, module in self.named_modules():
            module.register_state_dict_pre_hook(
                lambda *args, **kwargs: self._replace_param_with_distributed_if_needed()
            )

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        For grads shard mode there will actually always be gradient sync happening.
        """
        
        
        self.is_last_microbatch = False
        try:
            yield
        finally:
            self.is_last_microbatch = True

    @contextmanager
    def sync(self):
        """
        Context manager that synchronizes the MegatronFSDP model parameters and gradients
        every training step as opposed to every optimization cycle.
        """
        self.set_model_auto_sync(True)
        try:
            yield
        finally:
            self.set_model_auto_sync(False)

    def set_model_auto_sync(self, sync_model: bool = True):
        """
        Activate or deactivate flag that controls Megatron-FSDP model synchronization.
        When activated, the model parameters and gradients will be synchronized EVERY
        training step, i.e. gradient reduction will be waited upon instead of overlapped
        with subsequent compute, and all-gather + reduce operations across the DP-Outer
        ProcessGroup will be executed when sharding on DP-Outer during HSDP / HFSDP.
        Otherwise, MegatronFSDP will perform such synchronizations every optimization
        cycle depending on is_last_microbatch = True or microbatch_count = 0, which
        are more flexible but difficult to manage, e.g. microbatch_count and
        is_last_microbatch can be modified elsewhere for custom training strategies.

        Will commonly be called on the final microbatch of a training step before the
        model forward pass and gradient backward pass to ensure that the model gradients
        (prior to optimizer.step()) and model parameters (prior to dist. checkpointing)
        are synchronized and representative of the model trained at that particular
        training step. Otherwise, model training performance will slightly degrade when
        MegatronFSDP.model_auto_sync = True.

        Args:
            sync_model (bool, optional): Whether to synchronize the model every training step.
                MegatronFSDP.model_auto_sync will be set to the value of sync_model.
                Defaults to True. MegatronFSDP.model_auto_sync defaults to False.
        """
        if sync_model and (
            
            self.data_parallel_sharding_strategy in ["no_shard", "optim"]
            
            or self.dist_index.use_hybrid_fsdp
        ):
            log_single_rank(
                logger_=logger,
                level=logging.WARNING,
                msg="[Megatron-FSDP.set_model_auto_sync()] Detected Megatron-FSDP "
                "auto-synchronization with the 'no_shard' / 'optim' sharding "
                "strategy for either FSDP (DP-Shard) or HSDP (DP-Outer). "
                "`MegatronFSDP.zero_grad_buffer()` or `optimizer.zero_grad()` "
                "must be invoked every forward-backward optimization cycle to "
                "prevent successive in-place gradient buffer reductions from "
                "corrupting previously accumulated gradients.",
            )
        self.model_auto_sync = sync_model

    def get_distributed_index(self) -> FSDPDistributedIndex:
        """
        Get the distributed environment of Megatron-FSDP, which contains references
        to the process groups and device meshes used by Megatron-FSDP.
        """
        return self.dist_index

    @contextmanager
    def mixed_precision_context(self, mixed_precision_policy: MixedPrecisionPolicy):
        """
        Context manager for re-configuring the MixedPrecisionPolicy
        for MegatronFSDP / ParamAndGradBuffer.
        """
        mp_policy_backup = self.mp_policy
        self.reset_mixed_precision_policy(mixed_precision_policy)
        try:
            yield
        finally:
            self.reset_mixed_precision_policy(mp_policy_backup)

    def reset_mixed_precision_policy(self, mixed_precision_policy: MixedPrecisionPolicy):
        """
        Re-configure MixedPrecisionPolicy for MegatronFSDP / ParamAndGradBuffer.
        """
        mp_policy_reset = MixedPrecisionPolicy(
            
            main_params_dtype=self.mp_policy.main_params_dtype,
            main_grads_dtype=self.mp_policy.main_grads_dtype,
            
            
            
            grad_comm_dtype=(
                mixed_precision_policy.grad_comm_dtype
                if self.ddp_config.nccl_ub or self.ddp_config.fsdp_double_buffer
                else self.mp_policy.grad_comm_dtype
            ),
        )
        self.mp_policy = mp_policy_reset
        self.param_and_grad_buffer.mp_policy = mp_policy_reset

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        self._replace_param_with_raw_if_needed()

        if self.data_parallel_sharding_strategy == "no_shard":
            return

        if not force_sync and self.ddp_config.overlap_param_gather:
            
            if self.ddp_config.fsdp_all_gather_in_start_param_sync:
                first_param = list(self.module.parameters())[0]
                self.all_gather_and_wait_parameters_ready(
                    params=[first_param], prefetch=True, wait_bucket_ready=False
                )
        else:
            self.synchronize_param_gather()
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.async_bucket_gather(bucket_id=bucket_id, bwd=False)
                group = self.param_and_grad_buffer.parameter_groups[bucket_id]
                if group.model_weight_buffer is None:
                    continue

                if group.model_weight_buffer.is_data_distributed:
                    
                    
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id, False)

            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.wait_bucket_ready(bucket_id, False)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if not self.ddp_config.overlap_grad_reduce:
            if self.data_parallel_sharding_strategy == "no_shard":
                self.param_and_grad_buffer.all_reduce_gradients(
                    async_op=self.ddp_config.overlap_grad_reduce
                )
            else:
                self.param_and_grad_buffer.reduce_scatter_gradients()

    def synchronize_param_gather(self):
        """
        Synchronize parameter all-gather operations for all model parameters.
        """
        self.all_gather_pipeline.reset(preserve_non_fsdp_units=True)
        self._replace_param_with_distributed_if_needed()

    def synchronize_gradient_reduce(self):
        """
        Synchronize gradient reduce-scatter operations for all model gradients.
        """
        if self.ddp_config.overlap_grad_reduce:
            
            
            self.grad_reduce_pipeline.wait_for_previous_grad_reduce(0)
            self.grad_reduce_pipeline.reset()
        else:
            
            self.start_grad_sync()

    def attach_grad_to_optimizer_state(self):
        """
        Attach gradients to optimizer named parameters
        in preparation for optimizer.step().
        """
        self.param_and_grad_buffer.update_main_grads()

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients. Call prior to the optimization step to resolve
        asynchronous gradient reductions.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.

        NOTE: force_all_reduce is included as an argument to maintain API compatibility
        with DDP.force_grad_sync.
        """
        
        self.synchronize_gradient_reduce()

        
        
        self.attach_grad_to_optimizer_state()

        
        
        if self.ddp_config.overlap_param_gather:
            self.synchronize_param_gather()

        
        
        self._replace_param_with_distributed_if_needed()

        
        self.microbatch_count = 0

    def _replace_param_with_distributed_if_needed(self):
        if self.is_param_fsdp_distributed:
            return
        self.is_param_fsdp_distributed = True

        pg_buffer = self.param_and_grad_buffer
        fsdp_params = dict(pg_buffer.optimizer_named_parameters)
        for name, _ in self.module.named_parameters():
            assert name in fsdp_params, f"Parameter {name} not found in FSDP parameters."
            dist_param = fsdp_params[name]
            
            
            if not hasattr(dist_param, "__fsdp_param__"):
                dist_param.__fsdp_param__ = True
            _replace_module_parameter(self.module, name, dist_param)

        
        self._reestablish_shared_weights(self.raw_param, fsdp_params)

    def _replace_param_with_raw_if_needed(self):
        if not self.is_param_fsdp_distributed:
            return
        self.is_param_fsdp_distributed = False

        for name, _ in self.module.named_parameters():
            assert name in self.raw_param, f"Raw parameter {name} not found in module."
            _replace_module_parameter(self.module, name, self.raw_param[name])

        
        pg_buffer = self.param_and_grad_buffer
        fsdp_params = dict(pg_buffer.optimizer_named_parameters)
        self._reestablish_shared_weights(fsdp_params, self.raw_param)

    def _reestablish_shared_weights(self, old_params, new_params):
        """
        Reestablishes shared (tied) weights in a PyTorch module after parameter replacement.

        When iterating over `named_parameters()`, PyTorch skips parameters that are shared
        via weight-tying (e.g., `lm_head.weight` referencing `wte.weight`). After replacing
        parameters, these shared weights become independent, causing previously hidden
        parameters to appear in the parameter list. This function restores the original
        shared structure by ensuring parameters that were previously tied remain shared.

        Args:
            old_params (dict): Mapping from parameter names to original parameter tensors.
            new_params (dict): Mapping from parameter names to new parameter tensors.
        """
        for name, param in self.module.named_parameters():
            if name in new_params:
                
                continue

            
            shared_param = None
            for old_name, old_weight in old_params.items():
                
                if id(param) == id(old_weight):
                    shared_param = new_params.get(old_name)
                    break
            assert (
                shared_param is not None
            ), f"Parameter {name} not found in new parameters or as a shared weight."

            
            _replace_module_parameter(self.module, name, shared_param)
            setattr(shared_param, "_is_shared", True)  

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        self.param_and_grad_buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration alongside optimizer.zero_grad().
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        self.param_and_grad_buffer.zero_grad()

    def install_optimized_model_weights(self):
        """
        Copies optimized parameter values into the model training parameters
        managed by Megatron-FSDP. Should be called after the optimizer.step().
        """
        self.param_and_grad_buffer.copy_main_weights_to_model_weights()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, "allreduce", True)

            data_parallel_group = self.dist_index.get_dp_group(
                is_expert_parallel=is_expert_parallel
            )
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

    def forward(self, *inputs, **kwargs):
        """
        Wrapped forward pass of the model managed by FSDP.
        """
        self._replace_param_with_raw_if_needed()
        with torch.autograd.profiler.record_function("CustomFSDP.forward"):
            
            output = self.module.forward(*inputs, **kwargs)
            return output


class RegisterFSDPBackwardFunction(torch.autograd.Function):
    """
    Register a backward function that will be called after the backward pass
    of the model. This function is used to release the parameters after the
    backward pass.
    """

    @staticmethod
    def forward(ctx, post_backward, *inputs: torch.Tensor):
        """
        Forward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        """
        Backward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward()
        return (None,) + grads


def _replace_module_parameter(module, name, new_param):
    """
    Replace a module's parameter with a new parameter, preserving the hierarchy.
    """
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:  
        parent = getattr(parent, part)

    
    setattr(parent, parts[-1], new_param)
