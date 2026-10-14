# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Set

import torch

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    HAVE_FSDP = True
except ImportError:
    HAVE_FSDP = False

from torch.distributed import ProcessGroup

from megatron.core.fp8_utils import is_float8tensor

from .. import parallel_state, tensor_parallel
from ..models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from ..models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from ..transformer.transformer_config import TransformerConfig
from ..transformer.transformer_layer import TransformerLayer
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig


class TorchFullyShardedDataParallel(_BaseDataParallel):
    """
    Enables fully sharded data parallelism by wrapping the given model with
    the PyTorch FSDP2 API:
    https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
    To utilize this class, PyTorch version >= 2.4.0 is required.

    Args:
        config: Transformer config object.
        ddp_config: TorchDistributedDataParallel config object.
        module: Underlying model.
        sub_modules_to_wrap: Set of sub_modules to shard with FSDP.
            Parameters within each sub_module will be all-gathered just-in-time.
            The default set includes the following submodules derived from the
            GPT model architecture:
                TransformerLayer (all Transformer layers)
                LanguageModelEmbedding (initial embedding layer)
                RotaryEmbedding  (initial RoPE layer)
                tensor_parallel.ColumnParallelLinear (final output layer)

            User can set _fsdp_modules attribute on submodules to set additional
            submodules to shard with FSDP.
        process_group: Optional ProcessGroup to use for distributed operations.
            If None (default), the data parallel process group will be obtained from
            parallel_state.get_data_parallel_group(with_context_parallel=True).
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        sub_modules_to_wrap: Set[torch.nn.Module] = {
            TransformerLayer,
            LanguageModelEmbedding,
            RotaryEmbedding,
            tensor_parallel.ColumnParallelLinear,
        },
        disable_bucketing: bool = False,
        process_group: Optional[ProcessGroup] = None,
    ):

        assert (
            HAVE_FSDP
        ), 'TorchFullyShardedDataParallel requires PyTorch >= 2.4.0 with FSDP 2 support.'

        super().__init__(config=config, module=module)

        if process_group is None:
            self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        else:
            self.process_group = process_group

        self.device_mesh = DeviceMesh.from_group(self.process_group, "cuda")
        kwargs = {
            "mesh": self.device_mesh,
            "reshard_after_forward": getattr(ddp_config, "reshard_after_forward", True),
        }

        self.ddp_config = ddp_config

        def save_custom_attrs(module):
            custom_attrs = {}
            for name, param in module.named_parameters():
                attrs = vars(param)
                if is_float8tensor(param):
                    
                    
                    
                    attrs['_fp8_attrs']['transpose_invalid'] = False
                    del attrs['_fp8_attrs']['transpose']
                custom_attrs[name] = {k: v for k, v in attrs.items()}
            return custom_attrs

        def restore_custom_attrs(module, custom_attrs):
            for name, param in module.named_parameters():
                if name in custom_attrs:
                    for attr_name, attr_value in custom_attrs[name].items():
                        setattr(param, attr_name, attr_value)

        
        
        attrs = save_custom_attrs(self.module)

        
        if config.transformer_impl == "local":
            sub_modules_to_wrap = [
                sub_module
                for sub_module in sub_modules_to_wrap
                if sub_module != tensor_parallel.ColumnParallelLinear
            ]
        sub_modules_to_wrap = set(sub_modules_to_wrap)
        for sub_module in self.module.modules():
            fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
            for f in fsdp_modules:
                sub_modules_to_wrap.add(f)

        prev_module = None
        for sub_module in self.module.modules():
            
            
            
            if any(
                isinstance(sub_module, sub_module_to_wrap)
                for sub_module_to_wrap in sub_modules_to_wrap
            ):
                fully_shard(sub_module, **kwargs)

                
                
                if config.recompute_granularity is not None:
                    sub_module.set_modules_to_backward_prefetch(
                        [prev_module] if prev_module else []
                    )
                prev_module = sub_module

        
        
        fully_shard(self.module, **kwargs)

        restore_custom_attrs(self.module, attrs)

    def load_state_dict(self, state_dict, strict=True):
        """
        No-op because tensors are already loaded in-place by
        `_load_base_checkpoint` with FSDP2."""
        pass
