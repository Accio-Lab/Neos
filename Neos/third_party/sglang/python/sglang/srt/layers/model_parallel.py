"""
Common utilities for torch model parallelism.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

try:
    import torch.distributed.tensor as dt
except ImportError:
    
    import torch.distributed._tensor as dt

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def _shard_tensor(
    full_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[dt.Shard],
) -> "dt.DTensor":
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        device_mesh (:class:`DeviceMesh`): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, device_mesh, [Shard(1)])
    """
    shape, offset = dt._utils.compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return dt.DTensor.from_local(local_tensor, device_mesh, placements)


class ColwiseParallelSharded(ColwiseParallel):
    """
    A version of ColwiseParallel where the local weight has been already
    sharded.  This is used for the fused wqkv case, where during loading, we
    already sharded wq, wk, wv before fusing them.
    """

    
    def _partition_linear_fn(self, name, module, device_mesh):
        
        
        
        for name, param in module.named_parameters():
            dtensor = dt.DTensor.from_local(param, device_mesh, [dt.Shard(0)])
            dist_param = torch.nn.Parameter(dtensor, requires_grad=False)
            module.register_parameter(name, dist_param)


class RowwiseParallelMaybeWait(RowwiseParallel):
    """
    A version of RowwiseParallel that waits for the output (establish dependency
    between comm stream and compute stream in CUDA sense) before going into the
    next op. This is needed to workaround the current interaction between
    AsyncCollectiveTensor and multi-platform ops, such as `RMSNorm`.
    """

    def _partition_linear_fn(self, name, module, device_mesh):
        
        
        
        module.register_parameter(
            "weight",
            nn.Parameter(_shard_tensor(module.weight, device_mesh, [dt.Shard(1)])),
        )
        if getattr(module, "bias", None) is not None:
            
            module.register_parameter(
                "bias",
                nn.Parameter(
                    dt.distribute_tensor(module.bias, device_mesh, [dt.Replicate()])
                ),
            )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = super(
            RowwiseParallelMaybeWait, RowwiseParallelMaybeWait
        )._prepare_output_fn(
            output_layouts, use_local_output, mod, outputs, device_mesh
        )
        return torch.distributed._functional_collectives.wait_tensor(outputs)


def tensor_parallel(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
):
    """
    Tensor parallelize the model across the given device mesh.
    Args:
        module (`torch.nn.Module`):
            The module to tensor parallelize.
        device_mesh (`torch.distributed.DeviceMesh`):
            The device mesh to use for tensor parallelism.
    """

    
    
    
    
    def tplize(mod: torch.nn.Module) -> None:
        tp_plan = getattr(mod, "_tp_plan", None)
        if tp_plan is None:
            return
        for child_name, tp_style in tp_plan.items():
            submod = mod.get_submodule(child_name)
            if tp_style == "Colwise":
                parallelize_module(submod, device_mesh, ColwiseParallel())
            elif tp_style == "Rowwise":
                parallelize_module(submod, device_mesh, RowwiseParallelMaybeWait())
            elif tp_style == "Colwise_Sharded":
                parallelize_module(submod, device_mesh, ColwiseParallelSharded())
            else:
                raise ValueError(f"Unknown TP style {tp_style}")

    
    
    module.apply(tplize)
