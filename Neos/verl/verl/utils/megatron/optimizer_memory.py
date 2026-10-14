# Copyright 2026 Individual Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in memory headroom for Megatron optimizer collectives.

Warm only the optimizer's existing gradient-statistics groups, after initial
offload and before rollout. Never call optimizer.step/get_grad_norm for warmup:
those operate on training state. The scratch scalar below uses no RNG or grads.
"""

import gc
import json
import os

import torch


def optimizer_grad_stats_groups(optimizer):
    """Return leaf-optimizer groups in their original collective order.

    A ChainedOptimizer can have distinct dense/expert groups; its own getter
    asserts if they differ. None means the default WORLD group, not no group.
    Deduplicate by identity, without merging distinct communicators that happen
    to contain the same ranks. Include stub optimizers: they must participate.
    """
    groups = []

    def visit(current):
        children = getattr(current, "chained_optimizers", None)
        if children is not None:
            if not children:
                raise ValueError("Cannot warm an empty ChainedOptimizer")
            for child in children:
                visit(child)
            return
        getter = getattr(current, "get_grad_stats_parallel_group", None)
        if not callable(getter):
            raise TypeError("Optimizer does not expose its gradient-statistics group")
        group = getter()
        if not any(group is previous for previous in groups):
            groups.append(group)

    visit(optimizer)
    return groups


def _memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def _emit(event, **values):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    print(
        "megatron_optimizer_memory "
        + json.dumps({"event": event, "rank": rank, "pid": os.getpid(), **values}, sort_keys=True),
        flush=True,
    )


def release_optimizer_step_cache(*, phase="before_optimizer_step"):
    """Release unused allocator blocks, not live model/gradient/state tensors."""
    if not torch.cuda.is_available():
        raise RuntimeError("Optimizer memory headroom is currently CUDA-only")
    gc.collect()
    torch.cuda.synchronize()
    before = _memory_snapshot()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after = _memory_snapshot()
    report = {"phase": phase, "before": before, "after": after}
    _emit("cache_released", **report)
    return report


def warmup_optimizer_grad_stats(optimizer):
    """Initialize the real grad-norm NCCL communicators with a fp32 SUM."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("Gradient-statistics warmup requires initialized process groups")
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient-statistics warmup is currently CUDA-only")
    groups = optimizer_grad_stats_groups(optimizer)
    sizes = []
    
    for group in groups:
        backend = str(torch.distributed.get_backend(group)).lower()
        size = torch.distributed.get_world_size(group)
        
        
        cuda_uses_nccl = backend == "nccl" or "cuda:nccl" in [part.strip() for part in backend.split(",")]
        if not cuda_uses_nccl or size < 1:
            raise RuntimeError(f"Expected a member NCCL group, got {backend=} {size=}")
        sizes.append(size)
    release_optimizer_step_cache(phase="before_grad_stats_warmup")
    scratch = torch.zeros(1, dtype=torch.float32, device=torch.device("cuda", torch.cuda.current_device()))
    for group in groups:
        torch.distributed.all_reduce(scratch, op=torch.distributed.ReduceOp.SUM, group=group)
        
        torch.cuda.synchronize()
    del scratch
    torch.cuda.empty_cache()
    _emit("grad_stats_warmed", groups=len(groups), group_sizes=sizes, memory=_memory_snapshot())
    return sizes
