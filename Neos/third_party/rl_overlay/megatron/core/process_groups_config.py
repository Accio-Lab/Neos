# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field, fields
from functools import partial
from typing import List, Optional

import torch

from megatron.core import parallel_state


class ProcessGroupHelperMeta(type):
    """Metaclass to protect virtual_pipeline_model_parallel_size from direct assignment."""

    def __setattr__(cls, name, value):
        if name == 'virtual_pipeline_model_parallel_size':
            raise AttributeError(
                f"Cannot set '{name}' directly. Use set_virtual_pipeline_model_parallel_size() "
                f"method instead."
            )
        super().__setattr__(name, value)


@dataclass
class ProcessGroupCollection:
    """Unified process group collection for transformer model parallelism, gradient communication,
     and finalization.

    Fields use init=False and must be set after instance creation.

    Args:
        # Model Parallelism Groups
        tp: Tensor parallel process group
        pp: Pipeline parallel process group
        mp: Model parallel group (tensor + pipeline)
        embd: Embedding process group
        pos_embd: Position embedding process group
        cp: Context parallel process group
        tp_cp: Tensor and context parallel group
        hcp: Hierarchical context parallel groups
        ep: Expert model parallel group
        expt_tp: Expert tensor parallel group
        tp_ep: Tensor and expert parallel group
        tp_ep_pp: Tensor, expert, and pipeline parallel group

        # Data Parallelism Groups
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        intra_expt_dp: Intra partial expert data parallel group
        inter_dist_opt: Inter distributed optimizer instance group

    Example:
        # Create instance and set needed process groups
        pgs = ProcessGroupCollection()
        pgs.tp = tp_group
        pgs.pp = pp_group
        pgs.dp = dp_group

        # Pass to model components
        model = TransformerModel(..., pg_collection=pgs)
        ddp_model = DistributedDataParallel(..., pg_collection=pgs)
        finalize_model_grads(..., pg_collection=pgs)
    """

    
    
    tp: torch.distributed.ProcessGroup = field(init=False)

    
    pp: torch.distributed.ProcessGroup = field(init=False)

    
    mp: torch.distributed.ProcessGroup = field(init=False)

    
    embd: torch.distributed.ProcessGroup = field(init=False)

    
    pos_embd: torch.distributed.ProcessGroup = field(init=False)

    
    cp: torch.distributed.ProcessGroup = field(init=False)

    
    tp_cp: torch.distributed.ProcessGroup = field(init=False)

    
    hcp: List[torch.distributed.ProcessGroup] = field(init=False)

    
    
    ep: torch.distributed.ProcessGroup = field(init=False)

    
    expt_tp: torch.distributed.ProcessGroup = field(init=False)

    
    tp_ep: torch.distributed.ProcessGroup = field(init=False)

    
    tp_ep_pp: torch.distributed.ProcessGroup = field(init=False)

    
    tp_dp_cp: torch.distributed.ProcessGroup = field(init=False)

    
    
    dp: torch.distributed.ProcessGroup = field(init=False)

    
    dp_cp: torch.distributed.ProcessGroup = field(init=False)

    
    
    
    
    
    expt_dp: torch.distributed.ProcessGroup = field(init=False)

    
    intra_dp_cp: torch.distributed.ProcessGroup = field(init=False)

    
    intra_expt_dp: torch.distributed.ProcessGroup = field(init=False)

    
    inter_dist_opt: torch.distributed.ProcessGroup = field(init=False)

    
    intra_dist_opt: torch.distributed.ProcessGroup = field(init=False)

    def __init__(self, **kwargs):
        for key in kwargs:
            if key in [field.name for field in fields(self)]:
                setattr(self, key, kwargs[key])
            else:
                raise ValueError(f"Unknown attribute: {key}")

    def __repr__(self):
        """Return a concise representation showing which process groups exist and their sizes."""
        active_pgs = []
        for field_info in fields(self):
            if hasattr(self, field_info.name):
                pg = getattr(self, field_info.name)
                if pg is not None:
                    active_pgs.append(f"{field_info.name}({pg.size()})")
                else:
                    
                    active_pgs.append(f"{field_info.name}(None)")
        return (
            f"ProcessGroupCollection({', '.join(active_pgs)})"
            if active_pgs
            else "ProcessGroupCollection(empty)"
        )

    @classmethod
    def use_mpu_process_groups(cls, required_pgs: Optional[List[str]] = None):
        """
        Use the default process groups from parallel_state.

        Args:
            required_pgs (List[str], optional): List of process group names to initialize.
                If None, pull all default process groups. Each string should correspond to
                one of the dataclass process group attributes.
        """
        
        all_pgs = {field.name for field in fields(cls)}

        
        if required_pgs is None:
            required_pgs = list(all_pgs)

        
        invalid_pgs = [pg for pg in required_pgs if pg not in all_pgs]
        if invalid_pgs:
            raise ValueError(f"Invalid process groups requested: {invalid_pgs}")

        
        pg_to_func = {
            'tp': partial(parallel_state.get_tensor_model_parallel_group, check_initialized=False),
            'pp': partial(
                parallel_state.get_pipeline_model_parallel_group, check_initialized=False
            ),
            'mp': partial(parallel_state.get_model_parallel_group, check_initialized=False),
            'cp': partial(parallel_state.get_context_parallel_group, check_initialized=False),
            'tp_cp': partial(
                parallel_state.get_tensor_and_context_parallel_group, check_initialized=False
            ),
            'hcp': partial(
                parallel_state.get_hierarchical_context_parallel_groups, check_initialized=False
            ),
            'ep': partial(parallel_state.get_expert_model_parallel_group, check_initialized=False),
            'expt_tp': partial(
                parallel_state.get_expert_tensor_parallel_group, check_initialized=False
            ),
            'tp_ep': partial(
                parallel_state.get_expert_tensor_and_model_parallel_group, check_initialized=False
            ),
            'tp_ep_pp': partial(
                parallel_state.get_expert_tensor_model_pipeline_parallel_group,
                check_initialized=False,
            ),
            'embd': partial(parallel_state.get_embedding_group, check_initialized=False),
            'pos_embd': partial(
                parallel_state.get_position_embedding_group, check_initialized=False
            ),
            'dp': parallel_state.get_data_parallel_group,
            'dp_cp': partial(parallel_state.get_data_parallel_group, with_context_parallel=True),
            'intra_dp_cp': partial(
                parallel_state.get_data_parallel_group,
                with_context_parallel=True,
                partial_data_parallel=True,
            ),
            'intra_expt_dp': partial(
                parallel_state.get_expert_data_parallel_group,
                check_initialized=False,
                partial_expert_data_parallel=True,
            ),
            'inter_dist_opt': partial(
                parallel_state.get_inter_distributed_optimizer_instance_group,
                check_initialized=False,
            ),
            'intra_dist_opt': partial(
                parallel_state.get_intra_distributed_optimizer_instance_group,
                check_initialized=False,
            ),
            
            'expt_dp': partial(
                parallel_state.get_expert_data_parallel_group, check_initialized=False
            ),
            'tp_dp_cp': partial(
                parallel_state.get_tensor_and_data_parallel_group,
                check_initialized=False,
                with_context_parallel=True,
            ),
        }

        assert all(
            pg in pg_to_func for pg in required_pgs
        ), f"Initialization function for process group not defined for all \
        ProcessGroupCollection fields"

        
        init_dict = {pg: pg_to_func[pg]() for pg in required_pgs}

        return cls(**init_dict)

    @staticmethod
    def setup_process_groups_for_optimizer(
        pg_collection: Optional['ProcessGroupCollection'],
        model_chunks: List,
        use_gloo_process_groups: bool = True,
    ):
        """
        Helper method to set up process groups for optimizer and DDP with proper validation
        and fallbacks.

        Args:
            pg_collection: Optional process group collection. If None, uses parallel_state groups.
            model_chunks: List of model chunks to extract configuration from.
            use_gloo_process_groups: Whether to set up gloo process groups.

        Returns:
            Dictionary containing all required process groups:
                - dp_group: Data parallel group
                - dp_cp_group: Data parallel with context parallel group
                - intra_dp_cp_group: Intra data parallel with context parallel group
                - expt_dp_group: Expert data parallel group
                - intra_expt_dp_group: Intra expert data parallel group
                - mp_group: Model parallel group
                - expt_tp_pp_group: Expert tensor-model-pipeline parallel group
                - inter_dist_opt_group: Inter distributed optimizer group (may be None)
                - intra_dist_opt_group: Intra distributed optimizer group (may be None)
                - intra_dp_cp_group_gloo: Gloo version of intra_dp_cp_group (may be None)
                - intra_expt_dp_group_gloo: Gloo version of intra_expt_dp_group (may be None)
        """
        from megatron.core import parallel_state
        from megatron.core.utils import get_model_config

        if pg_collection is None:
            
            dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            intra_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            expt_dp_group = parallel_state.get_expert_data_parallel_group()
            intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=True
            )
            intra_dist_opt_group = parallel_state.get_intra_distributed_optimizer_instance_group()

            
            if use_gloo_process_groups:
                intra_dp_cp_group_gloo = parallel_state.get_data_parallel_group_gloo(
                    with_context_parallel=True, partial_data_parallel=True
                )
                intra_expt_dp_group_gloo = parallel_state.get_expert_data_parallel_group_gloo(
                    partial_expert_data_parallel=True
                )
            else:
                intra_dp_cp_group_gloo = None
                intra_expt_dp_group_gloo = None

            
            mp_group = parallel_state.get_model_parallel_group()
            expt_tp_pp_group = parallel_state.get_expert_tensor_model_pipeline_parallel_group()

            
            if hasattr(model_chunks[0], 'ddp_config'):
                ddp_config = model_chunks[0].ddp_config
                if ddp_config.num_distributed_optimizer_instances > 1:
                    inter_dist_opt_group = (
                        parallel_state.get_inter_distributed_optimizer_instance_group()
                    )
                else:
                    inter_dist_opt_group = None
            else:
                inter_dist_opt_group = None

        else:
            

            
            if not hasattr(pg_collection, 'dp'):
                raise ValueError("dp process group is required but not provided in pg_collection")
            dp_group = pg_collection.dp

            
            if hasattr(pg_collection, 'dp_cp'):
                dp_cp_group = pg_collection.dp_cp
            else:
                model_config = get_model_config(model_chunks[0])
                cp_size = getattr(model_config, 'context_parallel_size', 1)
                if cp_size == 1:
                    
                    dp_cp_group = dp_group
                else:
                    raise ValueError(
                        "dp_cp process group is required when context_parallel_size > 1 "
                        "but not provided in pg_collection"
                    )

            
            if not hasattr(pg_collection, 'expt_dp'):
                raise ValueError(
                    "expt_dp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            expt_dp_group = pg_collection.expt_dp

            
            if hasattr(model_chunks[0], 'ddp_config'):
                ddp_config = model_chunks[0].ddp_config
                if ddp_config.num_distributed_optimizer_instances == 1:
                    
                    
                    
                    
                    intra_dp_cp_group = dp_cp_group
                    intra_expt_dp_group = expt_dp_group
                    inter_dist_opt_group = None
                else:
                    
                    if not (
                        hasattr(pg_collection, 'intra_dp_cp')
                        and hasattr(pg_collection, 'intra_expt_dp')
                        and hasattr(pg_collection, 'inter_dist_opt')
                        and hasattr(pg_collection, 'intra_dist_opt')
                    ):
                        raise ValueError(
                            "intra_dp_cp, intra_expt_dp, inter_dist_opt, and intra_dist_opt "
                            "process groups are required when using multiple optimizer "
                            "instances (>1) but not provided in pg_collection"
                        )
                    intra_dp_cp_group = pg_collection.intra_dp_cp
                    intra_expt_dp_group = pg_collection.intra_expt_dp
                    inter_dist_opt_group = pg_collection.inter_dist_opt

                if ddp_config.use_distributed_optimizer:
                    if not hasattr(pg_collection, 'intra_dist_opt'):
                        raise ValueError(
                            "intra_dist_opt process group is required but not provided in "
                            "pg_collection. Please explicitly set it to None if you don't need it."
                        )
                    intra_dist_opt_group = pg_collection.intra_dist_opt
                else:
                    intra_dist_opt_group = None
            else:
                
                intra_dp_cp_group = dp_cp_group
                intra_expt_dp_group = expt_dp_group
                inter_dist_opt_group = None
                intra_dist_opt_group = None

            
            if not hasattr(pg_collection, 'mp'):
                raise ValueError(
                    "mp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            mp_group = pg_collection.mp

            
            if not hasattr(pg_collection, 'tp_ep_pp'):
                raise ValueError(
                    "tp_ep_pp process group is required but not provided in pg_collection. "
                    "Please explicitly set it to None if you don't need it."
                )
            expt_tp_pp_group = pg_collection.tp_ep_pp

            
            if use_gloo_process_groups:
                raise ValueError(
                    "Gloo process groups are not supported when pg_collection is "
                    "provided. Please set use_gloo_process_groups to False."
                )
            intra_dp_cp_group_gloo = None
            intra_expt_dp_group_gloo = None

        return {
            'dp_group': dp_group,
            'dp_cp_group': dp_cp_group,
            'intra_dp_cp_group': intra_dp_cp_group,
            'expt_dp_group': expt_dp_group,
            'intra_expt_dp_group': intra_expt_dp_group,
            'mp_group': mp_group,
            'expt_tp_pp_group': expt_tp_pp_group,
            'inter_dist_opt_group': inter_dist_opt_group,
            'intra_dist_opt_group': intra_dist_opt_group,
            'intra_dp_cp_group_gloo': intra_dp_cp_group_gloo,
            'intra_expt_dp_group_gloo': intra_expt_dp_group_gloo,
        }

    @staticmethod
    def setup_process_groups_for_ddp(
        pg_collection: Optional['ProcessGroupCollection'], config, ddp_config
    ):
        """
        Helper method to set up process groups for DDP with proper validation and fallbacks.

        Args:
            pg_collection: Optional process group collection. If None, uses parallel_state groups.
            config: Model config to extract context_parallel_size from.
            ddp_config: DDP config to extract num_distributed_optimizer_instances from.

        Returns:
            Dictionary containing all required process groups for DDP.
        """
        import logging

        import torch

        from megatron.core import parallel_state
        from megatron.core.utils import log_single_rank

        logger = logging.getLogger(__name__)

        if pg_collection is None:
            
            return {
                'dp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=False, partial_data_parallel=False
                ),
                'dp_cp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                ),
                'intra_dp_cp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                ),
                'expt_dp_group': parallel_state.get_expert_data_parallel_group(),
                'intra_expt_dp_group': parallel_state.get_expert_data_parallel_group(
                    partial_expert_data_parallel=True
                ),
                'tp_group': parallel_state.get_tensor_model_parallel_group(),
                'pp_group': parallel_state.get_pipeline_model_parallel_group(),
                'ep_group': parallel_state.get_expert_model_parallel_group(),
                'inter_dist_opt_group': (
                    parallel_state.get_inter_distributed_optimizer_instance_group()
                    if ddp_config.num_distributed_optimizer_instances > 1
                    else None
                ),
                'intra_dist_opt_group': (
                    parallel_state.get_intra_distributed_optimizer_instance_group()
                    if ddp_config.use_distributed_optimizer
                    else None
                ),
            }
        else:
            
            result = {}

            
            if not hasattr(pg_collection, 'dp'):
                raise ValueError("dp process group is required but not provided in pg_collection")
            result['dp_group'] = pg_collection.dp

            
            if hasattr(pg_collection, 'dp_cp'):
                result['dp_cp_group'] = pg_collection.dp_cp
            else:
                cp_size = getattr(config, 'context_parallel_size', 1)
                if cp_size == 1:
                    
                    result['dp_cp_group'] = result['dp_group']
                else:
                    raise ValueError(
                        "dp_cp process group is required when context_parallel_size > 1 "
                        "but not provided in pg_collection"
                    )

            
            if hasattr(pg_collection, 'expt_dp') and pg_collection.expt_dp is not None:
                result['expt_dp_group'] = pg_collection.expt_dp
            else:
                
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "No expert data parallel group provided in pg_collection, "
                    "creating a new one with just the current rank",
                )
                result['expt_dp_group'] = torch.distributed.new_group(
                    ranks=[torch.distributed.get_rank()]
                )

            
            if ddp_config.num_distributed_optimizer_instances == 1:
                result['intra_dp_cp_group'] = result['dp_cp_group']
                result['intra_expt_dp_group'] = result['expt_dp_group']
                result['inter_dist_opt_group'] = None
            else:
                
                if not (
                    hasattr(pg_collection, 'intra_dp_cp')
                    and hasattr(pg_collection, 'intra_expt_dp')
                    and hasattr(pg_collection, 'inter_dist_opt')
                ):
                    raise ValueError(
                        "intra_dp_cp, intra_expt_dp, and inter_dist_opt "
                        "process groups are required when using multiple optimizer "
                        "instances (>1) but not provided in pg_collection"
                    )
                result['intra_dp_cp_group'] = pg_collection.intra_dp_cp
                result['intra_expt_dp_group'] = pg_collection.intra_expt_dp
                result['inter_dist_opt_group'] = pg_collection.inter_dist_opt

            
            if not all(
                [
                    hasattr(pg_collection, 'tp'),
                    hasattr(pg_collection, 'pp'),
                    hasattr(pg_collection, 'ep'),
                ]
            ):
                raise ValueError(
                    "tp, pp and ep process groups are required but not provided in pg_collection"
                )
            result['tp_group'] = pg_collection.tp
            result['pp_group'] = pg_collection.pp
            result['ep_group'] = pg_collection.ep

            return result
