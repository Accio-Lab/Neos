# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
import logging
import os
from functools import partial
from typing import Any, Callable, ContextManager, Iterator, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.package_info import __version__
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    build_r3_replay_mask,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_entropy_with_chunking,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)
from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.utils.seqlen_balancing import restore_dynamic_batch
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _uniform_temperature_value(temperature) -> float | None:
    """Return a scalar when every sample uses the same temperature.

    GRPO represents even a uniform temperature as a per-sample tensor. Treat
    that as one scalar so the fused LM-head path does not silently fall back
    to materializing the complete vocabulary logits tensor.
    """
    if not isinstance(temperature, torch.Tensor):
        return float(temperature)
    flattened = temperature.detach().reshape(-1)
    if flattened.numel() == 0:
        return None
    first = flattened[0]
    if flattened.numel() > 1 and not torch.all(flattened == first).item():
        return None
    return float(first.item())


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.vanilla_mbridge:
                raise ValueError(
                    "QAT requires non-vanilla Megatron bridge. "
                    "Please set 'use_mbridge=True' and 'vanilla_mbridge=False'."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")

        
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []
        
        from verl.utils.device import is_cuda_available, is_npu_available

        if is_npu_available and __version__ >= "0.16.0":
            from verl.models.mcore.patch import apply_mtp_inference_patch

            apply_mtp_inference_patch()

        if is_cuda_available:
            from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

            apply_patch_megatron_recomputation_backward()

        if os.getenv("VERL_QWEN35_FIX_REPLICATED_KV_OUTPUT_GATE", "0") == "1":
            from verl.models.mcore.qwen35_gate_patch import patch_qwen35_attention_output_gate

            patch_qwen35_attention_output_gate()

    def _init_device_mesh(self):
        
        if mpu.is_initialized():
            return

        extra_args = dict()

        if self.engine_config.dynamic_context_parallel:
            assert "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters, (
                "dynamic_context_parallel is not supported in your megatron version, "
                + "please update your megatron version to the latest version"
            )
            assert self.engine_config.max_seqlen_per_dp_cp_rank is not None, (
                "max_seqlen_per_dp_cp_rank is required when dynamic_context_parallel is enabled"
            )
            extra_args["dynamic_context_parallel"] = self.engine_config.dynamic_context_parallel

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
            **extra_args,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        self.is_value_model = self.model_config.model_type == "value_model"
        self.share_embeddings_and_output_weights = self.model_config.share_embeddings_and_output_weights

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        if self.is_value_model:
            
            
            self.model_config.hf_config.tie_word_embeddings = False
            self.share_embeddings_and_output_weights = False
            override_transformer_config["share_embeddings_and_output_weights"] = False
        if self.engine_config.dynamic_context_parallel:
            override_transformer_config["max_seqlen_per_dp_cp_rank"] = self.engine_config.max_seqlen_per_dp_cp_rank
            
            
            
            
            override_transformer_config["dynamic_context_parallel"] = False
            override_transformer_config["context_parallel_size"] = mpu.get_data_parallel_world_size()
        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            
            provider = bridge.to_megatron_provider(load_weights=False)

            
            from megatron.core.transformer.enums import AttnBackend

            virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider_overrides = {
                "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
                "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
                "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
                "context_parallel_size": self.engine_config.context_parallel_size,
                "sequence_parallel": self.engine_config.sequence_parallel,
                "overlap_p2p_comm": (
                    virtual_pipeline_model_parallel_size is not None and virtual_pipeline_model_parallel_size > 1
                ),
                "batch_p2p_comm": False,
                "variable_seq_lengths": True,
                "attention_backend": AttnBackend.flash,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_load_balancing_type": "none",
            }
            for key, value in override_transformer_config.items():
                provider_overrides[key] = value
            if (
                self.model_config.hf_config.model_type == "deepseek_v4"
                and not self.model_config.mtp.enable
                and getattr(provider, "mtp_num_layers", 0)
            ):
                provider_overrides["mtp_num_layers"] = 0
                csa_compress_ratios = getattr(provider, "csa_compress_ratios", None)
                if csa_compress_ratios is not None:
                    provider_overrides["csa_compress_ratios"] = csa_compress_ratios[: provider.num_layers]
            if self.enable_routing_replay:
                if hasattr(provider, "moe_enable_routing_replay"):
                    provider_overrides["moe_enable_routing_replay"] = True
                else:
                    provider_overrides["enable_routing_replay"] = True

            if self._qat_enabled:
                from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

                provider.transformer_layer_spec = modelopt_transformer_layer_spec

            provider.apply_overrides_and_finalize(
                dtype=self.param_dtype,
                overrides=provider_overrides,
            )
            self.provider = provider
            tf_config = None  
        self.bridge = bridge

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        
        
        
        if self.enable_routing_replay and tf_config is not None:
            if hasattr(tf_config, "moe_enable_routing_replay"):
                tf_config.moe_enable_routing_replay = True
            else:
                tf_config.enable_routing_replay = True

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _resolve_override_ddp_config(self):
        """Keep the DDP grad-bucket dtype consistent with the optimizer's grad buffer.

        When the precision-aware optimizer is opted into with a sub-fp32
        ``main_grads_dtype``, the DDP grad bucket must reduce grads in the same
        dtype, so inject ``grad_reduce_in_fp32=False`` unless the user set it
        explicitly via ``override_ddp_config``. Default (opt-out) leaves the fp32
        grad bucket untouched, preserving prior behavior.
        """
        from verl.utils.torch_dtypes import PrecisionType

        override_ddp_config = dict(self.engine_config.override_ddp_config or {})
        opt_cfg = self.optimizer_config
        if (
            opt_cfg is not None
            and getattr(opt_cfg, "use_precision_aware_optimizer", False)
            and PrecisionType.to_dtype(getattr(opt_cfg, "main_grads_dtype", "fp32")) != torch.float32
            and "grad_reduce_in_fp32" not in override_ddp_config
        ):
            override_ddp_config["grad_reduce_in_fp32"] = False
        return override_ddp_config

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
            freeze_vision_tower=self.engine_config.freeze_vision_tower,
        )
        override_ddp_config = self._resolve_override_ddp_config()

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.engine_config.freeze_vision_tower:
            from verl.utils.vision_freeze import assert_vision_tower_frozen

            print({"vision_freeze_stage": "after_weight_load_before_optimizer",
                   **assert_vision_tower_frozen(module)}, flush=True)

        from verl.utils.moe_router_freeze import expert_router_only_requested, assert_expert_routers_frozen
        if expert_router_only_requested(self.engine_config):
            print({"router_freeze_stage": "after_weight_load_before_optimizer",
                   "rank": torch.distributed.get_rank(),
                   **assert_expert_routers_frozen(module)}, flush=True)

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels are not supported for value models or when MTP is enabled in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
            bf16=self.param_dtype == torch.bfloat16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

        self._maybe_enable_fused_kernels()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)
        elif (
            self.engine_config.forward_only
            and self.engine_config.override_transformer_config.get("mtp_num_layers") == 0
        ):
            from verl.models.mcore.mtp_patch import patch_postprocess

            for model in self.module:
                patch_postprocess(model)

        
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

        
        
        
        if os.getenv("VERL_MEGATRON_PREWARM_GRAD_STATS", "0") == "1":
            from verl.utils.megatron.optimizer_memory import warmup_optimizer_grad_stats

            warmup_optimizer_grad_stats(self.optimizer)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        
        for chunk in self.module:
            
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        
        
        if os.getenv("VERL_MEGATRON_PRE_OPTIMIZER_EMPTY_CACHE", "0") == "1":
            from verl.utils.megatron.optimizer_memory import release_optimizer_step_cache

            release_optimizer_step_cache()
        elif getattr(self, "_distillation_use_topk_active", False):
            get_torch_device().empty_cache()
        from verl.utils.moe_router_freeze import (
            expert_router_only_requested, assert_expert_routers_frozen, router_parameters,
        )
        check_routers = expert_router_only_requested(self.engine_config) and not getattr(
            self, '_expert_router_first_update_verified', False)
        if check_routers:
            assert_expert_routers_frozen(self.module)
            router_sentinels = [(p, p.detach().reshape(-1)[:16].clone())
                                for _, p in router_parameters(self.module)]
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        if check_routers:
            if not all(torch.equal(p.detach().reshape(-1)[:16], before) for p, before in router_sentinels):
                raise RuntimeError('Frozen expert-router values changed at the optimizer boundary')
            print({"router_freeze_stage": "after_first_optimizer_step",
                   "rank": torch.distributed.get_rank(), "sampled_values_unchanged": True,
                   **assert_expert_routers_frozen(self.module)}, flush=True)
            self._expert_router_first_update_verified = True

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        if self.engine_config.dynamic_context_parallel:
            
            return 0
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self.engine_config.dynamic_context_parallel:
            
            return 1
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return mpu.get_context_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def _routed_num_tokens(self, data: TensorDict) -> torch.Tensor:
        """Real (unpadded) tokens fed to the MoE router: attention_mask in the padded RL
        path, else the packed input_ids count in the no-padding SFT path. Not loss_mask,
        which counts response tokens only and would under-normalize the router loss."""
        attention_mask = data.get("attention_mask", None)
        if attention_mask is not None:
            return attention_mask.sum()
        input_ids = data["input_ids"]
        if input_ids.is_nested:
            return input_ids.offsets()[-1]
        return torch.tensor(input_ids.numel(), device=input_ids.device)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        self._distillation_use_topk_active = tu.get_non_tensor_data(data, key="distillation_use_topk", default=False)
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        
        
        
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss:
            routed_num_tokens = self._routed_num_tokens(data).to(get_device_id())
            torch.distributed.all_reduce(
                routed_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            tu.assign_non_tensor(data, routed_num_tokens=routed_num_tokens.item())

        
        
        
        pad_bshd_to_minibatch_max = self.engine_config.pad_bshd_to_minibatch_max
        global_max_seqlen = None
        if pad_bshd_to_minibatch_max and not self.engine_config.use_remove_padding and "input_ids" in data.keys():
            input_ids_for_max = data["input_ids"]
            if input_ids_for_max.is_nested:
                global_max_seqlen = int(input_ids_for_max.offsets().diff().max().item())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=True,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)
            if global_max_seqlen is not None:
                tu.assign_non_tensor(micro_batch, forced_max_seqlen=global_max_seqlen)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(
            self.forward_step,
            logits_processor_func=loss_function,
            postprocess_micro_batch_func=postprocess_micro_batch_func,
        )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            if forward_only and self.engine_config.router_replay.mode == "R2":
                
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        
        
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  
            micro_batch_size=1,  
            forward_only=forward_only,
        )

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            
            
            
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs():
                if "metrics" not in losses_reduced[0]:
                    losses_reduced[0]["metrics"] = {}
                losses_reduced[0]["metrics"].update(metrics)

        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                
                vp_size = len(self.module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                topk_idx_td = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                tensors = [tensor for nt in self.mini_layer_topk_idx_list for tensor in nt.unbind()]
                topk_idx_td = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            self.mini_layer_topk_idx_list = []

            layers_topk_idx = pp_gather(topk_idx_td.to(torch.uint8), self.tf_config)
            use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
            if use_dynamic_bsz and indices is not None:
                layers_topk_idx = restore_dynamic_batch(layers_topk_idx, indices)

        output = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            if RouterReplayHelper.is_r2_record_action(self.tf_config):
                output["model_output"]["routed_experts"] = layers_topk_idx
        if enable_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
        return output

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        if non_merge_lora_sync:
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
        
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
        else:
            per_tensor_param = (
                self.bridge.export_hf_weights(self.module, merge_adapter_weights=False)
                if non_merge_lora_sync
                else self.bridge.export_hf_weights(self.module)
            )

        
        if self._qat_enabled:
            from verl.utils.modelopt import export_qat_weights

            per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)

        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", None)

        return {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask", None),
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        return output

    def _lm_head_logits_processor(
        self,
        logits,
        label,
        temperature,
        *,
        calculate_sum_pi_squared: bool,
        calculate_entropy: bool,
        distillation_use_topk: bool,
        distillation_only: bool,
        logits_processor_func: Callable,
        batch: TensorDict,
        data_format: str,
    ):
        assert logits.shape[:2] == label.shape[:2]
        
        temperature[temperature <= 0] = 1e-8
        assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
        logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
        ret = {}
        
        if calculate_sum_pi_squared:
            ret["sum_pi_squared"] = vocab_parallel_sum_pi_squared(logits)
        if calculate_entropy:
            logits_bak = logits.clone()
            
            
            
            
            
            
            
            if self.engine_config.entropy_from_logits_with_chunking:
                entropy = vocab_parallel_entropy_with_chunking(
                    logits,
                    chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                )
            else:
                entropy = vocab_parallel_entropy(logits)

            ret["entropy"] = entropy
        else:
            logits_bak = logits

        
        if distillation_use_topk:
            ret.update(logits_processor_func(student_logits=logits_bak, data=batch, data_format=data_format))
        if not distillation_only:
            ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)

        return ret

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)

        if self.engine_config.dynamic_context_parallel:
            
            from verl.utils.megatron_utils import dynamic_cp_split_batch

            batch = dynamic_cp_split_batch(
                batch=batch,
                engine_config=self.engine_config,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
            )

        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)
        distillation_only = tu.get_non_tensor_data(batch, key="distillation_only", default=False)

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        loss_mask = model_inputs["loss_mask"]

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            replay_mask = None
            if self.engine_config.router_replay.mode == "R3":
                replay_mask = build_r3_replay_mask(input_ids, batch["response_mask"])
            set_router_replay_data(
                layers_topk_idx,
                None,
                self.tf_config,
                vp_rank,
                replay_mask=replay_mask,
            )

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        if use_fused_kernels:
            temperature_value = _uniform_temperature_value(temperature)
            if temperature_value is None:
                logger.warning_once(
                    "Fused kernels require one uniform temperature across "
                    "the micro-batch. Falling back to non-fused."
                )
                use_fused_kernels = False

        if use_fused_kernels:
            if self.engine_config.use_remove_padding:
                from verl.models.mcore import get_mcore_forward_fused_model_engine_fn

                logger.warning_once("Megatron fused loss path=THD-packed")
                fused_forward_fn = get_mcore_forward_fused_model_engine_fn(
                    self.model_config.hf_config
                )
                output = fused_forward_fn(
                    model=model,
                    input_ids=input_ids,
                    labels=label,
                    multi_modal_inputs=multi_modal_inputs,
                    temperature=temperature_value,
                    calculate_entropy=calculate_entropy,
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                )
            else:
                from verl.models.mcore import (
                    get_mcore_forward_fused_bshd_model_engine_fn,
                )

                logger.warning_once("Megatron fused loss path=BSHD-nonpacked")
                fused_forward_fn = get_mcore_forward_fused_bshd_model_engine_fn(
                    self.model_config.hf_config
                )
                output = fused_forward_fn(
                    model=model,
                    input_ids=input_ids,
                    labels=label,
                    multi_modal_inputs=multi_modal_inputs,
                    temperature=temperature_value,
                    calculate_entropy=calculate_entropy,
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                    forced_max_seqlen=tu.get_non_tensor_data(
                        data=batch,
                        key="forced_max_seqlen",
                        default=None,
                    ),
                )
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            data_format = "thd" if self.engine_config.use_remove_padding else "bshd"

            logits_processor = partial(
                self._lm_head_logits_processor,
                calculate_sum_pi_squared=calculate_sum_pi_squared,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
                distillation_only=distillation_only,
                logits_processor_func=logits_processor_func,
                batch=batch,
                data_format=data_format,
            )

            response_attention_mask = None
            if attention_mask is not None and not loss_mask.is_nested:
                response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
            logits_processor_args = {
                "label": label,
                "temperature": temperature,
                "loss_mask": loss_mask,
                "response_attention_mask": response_attention_mask,
            }

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format=data_format,
                mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                local_cp_size=local_cp_size,
                forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
            )

        
        if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
            merge_router_topk_indices(None, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank)

        
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=local_cp_size)

    def postprocess_micro_batch_func(
        self, output, data: TensorDict, forward_only: bool, loss_function, local_cp_size=None
    ):
        
        
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            
            
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            
            
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}
        if local_cp_size is not None:
            
            from verl.utils.megatron_utils import dynamic_cp_merge_output

            model_output = dynamic_cp_merge_output(
                model_output,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
                local_cp_size=local_cp_size,
            )

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        
        
        
        
        
        
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss and loss_function is not None:
            
            
            
            if hasattr(loss_function, "keywords") and "config" in loss_function.keywords:
                _agg_mode = getattr(loss_function.keywords["config"], "loss_agg_mode", None)
                if _agg_mode == "seq-mean-token-mean":
                    raise ValueError(
                        "loss_agg_mode='seq-mean-token-mean' is incompatible with "
                        "calculate_per_token_loss=True (auto-enabled by Megatron-Bridge "
                        "under CP>1). The per-sequence inner division by n_s requires "
                        "local-shard counts that diverge from global under CP. Use one "
                        "of: 'token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm'."
                    )
            
            
            
            
            if not self.engine_config.use_remove_padding:
                raise ValueError(
                    "calculate_per_token_loss=True requires use_remove_padding=True. "
                    "verl does not pass a padding_mask to the MoE router, so in BSHD it "
                    "normalizes the aux/z loss by the padding-inclusive token count (B*S) "
                    "while gradients are divided by the real token count. Use THD "
                    "(use_remove_padding=True) or disable CP."
                )
            
            
            
            
            cp_size = self.engine_config.context_parallel_size
            local_num_tokens = (self._routed_num_tokens(data) // cp_size).to(torch.int)
            
            
            
            routed_num_tokens = data["routed_num_tokens"] if "routed_num_tokens" in data.keys() else None
            if routed_num_tokens is None:
                routed_num_tokens = self._routed_num_tokens(data)
            dp_size = data["dp_size"] if "dp_size" in data.keys() else 1
            local_sum = loss * routed_num_tokens / dp_size
            return local_sum, local_num_tokens, output

        
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    
    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_engine_forward_fn

        forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
            forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
