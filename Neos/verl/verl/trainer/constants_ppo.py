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

import json
import os

import torch
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()



_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}





_rocm_ray_env = {}
if torch.version.hip is not None:
    _rocm_ray_env = {"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0"}








_is_hopper_or_ampere = (_major or 0) in (8, 9)


def _uses_megatron(config) -> bool:
    """Return True if any trainable engine in the config uses the Megatron strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "megatron":
            return True
    return False


def _uses_mindspeed(config) -> bool:
    """Return True if any trainable engine in the config uses the mindspeed strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "mindspeed":
            return True
    return False


PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        
        
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        
        
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        **_gb200_nccl_env,
        **_rocm_ray_env,
    },
}


def get_ppo_ray_runtime_env(config=None):
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.

    Args:
        config: Optional training config. When the engine strategy is Megatron
            and the GPU is Hopper/Ampere, CUDA_DEVICE_MAX_CONNECTIONS=1 is set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    
    if _is_hopper_or_ampere and _uses_megatron(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if _uses_mindspeed(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    
    for key in ("PYTHONHASHSEED", "VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "VERL_RL_INSIGHT_ENABLE"):
        runtime_env["env_vars"][key] = os.environ.get(key, "0")
    if os.getenv("NEOS_COEV_ROOT"):
        for key, value in os.environ.items():
            if key.startswith(("NEOS_COEV_", "NEOS_MEI_TRACE_", "NEOS_LOCAL_MEI_")):
                runtime_env["env_vars"][key] = value
    for key in ("NEOS_TURN_CREDIT_CAUSAL_COMPONENTS",
                "NEOS_FORCED_FINAL_CORRECT_REWARD"):
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]
    for key in ("VERL_ROLLING_ADMISSION_CONTROL_TIMEOUT_S", "VERL_ROLLING_ADMISSION_RPC_CONCURRENCY"):
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]
    for key in ("VERL_SGLANG_SCOPE_LOCAL_TP_SERVER_CUDA", "VERL_SGLANG_SCOPE_TP1_SERVER_CUDA",
                "VERL_QWEN35_FIX_REPLICATED_KV_OUTPUT_GATE"):
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]
    for key in ("NEOS_TURN_CREDIT_REQUEST_TIMEOUT", "NEOS_TURN_CREDIT_TOTAL_TIMEOUT",
                "NEOS_TURN_CREDIT_BATCH_WAIT_TIMEOUT"):
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
