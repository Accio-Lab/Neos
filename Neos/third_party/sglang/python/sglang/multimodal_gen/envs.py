# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL: int = 60
    SGLANG_DIFFUSION_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    SGLANG_DIFFUSION_CACHE_ROOT: str = os.path.expanduser("~/.cache/sgl_diffusion")
    SGLANG_DIFFUSION_CONFIG_ROOT: str = os.path.expanduser("~/.config/sgl_diffusion")
    SGLANG_DIFFUSION_CONFIGURE_LOGGING: int = 1
    SGLANG_DIFFUSION_LOGGING_LEVEL: str = "INFO"
    SGLANG_DIFFUSION_LOGGING_PREFIX: str = ""
    SGLANG_DIFFUSION_LOGGING_CONFIG_PATH: str | None = None
    SGLANG_DIFFUSION_TRACE_FUNCTION: int = 0
    SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD: str = "fork"
    SGLANG_DIFFUSION_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    SGLANG_DIFFUSION_SERVER_DEV_MODE: bool = False
    SGLANG_DIFFUSION_STAGE_LOGGING: bool = False
    
    SGLANG_CACHE_DIT_ENABLED: bool = False
    SGLANG_CACHE_DIT_FN: int = 1
    SGLANG_CACHE_DIT_BN: int = 0
    SGLANG_CACHE_DIT_WARMUP: int = 4
    SGLANG_CACHE_DIT_RDT: float = 0.24
    SGLANG_CACHE_DIT_MC: int = 3
    SGLANG_CACHE_DIT_TAYLORSEER: bool = False
    SGLANG_CACHE_DIT_TS_ORDER: int = 1
    SGLANG_CACHE_DIT_SCM_PRESET: str = "none"
    SGLANG_CACHE_DIT_SCM_COMPUTE_BINS: str | None = None
    SGLANG_CACHE_DIT_SCM_CACHE_BINS: str | None = None
    SGLANG_CACHE_DIT_SCM_POLICY: str = "dynamic"
    
    SGLANG_CACHE_DIT_SECONDARY_FN: int = 1
    SGLANG_CACHE_DIT_SECONDARY_BN: int = 0
    SGLANG_CACHE_DIT_SECONDARY_WARMUP: int = 4
    SGLANG_CACHE_DIT_SECONDARY_RDT: float = 0.24
    SGLANG_CACHE_DIT_SECONDARY_MC: int = 3
    SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER: bool = False
    SGLANG_CACHE_DIT_SECONDARY_TS_ORDER: int = 1
    
    SGLANG_USE_RUNAI_MODEL_STREAMER: bool = True
    SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D: bool = False


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    return int(value) if value is not None else None



def _lazy_str(key: str, default: str | None = None) -> Callable[[], str | None]:
    return lambda: os.getenv(key, default)


def _lazy_int(key: str, default: str | int | None = None) -> Callable[[], int | None]:
    def _getter():
        val = os.getenv(key)
        if val is None:
            return int(default) if default is not None else None
        return int(val)

    return _getter


def _lazy_float(key: str, default: str | float) -> Callable[[], float]:
    return lambda: float(os.getenv(key, str(default)))


def _lazy_bool(key: str, default: str = "false") -> Callable[[], bool]:
    return lambda: get_bool_env_var(key, default)


def _lazy_bool_any(keys: list[str], default: str = "false") -> Callable[[], bool]:
    def _getter():
        for key in keys:
            if get_bool_env_var(key, "false"):
                return True
        return (
            get_bool_env_var("", default)
            if not keys
            else get_bool_env_var(keys[0], default)
        )

    return _getter


def _lazy_path(
    key: str, default_func: Callable[[], str] | None = None
) -> Callable[[], str | None]:
    def _getter():
        val = os.getenv(key)
        if val is None:
            if default_func is None:
                return None
            val = default_func()
        return os.path.expanduser(val)

    return _getter







environment_variables: dict[str, Callable[[], Any]] = {
    
    
    
    "SGLANG_DIFFUSION_TARGET_DEVICE": _lazy_str(
        "SGLANG_DIFFUSION_TARGET_DEVICE", "cuda"
    ),
    
    
    "MAX_JOBS": _lazy_str("MAX_JOBS"),
    
    
    
    "NVCC_THREADS": _lazy_str("NVCC_THREADS"),
    
    "SGLANG_DIFFUSION_USE_PRECOMPILED": _lazy_bool_any(
        [
            "SGLANG_DIFFUSION_USE_PRECOMPILED",
            "SGLANG_DIFFUSION_PRECOMPILED_WHEEL_LOCATION",
        ]
    ),
    
    
    
    "CMAKE_BUILD_TYPE": _lazy_str("CMAKE_BUILD_TYPE"),
    
    "VERBOSE": _lazy_bool("VERBOSE"),
    
    
    
    
    
    "SGLANG_DIFFUSION_CONFIG_ROOT": _lazy_path(
        "SGLANG_DIFFUSION_CONFIG_ROOT",
        lambda: os.path.join(get_default_config_root(), "sgl_diffusion"),
    ),
    
    
    
    "SGLANG_DIFFUSION_CACHE_ROOT": _lazy_path(
        "SGLANG_DIFFUSION_CACHE_ROOT",
        lambda: os.path.join(get_default_cache_root(), "sgl_diffusion"),
    ),
    
    "SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL": _lazy_int(
        "SGLANG_DIFFUSION_RINGBUFFER_WARNING_INTERVAL", 60
    ),
    
    
    "SGLANG_DIFFUSION_NCCL_SO_PATH": _lazy_str("SGLANG_DIFFUSION_NCCL_SO_PATH"),
    
    
    "LD_LIBRARY_PATH": _lazy_str("LD_LIBRARY_PATH"),
    
    "SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE": _lazy_bool(
        "SGLANG_DIFFUSION_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1"
    ),
    
    
    "LOCAL_RANK": _lazy_int("LOCAL_RANK", 0),
    
    "CUDA_VISIBLE_DEVICES": _lazy_str("CUDA_VISIBLE_DEVICES"),
    
    "SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S": _lazy_int(
        "SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S", 60
    ),
    
    
    
    
    "SGLANG_DIFFUSION_CONFIGURE_LOGGING": _lazy_int(
        "SGLANG_DIFFUSION_CONFIGURE_LOGGING", 1
    ),
    "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH": _lazy_str(
        "SGLANG_DIFFUSION_LOGGING_CONFIG_PATH"
    ),
    
    "SGLANG_DIFFUSION_LOGGING_LEVEL": _lazy_str(
        "SGLANG_DIFFUSION_LOGGING_LEVEL", "INFO"
    ),
    
    "SGLANG_DIFFUSION_LOGGING_PREFIX": _lazy_str("SGLANG_DIFFUSION_LOGGING_PREFIX", ""),
    
    
    
    "SGLANG_DIFFUSION_TRACE_FUNCTION": _lazy_int("SGLANG_DIFFUSION_TRACE_FUNCTION", 0),
    
    
    "SGLANG_DIFFUSION_ATTENTION_CONFIG": _lazy_path(
        "SGLANG_DIFFUSION_ATTENTION_CONFIG"
    ),
    
    "SGLANG_DIFFUSION_ATTENTION_BACKEND": _lazy_str(
        "SGLANG_DIFFUSION_ATTENTION_BACKEND"
    ),
    
    
    "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD": _lazy_str(
        "SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD", "fork"
    ),
    
    
    "SGLANG_DIFFUSION_TORCH_PROFILER_DIR": _lazy_path(
        "SGLANG_DIFFUSION_TORCH_PROFILER_DIR"
    ),
    
    
    
    "SGLANG_DIFFUSION_SERVER_DEV_MODE": _lazy_bool("SGLANG_DIFFUSION_SERVER_DEV_MODE"),
    
    
    "SGLANG_DIFFUSION_STAGE_LOGGING": _lazy_bool("SGLANG_DIFFUSION_STAGE_LOGGING"),
    "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": _lazy_bool(
        "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", "false"
    ),
    
    
    "SGLANG_CACHE_DIT_ENABLED": _lazy_bool("SGLANG_CACHE_DIT_ENABLED"),
    
    "SGLANG_CACHE_DIT_FN": _lazy_int("SGLANG_CACHE_DIT_FN", 1),
    
    "SGLANG_CACHE_DIT_BN": _lazy_int("SGLANG_CACHE_DIT_BN", 0),
    
    "SGLANG_CACHE_DIT_WARMUP": _lazy_int("SGLANG_CACHE_DIT_WARMUP", 4),
    
    "SGLANG_CACHE_DIT_RDT": _lazy_float("SGLANG_CACHE_DIT_RDT", 0.24),
    
    "SGLANG_CACHE_DIT_MC": _lazy_int("SGLANG_CACHE_DIT_MC", 3),
    
    "SGLANG_CACHE_DIT_TAYLORSEER": _lazy_bool("SGLANG_CACHE_DIT_TAYLORSEER", "false"),
    
    "SGLANG_CACHE_DIT_TS_ORDER": _lazy_int("SGLANG_CACHE_DIT_TS_ORDER", 1),
    
    "SGLANG_CACHE_DIT_SCM_PRESET": _lazy_str("SGLANG_CACHE_DIT_SCM_PRESET", "none"),
    
    "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS": _lazy_str("SGLANG_CACHE_DIT_SCM_COMPUTE_BINS"),
    
    "SGLANG_CACHE_DIT_SCM_CACHE_BINS": _lazy_str("SGLANG_CACHE_DIT_SCM_CACHE_BINS"),
    
    "SGLANG_CACHE_DIT_SCM_POLICY": _lazy_str("SGLANG_CACHE_DIT_SCM_POLICY", "dynamic"),
    
    "SGLANG_USE_RUNAI_MODEL_STREAMER": _lazy_bool(
        "SGLANG_USE_RUNAI_MODEL_STREAMER", "true"
    ),
}


_CACHE_DIT_SECONDARY_CONFIGS = [
    ("FN", int, "1"),
    ("BN", int, "0"),
    ("WARMUP", int, "4"),
    ("RDT", float, "0.24"),
    ("MC", int, "3"),
    ("TS_ORDER", int, "1"),
]


def _create_secondary_getter(suffix, type_func, default_val):
    primary_key = f"SGLANG_CACHE_DIT_{suffix}"
    secondary_key = f"SGLANG_CACHE_DIT_SECONDARY_{suffix}"

    def _getter():
        val = os.getenv(secondary_key)
        if val is not None:
            return type_func(val)
        return type_func(os.getenv(primary_key, str(default_val)))

    return secondary_key, _getter


for suffix, type_func, default_val in _CACHE_DIT_SECONDARY_CONFIGS:
    key, getter = _create_secondary_getter(suffix, type_func, default_val)
    environment_variables[key] = getter



def _secondary_taylorseer_getter():
    return get_bool_env_var(
        "SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER",
        default=os.getenv("SGLANG_CACHE_DIT_TAYLORSEER", "false"),
    )


environment_variables["SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER"] = (
    _secondary_taylorseer_getter
)



def __getattr__(name: str):
    
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
