from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_device_name, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()


def get_config_file_name(
    E: int,
    N: int,
    dtype: Optional[str],
    block_shape: Optional[int] = None,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    down_moe_selector = "_down" if down_moe else ""
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    if get_global_server_args().enable_deterministic_inference:
        logger.warning(
            "Deterministic inference is enabled, using default MoE kernel config."
        )
        return None
    
    supported_triton_versions = ["3.4.0", "3.3.1", "3.2.0", "3.1.0"]

    
    
    json_file_name = get_config_file_name(
        E,
        N,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    
    
    config_dir = os.environ.get(
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            
            
            
            
            
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            
            return {int(key): val for key, val in json.load(f).items()}

    
    for try_triton_version in supported_triton_versions:
        if try_triton_version == triton_version:
            continue
        try_config_file_path = os.path.join(
            config_dir,
            "configs",
            f"triton_{try_triton_version.replace('.', '_')}",
            json_file_name,
        )
        if os.path.exists(try_config_file_path):
            with open(try_config_file_path) as f:
                logger.warning(
                    f"Config file not found at {config_file_path}. Fallback to triton version {try_triton_version} and use MoE kernel config from {try_config_file_path}. Performance might be sub-optimal!",
                )
                
                return {int(key): val for key, val in json.load(f).items()}

    
    
    if down_moe:
        logger.warning(
            (
                "Using MoE kernel config with down_moe=False. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    else:
        logger.warning(
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if get_global_server_args().enable_deterministic_inference:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        return config
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 2 if _is_hip else 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2 if _is_hip else 4,
                }
        else:
            
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_hip else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    per_channel_quant: bool = False,
    return_down_config: bool = False,
):
    from sglang.srt.layers.moe.fused_moe_triton import get_config

    down_config = None
    max_block_m = None
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        
        E, _, N = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=False,
        )

        if configs:
            
            
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            
            config = get_default_config(
                M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
            )
        if return_down_config:
            down_configs = get_moe_configs(
                E,
                N,
                dtype,
                block_n,
                block_k,
                per_channel_quant=per_channel_quant,
                down_moe=True,
            )
            if down_configs:
                down_config = down_configs[
                    min(down_configs.keys(), key=lambda x: abs(x - M))
                ]
                down_config = dict(**down_config)
                max_block_m = max(
                    [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
                )
    if return_down_config:
        assert (
            down_config is None or config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
        )
        return config, (down_config, max_block_m)
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        
        
        return "float32"
    return None
