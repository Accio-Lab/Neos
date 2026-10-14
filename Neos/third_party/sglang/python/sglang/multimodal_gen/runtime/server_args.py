# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of sglang-diffusion Inference."""

import argparse
import dataclasses
import inspect
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import field
from enum import Enum
from typing import Any, Optional

import addict
import yaml

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.quantization import NunchakuSVDQuantArgs
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.common import (
    is_port_available,
    is_valid_ipv6_address,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.utils import (
    FlexibleArgumentParser,
    StoreBoolean,
    expand_path_fields,
)

logger = init_logger(__name__)


def _is_torch_tensor(obj: Any) -> tuple[bool, Any]:
    """Return (is_tensor, torch_module_or_None) without importing torch at module import time."""
    try:
        import torch  # type: ignore

        return isinstance(obj, torch.Tensor), torch
    except Exception:
        return False, None


def _sanitize_for_logging(obj: Any, key_hint: str | None = None) -> Any:
    """Recursively convert objects to JSON-serializable forms for concise logging.

    Rules:
    - Drop any field/dict key named 'param_names_mapping'.
    - Render Enums using their value.
    - Render torch.Tensor as a compact summary; if key name is 'scaling_factor', include stats.
    - Dataclasses are expanded to dicts and sanitized recursively.
    - Callables/functions are rendered as their qualified name.
    - Redact sensitive fields like 'prompt' and 'negative_prompt' (only show length).
    - Fallback to str(...) for unknown types.
    """
    
    if obj is None or isinstance(obj, (str, int, float, bool)):
        
        if key_hint in ("prompt", "negative_prompt"):
            if isinstance(obj, str):
                return f"<redacted, len={len(obj)}>"
        return obj

    
    if isinstance(obj, Enum):
        return obj.value

    
    is_tensor, torch_mod = _is_torch_tensor(obj)
    if is_tensor:
        try:
            ten = obj.detach().cpu()
            if key_hint == "scaling_factor":
                
                stats = {
                    "shape": list(ten.shape),
                    "dtype": str(ten.dtype),
                }
                
                try:
                    stats["min"] = float(ten.min().item())
                except Exception:
                    pass
                try:
                    stats["max"] = float(ten.max().item())
                except Exception:
                    pass
                try:
                    stats["mean"] = float(ten.float().mean().item())
                except Exception:
                    pass
                return {"tensor": "scaling_factor", **stats}
            
            return {"tensor": True, "shape": list(ten.shape), "dtype": str(ten.dtype)}
        except Exception:
            return "<tensor>"

    
    if dataclasses.is_dataclass(obj):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if not f.repr:
                continue
            name = f.name
            if "names_mapping" in name:  
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            result[name] = _sanitize_for_logging(value, key_hint=name)
        return result

    
    if isinstance(obj, dict):
        result_dict: dict[str, Any] = {}
        for k, v in obj.items():
            try:
                key_str = str(k)
            except Exception:
                key_str = "<key>"
            if key_str == "param_names_mapping":
                continue
            result_dict[key_str] = _sanitize_for_logging(v, key_hint=key_str)
        return result_dict

    
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_logging(x, key_hint=key_hint) for x in obj]

    
    try:
        if inspect.isroutine(obj) or inspect.isclass(obj):
            module = getattr(obj, "__module__", "")
            qn = getattr(obj, "__qualname__", getattr(obj, "__name__", "<callable>"))
            return f"{module}.{qn}" if module else qn
    except Exception:
        pass

    
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


class Backend(str, Enum):
    """
    Enumeration for different model backends.
    - AUTO: Automatically select backend (prefer sglang native, fallback to diffusers)
    - SGLANG: Use sglang's native optimized implementation
    - DIFFUSERS: Use vanilla diffusers pipeline (supports all diffusers models)
    """

    AUTO = "auto"
    SGLANG = "sglang"
    DIFFUSERS = "diffusers"

    @classmethod
    def from_string(cls, value: str) -> "Backend":
        """Convert string to Backend enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid backend: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [backend.value for backend in cls]


@dataclasses.dataclass
class ServerArgs:
    
    model_path: str

    
    model_id: str | None = None

    
    backend: Backend = Backend.AUTO

    
    attention_backend: str = None
    attention_backend_config: addict.Dict | None = None
    cache_dit_config: str | dict[str, Any] | None = (
        None  
    )

    
    nccl_port: Optional[int] = None

    
    trust_remote_code: bool = False
    revision: str | None = None

    
    num_gpus: int = 1
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None
    
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    
    
    dp_size: int = 1
    
    dp_degree: int = 1
    
    enable_cfg_parallel: bool = False

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: Optional[int] = None
    dist_timeout: int | None = 3600  

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    
    pipeline_class_name: str | None = (
        None  
    )

    
    
    lora_path: str | None = None
    lora_nickname: str = "default"  
    lora_scale: float = 1.0  

    
    component_paths: dict[str, str] = field(default_factory=dict)

    
    transformer_weights_path: str | None = None
    
    
    lora_target_modules: list[str] | None = None

    
    dit_cpu_offload: bool | None = None
    dit_layerwise_offload: bool | None = None
    dit_offload_prefetch_size: float = 0.0
    text_encoder_cpu_offload: bool | None = None
    image_encoder_cpu_offload: bool | None = None
    vae_cpu_offload: bool | None = None
    use_fsdp_inference: bool = False
    pin_cpu_memory: bool = True

    
    comfyui_mode: bool = False

    
    enable_torch_compile: bool = False

    
    warmup: bool = False
    warmup_resolutions: list[str] = None
    warmup_steps: int = 1

    disable_autocast: bool | None = None

    
    nunchaku_config: NunchakuSVDQuantArgs | NunchakuConfig | None = field(
        default_factory=NunchakuSVDQuantArgs, repr=False
    )

    
    
    master_port: int | None = None

    
    host: str | None = "127.0.0.1"
    port: int | None = 30000

    
    webui: bool = False
    webui_port: int | None = 12312

    scheduler_port: int = 5555

    output_path: str | None = "outputs/"
    input_save_path: str | None = "inputs/uploads"

    
    prompt_file_path: str | None = None

    
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
            "video_vae": True,
            "audio_vae": True,
            "video_dit": True,
            "audio_dit": True,
            "dual_tower_bridge": True,
        }
    )

    
    

    
    boundary_ratio: float | None = None

    
    log_level: str = "info"

    @property
    def broker_port(self) -> int:
        return self.port + 1

    @property
    def is_local_mode(self) -> bool:
        """
        If no server is running when a generation task begins, 'local_mode' will be enabled: a dedicated server will be launched
        """
        return self.host is None or self.port is None

    def _adjust_path(self):
        expand_path_fields(self)
        self._adjust_save_paths()

    def _adjust_parameters(self):
        """set defaults and normalize values."""
        self._adjust_offload()
        self._adjust_path()
        self._adjust_quant_config()
        self._adjust_warmup()
        self._adjust_network_ports()
        
        self._adjust_parallelism()
        self._adjust_attention_backend()
        self._adjust_platform_specific()
        self._adjust_autocast()

    def _validate_parameters(self):
        """check consistency and raise errors for invalid configs"""
        self._validate_pipeline()
        self._validate_offload()
        self._validate_parallelism()
        self._validate_cfg_parallel()

    def _adjust_save_paths(self):
        """Normalize empty-string save paths to None (disabled)."""
        if self.output_path is not None and self.output_path.strip() == "":
            self.output_path = None
        if self.input_save_path is not None and self.input_save_path.strip() == "":
            self.input_save_path = None

    def _adjust_quant_config(self):
        """validate and adjust"""

        
        ncfg = self.nunchaku_config
        if ncfg is None or isinstance(ncfg, NunchakuConfig):
            return
        ncfg.validate()

        
        if ncfg.transformer_weights_path:
            self.transformer_weights_path = ncfg.transformer_weights_path

        if not ncfg.enable_svdquant or not ncfg.transformer_weights_path:
            self.nunchaku_config = None
        else:
            self.nunchaku_config = NunchakuConfig(
                precision=self.nunchaku_config.quantization_precision,
                rank=self.nunchaku_config.quantization_rank,
                act_unsigned=self.nunchaku_config.quantization_act_unsigned,
                transformer_weights_path=self.nunchaku_config.transformer_weights_path,
            )

    def _adjust_offload(self):
        
        if current_platform.get_device_total_memory() / BYTES_PER_GB < 30:
            logger.info("Enabling all offloading for GPU with low device memory")
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = True
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = True
        elif self.pipeline_config.task_type.is_image_gen():
            logger.info(
                "Disabling some offloading (except dit, text_encoder) for image generation model"
            )
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = False
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = False
        else:
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = True
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = True

    def _adjust_attention_backend(self):
        if self.attention_backend in ["fa3", "fa4"]:
            self.attention_backend = "fa"

        
        if self.attention_backend_config is None:
            self.attention_backend_config = addict.Dict()
        elif isinstance(self.attention_backend_config, str):
            self.attention_backend_config = addict.Dict(
                self._parse_attention_backend_config(self.attention_backend_config)
            )

        if self.ring_degree > 1:
            if self.attention_backend is not None and self.attention_backend not in (
                "fa",
                "sage_attn",
            ):
                raise ValueError(
                    "Ring Attention is only supported for flash attention or sage attention backend for now"
                )
            if self.attention_backend is None:
                self.attention_backend = "fa"
                logger.info(
                    "Ring Attention is currently only supported for flash attention or sage attention; "
                    "attention_backend has been automatically set to flash attention"
                )

        if self.attention_backend is None and self.backend != Backend.DIFFUSERS:
            self._set_default_attention_backend()

    def _adjust_warmup(self):
        if self.warmup_resolutions is not None:
            self.warmup = True

        if self.warmup:
            logger.info(
                "Warmup enabled, the launch time is expected to be longer than usual"
            )

    def _adjust_network_ports(self):
        self.port = self.settle_port(self.port)
        initial_scheduler_port = self.scheduler_port + (
            random.randint(0, 100) if self.scheduler_port == 5555 else 0
        )
        self.scheduler_port = self.settle_port(initial_scheduler_port)
        initial_master_port = (
            self.master_port
            if self.master_port is not None
            else (30005 + random.randint(0, 100))
        )
        self.master_port = self.settle_port(initial_master_port, 37)

    def _adjust_parallelism(self):
        if self.tp_size is None:
            self.tp_size = 1

        if self.hsdp_shard_dim is None:
            self.hsdp_shard_dim = self.num_gpus

        
        if self.sp_degree is None:
            num_gpus_per_group = self.dp_size * self.tp_size
            if self.enable_cfg_parallel:
                num_gpus_per_group *= 2
            if self.num_gpus % num_gpus_per_group == 0:
                self.sp_degree = self.num_gpus // num_gpus_per_group
            else:
                
                self.sp_degree = 1

        if (
            self.ulysses_degree is None
            and self.ring_degree is None
            and self.sp_degree != 1
        ):
            self.ulysses_degree = self.sp_degree
            logger.info(
                f"Automatically set ulysses_degree=sp_degree={self.ulysses_degree} for best performance"
            )

        if self.ulysses_degree is None:
            self.ulysses_degree = 1
            logger.debug(
                f"Ulysses degree not set, using default value {self.ulysses_degree}"
            )

        if self.ring_degree is None:
            self.ring_degree = 1
            logger.debug(f"Ring degree not set, using default value {self.ring_degree}")

    def _adjust_platform_specific(self):
        if current_platform.is_mps():
            self.use_fsdp_inference = False
            self.dit_layerwise_offload = False

        
        if not envs.SGLANG_CACHE_DIT_ENABLED:
            pipeline_name_lower = self.pipeline_config.__class__.__name__.lower()
            if (
                ("wan" in pipeline_name_lower or "mova" in pipeline_name_lower)
                and self.dit_layerwise_offload is None
                and current_platform.enable_dit_layerwise_offload_for_wan_by_default()
            ):
                logger.info(
                    f"Automatically enable dit_layerwise_offload for {self.pipeline_config.__class__.__name__} "
                    "for low memory and performance balance"
                )
                self.dit_layerwise_offload = True

    def _adjust_autocast(self):
        if self.disable_autocast is None:
            self.disable_autocast = not self.pipeline_config.enable_autocast

    def _parse_attention_backend_config(self, config_str: str) -> dict[str, Any]:
        """parse attention backend config from string."""
        if not config_str:
            return {}

        
        if os.path.exists(config_str):
            if config_str.endswith((".yaml", ".yml")):
                with open(config_str, "r") as f:
                    return yaml.safe_load(f)
            elif config_str.endswith(".json"):
                with open(config_str, "r") as f:
                    return json.load(f)

        
        try:
            return json.loads(config_str)
        except json.JSONDecodeError:
            pass

        
        try:
            config = {}
            pairs = config_str.split(",")
            for pair in pairs:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.replace(".", "", 1).isdigit():
                    v = float(v) if "." in v else int(v)
                config[k] = v
            return config
        except Exception:
            raise ValueError(f"Could not parse attention backend config: {config_str}")

    def __post_init__(self):
        
        configure_logger(server_args=self)

        
        self._adjust_parameters()

        
        self._validate_parameters()

        
        try:
            safe_args = _sanitize_for_logging(self, key_hint="server_args")
            logger.info("server_args: %s", json.dumps(safe_args, ensure_ascii=False))
        except Exception:
            
            logger.info(f"server_args: {self}")

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-id",
            type=str,
            default=ServerArgs.model_id,
            help=(
                "Override the model ID used for config resolution. "
                "Useful when --model-path is a local directory whose name does not match "
                "any registered HF repo name. Should be the repo name portion of the HF ID "
                "(e.g. 'Qwen-Image' for 'Qwen/Qwen-Image')."
            ),
        )
        
        parser.add_argument(
            "--attention-backend",
            type=str,
            default=None,
            help=(
                "The attention backend to use. For SGLang-native pipelines, use "
                "values like fa, torch_sdpa, sage_attn, etc. For diffusers pipelines, "
                "use diffusers attention backend names such as flash, _flash_3_hub, "
                "sage, or xformers."
            ),
        )
        parser.add_argument(
            "--attention-backend-config",
            type=str,
            default=None,
            help="Configuration for the attention backend. Can be a JSON string, a path to a JSON/YAML file, or key=value pairs.",
        )
        parser.add_argument(
            "--cache-dit-config",
            type=str,
            default=ServerArgs.cache_dit_config,
            help="Path to a Cache-DiT YAML/JSON config. Enables cache-dit for diffusers backend.",
        )

        
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=ServerArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=ServerArgs.revision,
            help="The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=ServerArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=None,
            help="The tensor parallelism size. Defaults to 1 if not specified.",
        )
        parser.add_argument(
            "--sp-degree",
            type=int,
            default=None,
            help="The sequence parallelism size. If not specified, will use all remaining GPUs after accounting for TP and DP.",
        )
        parser.add_argument(
            "--ulysses-degree",
            type=int,
            default=ServerArgs.ulysses_degree,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--ring-degree",
            type=int,
            default=ServerArgs.ring_degree,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--enable-cfg-parallel",
            action="store_true",
            default=ServerArgs.enable_cfg_parallel,
            help="Enable cfg parallel.",
        )
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            "--dp",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )

        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=ServerArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=None,
            help="The data parallelism shards. Defaults to num_gpus if not specified.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Timeout for torch.distributed operations in seconds. "
            "Increase this value if you encounter 'Connection closed by peer' errors after the service is idle. ",
        )

        
        parser.add_argument(
            "--prompt-file-path",
            type=str,
            default=ServerArgs.prompt_file_path,
            help="Path to a text file containing prompts (one per line) for batch processing",
        )

        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=ServerArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference."
            + "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )

        
        parser.add_argument(
            "--warmup",
            action=StoreBoolean,
            default=ServerArgs.warmup,
            help="Perform some warmup after server starts (if `--warmup-resolutions` is specified) or before processing the first request (if `--warmup-resolutions` is not specified)."
            "Recommended to enable when benchmarking to ensure fair comparison and best performance."
            "When enabled with `--warmup-resolutions` unspecified, look for the line ending with `(with warmup excluded)` for actual processing time.",
        )
        parser.add_argument(
            "--warmup-resolutions",
            type=str,
            nargs="+",
            default=ServerArgs.warmup_resolutions,
            help="Specify resolutions for server to warmup. e.g., `--warmup-resolutions 256x256, 720x720`",
        )
        parser.add_argument(
            "--warmup-steps",
            type=int,
            default=ServerArgs.warmup_steps,
            help="The number of warmup steps to perform for each resolution.",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--dit-layerwise-offload",
            action=StoreBoolean,
            default=ServerArgs.dit_layerwise_offload,
            help="Enable layerwise CPU offload with async H2D prefetch overlap for supported DiT models (e.g., Wan, MOVA). "
            "Cannot be used together with cache-dit (SGLANG_CACHE_DIT_ENABLED), dit_cpu_offload, or use_fsdp_inference.",
        )
        parser.add_argument(
            "--dit-offload-prefetch-size",
            type=float,
            default=ServerArgs.dit_offload_prefetch_size,
            help="The size of prefetch for dit-layerwise-offload. If the value is between 0.0 and 1.0, it is treated as a ratio of the total number of layers. If the value is >= 1, it is treated as the absolute number of layers. 0.0 means prefetch 1 layer (lowest memory). Values above 0.5 might have peak memory close to no offload but worse performance.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help="Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help='Pin memory for CPU offload. Only added as a temp workaround if it throws "CUDA error: invalid argument". '
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        
        NunchakuSVDQuantArgs.add_cli_args(parser)

        
        parser.add_argument(
            "--master-port",
            type=int,
            default=ServerArgs.master_port,
            help="Master port for distributed inference. If not set, a random free port will be used.",
        )
        parser.add_argument(
            "--scheduler-port",
            type=int,
            default=ServerArgs.scheduler_port,
            help="Port for the scheduler server.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="Host for the HTTP API server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="Port for the HTTP API server.",
        )
        parser.add_argument(
            "--webui",
            action=StoreBoolean,
            default=ServerArgs.webui,
            help="Whether to use webui for better display",
        )

        parser.add_argument(
            "--webui-port",
            type=int,
            default=ServerArgs.webui_port,
            help="Whether to use webui for better display",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=ServerArgs.output_path,
            help='Directory path to save generated images/videos. Set to "" to disable persistent saving.',
        )
        parser.add_argument(
            "--input-save-path",
            type=str,
            default=ServerArgs.input_save_path,
            help='Directory path to save uploaded input images/videos. Set to "" to disable persistent saving.',
        )

        
        parser.add_argument(
            "--lora-path",
            type=str,
            default=ServerArgs.lora_path,
            help="The path to the LoRA adapter weights (can be local file path or HF hub id) to launch with",
        )
        parser.add_argument(
            "--lora-nickname",
            type=str,
            default=ServerArgs.lora_nickname,
            help="The nickname for the LoRA adapter to launch with",
        )
        parser.add_argument(
            "--lora-scale",
            type=float,
            default=ServerArgs.lora_scale,
            help="LoRA scale for merging (e.g., 0.125 for Hyper-SD). Same as lora_scale in Diffusers",
        )
        
        PipelineConfig.add_cli_args(parser)

        
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            choices=Backend.choices(),
            default=ServerArgs.backend.value,
            help="The model backend to use. 'auto' prefers sglang native and falls back to diffusers. "
            "'sglang' uses native optimized implementation. 'diffusers' uses vanilla diffusers pipeline.",
        )
        return parser

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    @property
    def scheduler_endpoint(self):
        """
        Internal endpoint for scheduler.
        Prefers the configured host but normalizes localhost -> 127.0.0.1 to avoid ZMQ issues.
        """
        scheduler_host = self.host
        if scheduler_host is None or scheduler_host == "localhost":
            scheduler_host = "127.0.0.1"
        return f"tcp://{scheduler_host}:{self.scheduler_port}"

    def settle_port(
        self, port: int, port_inc: int = 42, max_attempts: int = 100
    ) -> int:
        """
        Find an available port with retry logic.
        """
        attempts = 0
        original_port = port

        while attempts < max_attempts:
            if is_port_available(port):
                if attempts > 0:
                    logger.info(
                        f"Port {original_port} was unavailable, using port {port} instead"
                    )
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts "
            f"(started from port {original_port})"
        )

    @staticmethod
    def _extract_component_paths(
        unknown_args: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """
        Extract dynamic ``--<component>-path`` args from unrecognised CLI args.
        """
        component_paths: dict[str, str] = {}
        remaining: list[str] = []
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            key_part = arg.split("=", 1)[0] if "=" in arg else arg
            if key_part.startswith("--") and key_part.endswith("-path"):
                component = key_part[2:-5].replace("-", "_")
                if "=" in arg:
                    component_paths[component] = arg.split("=", 1)[1]
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith(
                    "-"
                ):
                    i += 1
                    component_paths[component] = unknown_args[i]
                else:
                    remaining.append(arg)
                    i += 1
                    continue
            else:
                remaining.append(arg)
            i += 1

        
        for component, path in component_paths.items():
            path = os.path.expanduser(path)
            component_paths[component] = path
        return component_paths, remaining

    @classmethod
    def from_cli_args(
        cls, args: argparse.Namespace, unknown_args: list[str] | None = None
    ) -> "ServerArgs":
        if unknown_args is None:
            unknown_args = []

        
        dynamic_paths, remaining = cls._extract_component_paths(unknown_args)
        if remaining:
            raise SystemExit(f"error: unrecognized arguments: {' '.join(remaining)}")

        provided_args = cls.get_provided_args(args, unknown_args)

        
        config_file = provided_args.get("config")
        if config_file:
            config_args = cls.load_config_file(config_file)
            provided_args = {**config_args, **provided_args}

        if dynamic_paths:
            existing = dict(provided_args.get("component_paths") or {})
            existing.update(dynamic_paths)
            provided_args["component_paths"] = existing

        return cls.from_dict(provided_args)

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> "ServerArgs":
        """Create a ServerArgs object from a dictionary."""
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        server_args_kwargs: dict[str, Any] = {}

        component_paths = dict(kwargs.get("component_paths") or {})
        if component_paths:
            server_args_kwargs["component_paths"] = component_paths

        for attr in attrs:
            if attr == "pipeline_config":
                pipeline_config = PipelineConfig.from_kwargs(kwargs)
                logger.debug(f"Using PipelineConfig: {type(pipeline_config)}")
                server_args_kwargs["pipeline_config"] = pipeline_config
            elif attr == "nunchaku_config":
                nunchaku_config = NunchakuSVDQuantArgs.from_dict(kwargs)
                server_args_kwargs["nunchaku_config"] = nunchaku_config
            elif attr in kwargs:
                server_args_kwargs[attr] = kwargs[attr]

        return cls(**server_args_kwargs)

    @staticmethod
    def load_config_file(config_file: str) -> dict[str, Any]:
        """Load a config file."""
        if config_file.endswith(".json"):
            with open(config_file, "r") as f:
                return json.load(f)
        elif config_file.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ServerArgs":
        
        if "backend" in kwargs and isinstance(kwargs["backend"], str):
            kwargs["backend"] = Backend.from_string(kwargs["backend"])

        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    @staticmethod
    def get_provided_args(
        args: argparse.Namespace, unknown_args: list[str]
    ) -> dict[str, Any]:
        """Get the arguments provided by the user."""
        provided_args = {}
        
        
        raw_argv = sys.argv + unknown_args

        
        
        provided_arg_names = set()
        for arg in raw_argv:
            if arg.startswith("--"):
                
                arg_name = arg.split("=", 1)[0].replace("-", "_").lstrip("_")
                provided_arg_names.add(arg_name)

        
        for k, v in vars(args).items():
            if k in provided_arg_names:
                provided_args[k] = v

        return provided_args

    def _validate_pipeline(self):
        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in ServerArgs")

        self.pipeline_config.check_pipeline_config()

    def _validate_offload(self):
        
        if self.dit_offload_prefetch_size > 1 and (
            isinstance(self.dit_offload_prefetch_size, float)
            and not self.dit_offload_prefetch_size.is_integer()
        ):
            self.dit_offload_prefetch_size = int(
                math.floor(self.dit_offload_prefetch_size)
            )
            logger.info(
                f"Invalid --dit-offload-prefetch-size value passed, truncated to: {self.dit_offload_prefetch_size}"
            )

        if 0.5 <= self.dit_offload_prefetch_size < 1.0:
            logger.info(
                "We do not recommend --dit-offload-prefetch-size to be between 0.5 and 1.0"
            )

        
        if self.dit_layerwise_offload:
            if self.dit_offload_prefetch_size < 0.0:
                raise ValueError("dit_offload_prefetch_size must be non-negative")

            if self.use_fsdp_inference:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference."
                )
                self.use_fsdp_inference = False

            if self.dit_cpu_offload is None:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload."
                )
                self.dit_cpu_offload = False

            if envs.SGLANG_CACHE_DIT_ENABLED:
                raise ValueError(
                    "dit_layerwise_offload cannot be enabled together with cache-dit. "
                    "cache-dit may reuse skipped blocks whose weights have been released by layerwise offload, "
                    "causing shape mismatch errors. "
                    "Please disable either --dit-layerwise-offload or SGLANG_CACHE_DIT_ENABLED."
                )

    def _validate_parallelism(self):
        if self.sp_degree > self.num_gpus or self.num_gpus % self.sp_degree != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by sp_degree ({self.sp_degree})"
            )

        if (
            self.hsdp_replicate_dim > self.num_gpus
            or self.num_gpus % self.hsdp_replicate_dim != 0
        ):
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by hsdp_replicate_dim ({self.hsdp_replicate_dim})"
            )

        if (
            self.hsdp_shard_dim > self.num_gpus
            or self.num_gpus % self.hsdp_shard_dim != 0
        ):
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by hsdp_shard_dim ({self.hsdp_shard_dim})"
            )

        if self.num_gpus % self.dp_size != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be divisible by dp_size ({self.dp_size})"
            )

        if self.dp_size < 1:
            raise ValueError("--dp-size must be a natural number")

        if self.dp_size > 1:
            raise ValueError("DP is not yet supported")

        num_gpus_per_group = self.dp_size * self.tp_size
        if self.enable_cfg_parallel:
            num_gpus_per_group *= 2

        if self.num_gpus % num_gpus_per_group != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be divisible by (dp_size * tp_size{' * 2' if self.enable_cfg_parallel else ''}) = {num_gpus_per_group}"
            )

        if self.sp_degree != self.ring_degree * self.ulysses_degree:
            raise ValueError(
                f"sp_degree ({self.sp_degree}) must equal ring_degree * ulysses_degree "
                f"({self.ring_degree} * {self.ulysses_degree} = {self.ring_degree * self.ulysses_degree})"
            )

        if os.getenv("SGLANG_CACHE_DIT_ENABLED", "").lower() == "true":
            has_sp = self.sp_degree > 1
            has_tp = self.tp_size > 1
            if has_sp and has_tp:
                logger.warning(
                    "cache-dit is enabled with hybrid parallelism (SP + TP). "
                    "Proceeding anyway (SGLang integration may support this mode)."
                )

    def _validate_cfg_parallel(self):
        if self.enable_cfg_parallel and self.num_gpus == 1:
            raise ValueError(
                "CFG Parallelism is enabled via `--enable-cfg-parallel`, but num_gpus == 1"
            )

    def _set_default_attention_backend(self) -> None:
        """Configure ROCm defaults when users do not specify an attention backend."""
        if current_platform.is_rocm():
            default_backend = AttentionBackendEnum.AITER.name.lower()
            self.attention_backend = default_backend
            logger.info(
                "Attention backend not specified. Using '%s' by default on ROCm "
                "to match SGLang SRT defaults.",
                default_backend,
            )


@dataclasses.dataclass
class PortArgs:
    
    scheduler_input_ipc_name: str

    
    nccl_port: int

    
    rpc_ipc_name: str

    
    metrics_ipc_name: str

    
    master_port: int | None = None

    @staticmethod
    def from_server_args(
        server_args: ServerArgs, dp_rank: Optional[int] = None
    ) -> "PortArgs":
        if server_args.nccl_port is None:
            nccl_port = server_args.scheduler_port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = server_args.nccl_port

        
        return PortArgs(
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            master_port=server_args.master_port,
        )


_global_server_args = None


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the inference arguments from the command line arguments.
    """
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args, unknown_args = parser.parse_known_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args, unknown_args)
    return server_args


def set_global_server_args(server_args: ServerArgs):
    """
    Set the global sgl_diffusion config for each process
    """
    global _global_server_args
    _global_server_args = server_args


def get_global_server_args() -> ServerArgs:
    if _global_server_args is None:
        
        
        
        
        raise ValueError("Global sgl_diffusion args is not set.")
    return _global_server_args
