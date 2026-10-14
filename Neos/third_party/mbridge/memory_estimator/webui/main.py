import argparse
import glob
import json
import os
import tempfile
from typing import Optional

import requests
from estimate_013 import estimate_from_config
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from megatron.core import parallel_state as mpu
from pydantic import BaseModel, field_validator

from mbridge import AutoBridge


WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()


app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(WEBUI_DIR, "index.html"))


@app.get("/style.css")
async def read_css():
    return FileResponse(os.path.join(WEBUI_DIR, "style.css"))


@app.get("/script.js")
async def read_js():
    return FileResponse(os.path.join(WEBUI_DIR, "script.js"))


SUPPORTED_MODELS = [
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "moonshotai/Moonlight-16B-A3B",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "XiaomiMiMo/MiMo-7B-RL",
]


@app.get("/local-hf-configs")
async def get_supported_models():
    """Return the list of HF model identifiers supported by the UI."""
    return SUPPORTED_MODELS


@app.get("/get-megatron-config/{model_path:path}")
async def get_remote_hf_config(model_path: str):
    """Fetch the HuggingFace config.json for the given model id."""
    url = f"https://huggingface.co/{model_path}/raw/main/config.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Failed to fetch config from {url}: {str(e)}"}


class MBridgeEstimateConfig(BaseModel):
    hf_model_path: str
    custom_hf_config: Optional[dict] = None  

    
    num_gpus: int = 8
    mbs: int = 1
    seq_len: int = 4096
    use_distributed_optimizer: bool = True
    
    recompute_granularity: str = "selective"
    recompute_method: str = "uniform"
    recompute_num_layers: Optional[int] = 1

    
    recompute_modules: Optional[list[str]] = None

    
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    vpp: Optional[int] = None
    etp: Optional[int] = None

    
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None

    
    pipeline_model_parallel_layout: Optional[str] = None  

    @field_validator("num_gpus")
    def num_gpus_must_be_multiple_of_8(cls, v):
        if v <= 0 or v % 8 != 0:
            raise ValueError("must be a positive multiple of 8")
        return v


def patch_parallel_states(config: MBridgeEstimateConfig):
    from mbridge.core.parallel_states import ParallelStates

    ParallelStates.get_default_parallel_states = lambda: ParallelStates(
        tp_size=config.tp,
        pp_size=config.pp,
        ep_size=config.ep,
        cp_size=config.cp,
        vpp_size=config.vpp,
        etp_size=config.etp,
    )


@app.post("/estimate_with_mbridge")
async def estimate_with_mbridge(config: MBridgeEstimateConfig):
    
    if config.num_gpus <= 0 or config.num_gpus % 8 != 0:
        return {"error": "Total number of GPUs must be a positive multiple of 8."}

    parallel_product = config.tp * config.pp * config.cp
    if parallel_product == 0:  
        return {"error": "Parallelism dimensions (TP, PP, CP) cannot be zero."}

    if config.num_gpus % parallel_product != 0:
        return {
            "error": f"Number of GPUs ({config.num_gpus}) must be divisible by the product of TP*PP*CP ({parallel_product})."
        }

    patch_parallel_states(config)

    
    hf_model_path = config.hf_model_path
    
    
    if config.custom_hf_config:
        try:
            
            with tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                suffix=".json",
                dir=os.path.join("/dev/shm"),
            ) as tmp:
                json.dump(config.custom_hf_config, tmp)
                tmp_path = tmp.name

            
            from transformers import AutoConfig

            AutoConfig.trust_remote_code = True
            bridge = AutoBridge.from_pretrained(tmp_path)
            tf_config = bridge.config
            hf_config = bridge.hf_config

        finally:
            
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        
        if not os.path.isabs(hf_model_path) and not hf_model_path.startswith(
            ("http", "./", "../")
        ):
            hf_model_path = os.path.join("/dev/shm", hf_model_path)
        bridge = AutoBridge.from_pretrained(hf_model_path)
        tf_config = bridge.config
        hf_config = bridge.hf_config

    
    
    tf_config.tensor_model_parallel_size = config.tp
    tf_config.pipeline_model_parallel_size = config.pp
    tf_config.expert_model_parallel_size = config.ep
    tf_config.context_parallel_size = config.cp
    tf_config.recompute_granularity = config.recompute_granularity
    tf_config.recompute_method = config.recompute_method
    tf_config.recompute_num_layers = config.recompute_num_layers
    
    tf_config.recompute_modules = (
        config.recompute_modules if config.recompute_modules is not None else []
    )
    
    tf_config.account_for_embedding_in_pipeline_split = (
        config.account_for_embedding_in_pipeline_split
    )
    tf_config.account_for_loss_in_pipeline_split = (
        config.account_for_loss_in_pipeline_split
    )
    tf_config.num_layers_per_virtual_pipeline_stage = (
        config.vpp if config.vpp and config.vpp > 1 else None
    )

    if config.num_layers_in_first_pipeline_stage is not None:
        tf_config.num_layers_in_first_pipeline_stage = (
            config.num_layers_in_first_pipeline_stage
        )
    if config.num_layers_in_last_pipeline_stage is not None:
        tf_config.num_layers_in_last_pipeline_stage = (
            config.num_layers_in_last_pipeline_stage
        )

    
    if config.pipeline_model_parallel_layout:
        from megatron.core.transformer.pipeline_parallel_layer_layout import (
            PipelineParallelLayerLayout,
        )

        tf_config.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
            config.pipeline_model_parallel_layout, config.pp
        )
    

    
    args = argparse.Namespace()
    args.micro_batch_size = config.mbs
    args.seq_length = config.seq_len
    args.use_distributed_optimizer = config.use_distributed_optimizer
    args.data_parallel_size = config.num_gpus // parallel_product
    args.expert_tensor_parallel_size = config.etp if config.etp else 1

    
    args.transformer_impl = "transformer_engine"
    args.fp8 = False
    args.num_experts = getattr(tf_config, "num_moe_experts", 1)  
    args.moe_grouped_gemm = True  
    args.qk_layernorm = tf_config.qk_layernorm
    args.multi_latent_attention = "deepseek" in getattr(hf_config, "model_type", "")
    if hasattr(hf_config, "vocab_size"):
        args.padded_vocab_size = hf_config.vocab_size
    elif hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "vocab_size"):
        args.padded_vocab_size = hf_config.text_config.vocab_size
    else:
        raise ValueError(f"Vocab size attribute not found for the current model configuration {type(hf_config).__name__}. "
                         f"It should be either `vocab_size` (for text-only models) or `text_config.vocab_size` (for multimodal models).")

    if hasattr(hf_config, "max_position_embeddings"):
        args.max_position_embeddings = hf_config.max_position_embeddings
    elif hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "max_position_embeddings"):
        args.max_position_embeddings = hf_config.text_config.max_position_embeddings
    else:
        raise ValueError(f"Max position embeddings attribute not found for the current model configuration {type(hf_config).__name__}. "
                         f"It should be either `max_position_embeddings` (for text-only models) or `text_config.max_position_embeddings` (for multimodal models).")
    
    args.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    args.world_size = config.num_gpus

    
    aggregated_reports, raw_chunk_reports = estimate_from_config(tf_config, args)

    processed_reports = []
    for rpt in aggregated_reports:
        p = rpt.copy()
        p.pop("details", None)
        processed_reports.append(p)

    return {"processed_report": processed_reports, "raw_report": raw_chunk_reports}
