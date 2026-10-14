# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from collections import OrderedDict
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from packaging import version
from torch import Tensor

from verl.models.mcore.util import (
    build_vlm_attn_mask_bshd,
    postprocess_bshd_engine,
    preprocess_bshd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output, postprocess_thd_engine


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    assert version.parse(mcore.__version__) >= version.parse("0.13.0"), (
        "Fused forward patching requires mecore >= 0.13.0"
    )
    model = _get_patching_model(model)
    if model is not None:
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model.forward = model.forward_backup


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  
        post_process: bool = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            
            
            
            
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def fused_forward_model_engine(vision_model: bool = False):
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        config = unwrap_model(model).config
        min_local_rows = (
            config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
        )

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids,
            pre_process=pre_process,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        attention_mask = None
        if vision_model:
            input_ids_rmpad = input_ids.to_padded_tensor(pad_token_id)
            seqlens_in_batch = input_ids.offsets().diff().to(input_ids.device)
            max_seq_len = input_ids_rmpad.shape[1]
            attention_mask = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(
                0
            ) < seqlens_in_batch.unsqueeze(1)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
        )
        labels_rmpad = labels_rmpad.contiguous()
        output_orig: CausalLMOutputForPPO = model(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
            )
            output["entropy"] = entropy

        return output

    return fused_forward_model_engine_inner


def fused_forward_model_engine_bshd(vision_model: bool = False):
    """Fused LM-head path for Qwen3.5 GDN's non-packed BSHD format.

    Gated Delta Net currently rejects ``PackedSeqParams``. This adapter keeps
    the proven BSHD/CP input path while the patched GPT model computes only
    fused per-token log-probabilities and entropy, never a full vocabulary
    logits tensor.
    """

    def fused_forward_model_engine_bshd_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        forced_max_seqlen: Optional[int] = None,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        input_ids_bshd, local_attention_mask, position_ids_bshd = (
            preprocess_bshd_engine(
                input_ids,
                pre_process=pre_process,
                use_fp8_padding=use_fp8_padding,
                forced_max_seqlen=forced_max_seqlen,
            )
        )
        labels_bshd, _, _ = preprocess_bshd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            forced_max_seqlen=forced_max_seqlen,
        )
        labels_bshd = labels_bshd.contiguous()

        model_kwargs = {}
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            if key in multi_modal_inputs:
                model_kwargs[key] = multi_modal_inputs[key].to(input_ids.device)

        if vision_model:
            model_input_ids, model_attention_mask = build_vlm_attn_mask_bshd(
                input_ids,
                input_ids.shape[0],
                pad_token_id,
                forced_max_seqlen=forced_max_seqlen,
            )
            position_ids = None
        else:
            model_input_ids = input_ids_bshd
            model_attention_mask = local_attention_mask
            position_ids = position_ids_bshd

        output_orig: CausalLMOutputForPPO = model(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            packed_seq_params=None,
            labels=labels_bshd,
            temperature=temperature,
            **model_kwargs,
        )
        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs.reshape_as(labels_bshd)
        output = {
            "log_probs": postprocess_bshd_engine(
                log_probs,
                local_attention_mask,
                post_process=post_process,
            )
        }
        if calculate_entropy:
            entropy = output_orig.entropy.reshape_as(labels_bshd)
            output["entropy"] = postprocess_bshd_engine(
                entropy,
                local_attention_mask,
                post_process=post_process,
            )
        return output

    return fused_forward_model_engine_bshd_inner


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    decoder_extra_block_kwargs = extra_block_kwargs or {}
    if getattr(model.config, "moe_n_hash_layers", 0) > 0 and input_ids is not None:
        decoder_extra_block_kwargs["input_ids"] = input_ids

    
    decoder_output = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **decoder_extra_block_kwargs,
        **kwargs,
    )
    hidden_states = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    
    
    
    
    if (
        packed_seq_params is None
        and labels is not None
        and hidden_states.ndim == 3
        and labels.ndim == 2
        and hidden_states.shape[0] == labels.shape[1]
        and hidden_states.shape[1] == labels.shape[0]
    ):
        hidden_states = hidden_states.transpose(0, 1).contiguous()

    
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        
        output_weight = model.embedding.word_embeddings.weight

    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        output_weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
