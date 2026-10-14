
from __future__ import annotations

from copy import deepcopy
import functools
import json
import os
from typing import Any

from swift.dataset.preprocessor.core import MessagesPreprocessor
from swift.template import register_template
from swift.template.templates.qwen import Qwen3_5Template, Qwen3VLTemplate, QwenTemplateMeta


def _install_qwen35_moe_fused_lce() -> None:
    pass
    if os.environ.get("NEOS_QWEN35_FUSED_LCE", "0") != "1":
        return

    from liger_kernel.transformers.model.loss_utils import (
        LigerForCausalLMLoss,
        unpack_cross_entropy_result,
    )
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    model_cls = modeling_qwen3_5_moe.Qwen3_5MoeForConditionalGeneration
    if getattr(model_cls, "_neos_fused_lce", False):
        return
    original_forward = model_cls.forward

    @functools.wraps(original_forward)
    def fused_lce_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        logits_to_keep=0,
        **kwargs,
    ):
        
        
        if labels is None or not self.training:
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        
        
        
        loss_kwargs = {}
        for key in (
            "num_items_in_batch",
            "ignore_index",
            "shift_labels",
            "final_logit_softcapping",
            "return_token_accuracy",
        ):
            if key in kwargs:
                loss_kwargs[key] = kwargs.pop(key)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        kept_hidden_states = hidden_states[:, slice_indices, :].contiguous()

        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=self.config.text_config.hidden_size,
            **loss_kwargs,
        )
        loss, _, _ = unpack_cross_entropy_result(result)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = modeling_qwen3_5_moe.load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return modeling_qwen3_5_moe.Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )

    model_cls.forward = fused_lce_forward
    model_cls._neos_fused_lce = True


_install_qwen35_moe_fused_lce()






def _arguments_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("native function.arguments must be an object")
    return value


def canonical_messages_to_swift(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pass
    converted: list[dict[str, Any]] = []
    for raw in deepcopy(messages):
        role = raw.get("role")
        if role != "assistant":
            if role == "tool":
                converted.append({"role": "tool", "content": str(raw.get("content") or "")})
            else:
                converted.append({"role": role, "content": raw.get("content", "")})
            continue

        
        
        
        
        reasoning_key_present = "reasoning_content" in raw or "reasoning" in raw
        reasoning = raw.get("reasoning_content")
        if reasoning is None and reasoning_key_present:
            reasoning = raw.get("reasoning")
        reasoning = str(reasoning or "").strip()
        visible = raw.get("content")
        if visible is None:
            visible = ""
        if not isinstance(visible, str):
            raise ValueError("canonical assistant content must be text")
        visible = visible.strip()
        calls = raw.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("canonical assistant tool_calls must be a list")

        assistant_content = visible
        if reasoning:
            assistant_content = f"<think>\n{reasoning}\n</think>"
            if visible:
                assistant_content += f"\n\n{visible}"
            elif calls:
                
                
                assistant_content += "\n\n"
        if assistant_content or not calls:
            converted.append({"role": "assistant", "content": assistant_content})

        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise ValueError("canonical tool call must have type=function")
            function = call.get("function")
            if not isinstance(function, dict) or not str(function.get("name") or "").strip():
                raise ValueError("canonical tool call has no function name")
            converted.append(
                {
                    "role": "tool_call",
                    "content": {
                        "name": str(function["name"]),
                        "arguments": _arguments_object(function.get("arguments")),
                    },
                }
            )
    return converted


def _install_native_dataset_bridge() -> None:
    original = MessagesPreprocessor.preprocess
    if getattr(original, "_neos_openai_native_bridge", False):
        return

    @functools.wraps(original)
    def preprocess(self, row):
        messages = row.get("messages")
        if isinstance(messages, list) and any(
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and ("reasoning_content" in message or "tool_calls" in message)
            for message in messages
        ):
            row = dict(row)
            row["messages"] = canonical_messages_to_swift(messages)
        return original(self, row)

    preprocess._neos_openai_native_bridge = True
    MessagesPreprocessor.preprocess = preprocess


_install_native_dataset_bridge()


class Qwen35NeosTemplate(Qwen3_5Template):
    def _swift_prepare_inputs(self, inputs):
        return Qwen3VLTemplate._swift_prepare_inputs(self, inputs)


register_template(
    QwenTemplateMeta(
        "qwen3_5_neos",
        template_cls=Qwen35NeosTemplate,
        default_system=None,
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n\n"],
        thinking_prefix="<think>\n",
        non_thinking_prefix="<think>\n\n</think>\n\n",
        agent_template="qwen3_5",
        is_thinking=True,
    )
)
