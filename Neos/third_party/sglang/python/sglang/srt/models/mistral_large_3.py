# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3.py
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable

import regex as re
import torch

from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    
    remapping = {
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",  
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",  
        r"layers\.(\d+)\.attention\.wq_a\.(\w+)": r"model.layers.\1.self_attn.q_a_proj.\2",  
        r"layers\.(\d+)\.attention\.q_a_norm\.weight": r"model.layers.\1.self_attn.q_a_layernorm.weight",  
        r"layers\.(\d+)\.attention\.wq_b\.(\w+)": r"model.layers.\1.self_attn.q_b_proj.\2",  
        r"layers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)": r"model.layers.\1.self_attn.kv_a_proj_with_mqa.\2",  
        r"layers\.(\d+)\.attention\.kv_a_norm\.weight": r"model.layers.\1.self_attn.kv_a_layernorm.weight",  
        r"layers\.(\d+)\.attention\.wkv_b\.(\w+)": r"model.layers.\1.self_attn.kv_b_proj.\2",  
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",  
        
        r"layers\.(\d+)\.attention\.k_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.k_scale",  
        r"layers\.(\d+)\.attention\.q_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.q_scale",  
        r"layers\.(\d+)\.attention\.v_fake_quantizer\.qscale_act": r"model.layers.\1.self_attn.mla_attn.mla_attn.v_scale",  
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",  
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",  
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",  
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",  
        r"layers\.(\d+)\.gate\.weight": r"model.layers.\1.mlp.gate.weight",  
        r"layers\.(\d+)\.shared_experts\.w1\.(\w+)": r"model.layers.\1.mlp.shared_experts.gate_proj.\2",  
        r"layers\.(\d+)\.shared_experts\.w2\.(\w+)": r"model.layers.\1.mlp.shared_experts.down_proj.\2",  
        r"layers\.(\d+)\.shared_experts\.w3\.(\w+)": r"model.layers.\1.mlp.shared_experts.up_proj.\2",  
        r"layers\.(\d+)\.experts\.(\d+)\.w1\.(\w+)": r"model.layers.\1.mlp.experts.\2.gate_proj.\3",  
        r"layers\.(\d+)\.experts\.(\d+)\.w2\.(\w+)": r"model.layers.\1.mlp.experts.\2.down_proj.\3",  
        r"layers\.(\d+)\.experts\.(\d+)\.w3\.(\w+)": r"model.layers.\1.mlp.experts.\2.up_proj.\3",  
        r"layers\.(\d+)\.router_biases": r"model.layers.\1.mlp.gate.e_score_correction_bias",  
        r"norm\.weight": "model.norm.weight",  
        r"tok_embeddings\.weight": "model.embed_tokens.weight",  
        r"output\.weight": "lm_head.weight",  
    }
    

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._iterable_remap_mistral_to_ds(weights))

    def _iterable_remap_mistral_to_ds(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Remap Mistral parameters to DeepseekV2 parameters."""
        for name, loaded_weight in weights:
            for k, v in self.remapping.items():
                match = re.fullmatch(k, name)
                if match:
                    name = re.sub(k, v, name)
                    break
            else:
                import logging

                logging.warning(f"Unrecognized weight: {name}. Skipping.")
                continue

            
            
            

            
            
            if name.endswith(".qscale_act"):
                name = re.sub(r"\.qscale_act$", ".input_scale", name)
            elif name.endswith(".qscale_weight"):
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)

            yield name, loaded_weight


EntryClass = MistralLarge3ForCausalLM
