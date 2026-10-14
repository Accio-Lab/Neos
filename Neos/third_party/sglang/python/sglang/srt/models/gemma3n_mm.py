import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import (
    Gemma3nAudioConfig,
    Gemma3nConfig,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
    PreTrainedModel,
)
from transformers.models.auto.modeling_auto import AutoModel

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3n_audio import Gemma3nAudioEncoder
from sglang.srt.models.gemma3n_causal import Gemma3nRMSNorm, Gemma3nTextModel
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Gemma3nImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class Gemma3nAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""
    input_features_mask: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length)`"""


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.multimodal_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embedding", prefix),
        )

        self.hard_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.soft_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.embedding_projection = ReplicatedLinear(
            self.multimodal_hidden_size,
            self.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("embedding_projection", prefix),
        )

        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            self.text_hidden_size,
            eps=self.eps,
            with_scale=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            
            out_of_vocab_id = self.vocab_size - 1
            adjusted_ids = input_ids - self.vocab_offset
            adjusted_ids = torch.where(adjusted_ids < 0, out_of_vocab_id, adjusted_ids)
            adjusted_ids = torch.where(
                adjusted_ids >= self.vocab_size, out_of_vocab_id, adjusted_ids
            )
            hard_emb = self.embedding(adjusted_ids)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj, _ = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


class Gemma3nForConditionalGeneration(PreTrainedModel):
    config_class = Gemma3nConfig
    """Gemma3n multimodal model for conditional generation."""

    
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".out_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "out_proj": ("proj", 0),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma3nConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        prefix = add_prefix("model", prefix)

        
        
        self.vision_tower = AutoModel.from_config(config=config.vision_config)

        self.embed_vision = Gemma3nMultimodalEmbedder(
            config.vision_config,
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("embed_vision", prefix),
        )

        
        self.embed_audio = Gemma3nMultimodalEmbedder(
            config.audio_config,
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("embed_audio", prefix),
        )

        self.audio_tower = Gemma3nAudioEncoder(
            config.audio_config,
            quant_config=quant_config,
            prefix=add_prefix("audio_tower", prefix),
        )

        self.vocab_size = config.text_config.vocab_size
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input

        
        self.language_model = Gemma3nTextModel(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        
        self.logits_processor = LogitsProcessor(config.text_config)

        self.post_init()

    def pad_input_ids(
        self,
        input_ids: List[int],
        mm_inputs: MultimodalInputs,
    ) -> List[int]:
        """Pad input IDs with image and audio tokens."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        return self.config.text_config.sliding_window - 1

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        
        all_pixel_values = flatten_nested_list([item.feature for item in items])
        vision_outputs_list = []

        for pixel_values_batch in all_pixel_values:
            
            if pixel_values_batch.dim() == 5:
                pixel_values_batch = pixel_values_batch.squeeze(0)
            elif pixel_values_batch.dim() == 3:
                pixel_values_batch = pixel_values_batch.unsqueeze(0)
            elif pixel_values_batch.dim() != 4:
                raise ValueError(
                    f"Unexpected pixel_values shape: {pixel_values_batch.shape}"
                )

            
            batch_size = pixel_values_batch.shape[0]
            for i in range(batch_size):
                pixel_value = pixel_values_batch[i : i + 1]  
                pixel_value = pixel_value.to(
                    device=self.vision_tower.device, dtype=self.language_model.dtype()
                )
                vision_outputs = self.vision_tower(
                    pixel_values=pixel_value, do_pooling=False, return_dict=True
                ).last_hidden_state
                vision_outputs_list.append(vision_outputs)

        
        vision_outputs = torch.cat(vision_outputs_list, dim=0)

        
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)

        
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Projects the last hidden state from the audio encoder into language model space.

        Args:
            items: List of multimodal data items containing audio data.

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_audios, audio_length, embed_dim)`).
        """
        
        all_input_features = flatten_nested_list([item.feature for item in items])
        all_input_features_mask = flatten_nested_list(
            [~item.input_features_mask for item in items]
        )  

        
        audio_features_list = []

        for input_features, input_features_mask in zip(
            all_input_features, all_input_features_mask
        ):
            
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if input_features_mask.dim() == 1:
                input_features_mask = input_features_mask.unsqueeze(0)

            
            input_features = input_features.to(
                device=next(self.audio_tower.parameters()).device,
                dtype=self.language_model.dtype(),
            )
            input_features_mask = input_features_mask.to(device=input_features.device)

            
            audio_outputs, audio_mask = self.audio_tower(
                input_features, input_features_mask
            )

            
            audio_embeds = self.embed_audio(inputs_embeds=audio_outputs)
            audio_features_list.append(audio_embeds)

        
        if audio_features_list:
            audio_features = torch.cat(audio_features_list, dim=0)

            
            
            
            
            
            audio_padding_toks = torch.tensor(
                [[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device
            )
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
            audio_features = torch.where(
                audio_mask.unsqueeze(-1), audio_padding_embs, audio_features
            )

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = (
                self.config.audio_soft_tokens_per_image - audio_seq_len
            )
            extra_padding_features = audio_padding_embs.expand(
                audio_batch_size, extra_padding_tokens, audio_embed_dim
            )

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
            return audio_features
        else:
            return torch.empty(
                0,
                0,
                self.language_model.config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_per_layer_inputs(
        self, input_ids: torch.LongTensor
    ) -> Optional[torch.Tensor]:
        return self.language_model.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.language_model.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs: object,
    ) -> LogitsProcessor:
        """Forward pass for multimodal Gemma3n."""
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        positions += 1
        if input_ids is not None:
            
            per_layer_inputs_mask = torch.logical_and(
                input_ids >= 0, input_ids < self.vocab_size_per_layer_input
            )
            per_layer_inputs_tokens = torch.where(
                per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(
                per_layer_inputs_tokens
            )

        
        
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
            per_layer_inputs=per_layer_inputs,
        )

        
        return self.logits_processor(
            input_ids, hidden_states, self.language_model.embed_tokens, forward_batch
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        """Load weights for the model."""
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            name = re.sub(r"^model\.", "", name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_model" in name:
                    
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")
                
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    lora_pattern = re.compile(
        r"^language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self.lora_pattern.match(module_name))

    def get_hidden_dim(self, module_name, layer_idx):
        
        if module_name == "qkv_proj":
            return (
                self.config.hidden_size,
                self.config.head_dim
                * (
                    self.config.num_attention_heads
                    + self.config.num_key_value_heads * 2
                ),
            )
        elif module_name == "o_proj":
            return (
                self.config.head_dim * self.config.num_attention_heads,
                self.config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            assert len(set(self.config.intermediate_size)) == 1, (
                "Currently SGLang requires uniform intermediate size for all layers. "
                "Please file an issue if you need support for non-uniform intermediate sizes."
            )
            return self.config.hidden_size, self.config.intermediate_size[0] * 2
        elif module_name == "down_proj":
            assert len(set(self.config.intermediate_size)) == 1, (
                "Currently SGLang requires uniform intermediate size for all layers. "
                "Please file an issue if you need support for non-uniform intermediate sizes."
            )
            return self.config.intermediate_size[0], self.config.hidden_size
        else:
            raise NotImplementedError()


EntryClass = Gemma3nForConditionalGeneration
