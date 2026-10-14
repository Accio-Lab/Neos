import logging
from typing import List, Union

from transformers.processing_utils import ProcessorMixin

from sglang.srt.models.phi4mm import Phi4MMForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)




class Phi4MMProcessorAdapter(ProcessorMixin):
    def __init__(self, _processor) -> None:
        self._processor = _processor

    def __call__(self, **kwargs):
        result = self._processor(**kwargs)

        
        key_mapping = {
            "input_image_embeds": "pixel_values",
            "input_audio_embeds": "audio_features",
            "audio_embed_sizes": "audio_feature_lens",
        }
        for hf_key, sglang_key in key_mapping.items():
            if hf_key in result:
                result[sglang_key] = result[hf_key]
                del result[hf_key]

        
        
        
        filtered_result = {
            k: v
            for k, v in result.items()
            if v is not None and (not hasattr(v, "numel") or v.numel() > 0)
        }
        return filtered_result


class Phi4MMMultimodalProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.processor = Phi4MMProcessorAdapter(_processor)
        super().__init__(hf_config, server_args, self.processor, *args, **kwargs)

        
        
        self.IMAGE_TOKEN = "<|endoftext10|>"
        self.AUDIO_TOKEN = "<|endoftext11|>"
        self.IM_TOKEN_ID = 200010
        self.AUDIO_TOKEN_ID = 200011
        self.AUDIO_SAMPLE_RATE = 16000

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            audio_token=self.AUDIO_TOKEN,
            audio_token_id=self.AUDIO_TOKEN_ID,
        ).build(self.processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.AUDIO_SAMPLE_RATE,
        )

        if base_output.audios is not None:
            
            
            base_output.audios = [
                (audio, self.AUDIO_SAMPLE_RATE) for audio in base_output.audios
            ]

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "audio_token_id": self.mm_tokens.audio_token_id,
        }
