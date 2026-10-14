import logging
from typing import Any, Dict, Optional

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio

logger = logging.getLogger(__name__)




ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}


LANG_NAME_TO_CODE = {
    name.lower(): code for code, name in ISO639_1_SUPPORTED_LANGS.items()
}


def normalize_language_to_code(language: Optional[str]) -> Optional[str]:
    """Convert a language input (full name or code) to ISO 639-1 code.

    Args:
        language: Language as full name (e.g., 'English', 'Spanish') or
                  ISO 639-1 code (e.g., 'en', 'es')

    Returns:
        ISO 639-1 code or None if input is None
    """
    if language is None:
        return None

    language_lower = language.lower().strip()

    
    if language_lower in ISO639_1_SUPPORTED_LANGS:
        return language_lower

    
    if language_lower in LANG_NAME_TO_CODE:
        return LANG_NAME_TO_CODE[language_lower]

    
    raise ValueError(
        f"Language '{language}' not recognized. "
        f"Use full name (e.g., 'English') or ISO 639-1 code (e.g., 'en')."
    )


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _extract_language_from_request(self, request_obj) -> Optional[str]:
        sampling_params = getattr(request_obj, "sampling_params", None) or {}
        language = sampling_params.pop("language", None)
        return normalize_language_to_code(language)

    def _get_language_token_id(self, language: Optional[str]) -> int:
        
        if language is None:
            language = "en"  
        language_token = f"<|{language}|>"
        return self._tokenizer.convert_tokens_to_ids(language_token)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None

        if len(audio_data) != 1:
            raise ValueError(
                f"Whisper expects exactly 1 audio input, got {len(audio_data)}"
            )

        audios = [load_audio(audio) for audio in audio_data]

        
        
        
        

        
        language = self._extract_language_from_request(request_obj)
        language_token_id = self._get_language_token_id(language)

        
        
        decoder_start_token_id = getattr(
            self.hf_config, "decoder_start_token_id", 50258
        )
        transcribe_token_id = self._tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self._tokenizer.convert_tokens_to_ids(
            "<|notimestamps|>"
        )

        input_ids = [
            decoder_start_token_id,
            language_token_id,
            transcribe_token_id,
            notimestamps_token_id,
        ]

        
        
        input_features = self._processor.feature_extractor(
            audios[0],
            sampling_rate=16000,
            padding="max_length",  
            return_tensors="pt",
        )["input_features"][0]

        return {
            "input_ids": input_ids,
            "mm_items": [
                MultimodalDataItem(
                    feature=input_features,
                    modality=Modality.AUDIO,
                )
            ],
        }
