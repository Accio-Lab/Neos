# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""CLIP model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openai/clip-vit-base-patch32")
@strict
class CLIPTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import CLIPTextConfig, CLIPTextModel

    >>> # Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPTextConfig()

    >>> # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int | None = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float | None = 1e-5
    attention_dropout: int | float | None = 0.0
    initializer_range: float = 0.02
    initializer_factor: float | None = 1.0

    
    
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


@auto_docstring(checkpoint="openai/clip-vit-base-patch32")
@strict
class CLIPVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] | None = 224
    patch_size: int | list[int] | tuple[int, int] | None = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: int | float | None = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


@auto_docstring(checkpoint="openai/clip-vit-base-patch32")
@strict
class CLIPConfig(PreTrainedConfig):
    r"""
    text_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`CLIPTextConfig`].
    vision_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
    logit_scale_init_value (`float | int`, *optional*, defaults to 2.6592):
        The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.

    Example:

    ```python
    >>> from transformers import CLIPConfig, CLIPModel

    >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPConfig()

    >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
    >>> from transformers import CLIPTextConfig, CLIPVisionConfig

    >>> # Initializing a CLIPText and CLIPVision configuration
    >>> config_text = CLIPTextConfig()
    >>> config_vision = CLIPVisionConfig()

    >>> config = CLIPConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "clip"
    sub_configs = {"text_config": CLIPTextConfig, "vision_config": CLIPVisionConfig}

    text_config: dict | CLIPTextConfig | None = None
    vision_config: dict | CLIPVisionConfig | None = None
    projection_dim: int | None = 512
    logit_scale_init_value: float | int | None = 2.6592
    initializer_factor: float | None = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `CLIPTextConfig` with default values.")
        elif isinstance(self.text_config, CLIPTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `CLIPVisionConfig` with default values.")
        elif isinstance(self.vision_config, CLIPVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        
        
        
        
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            
            _text_config_dict = CLIPTextConfig(**text_config_dict).to_dict()

            
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            
            _vision_config_dict = CLIPVisionConfig(**vision_config_dict).to_dict()
            
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key != "transformers_version":
                    
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            
            vision_config.update(_vision_config_dict)

        
        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["CLIPConfig", "CLIPTextConfig", "CLIPVisionConfig"]
