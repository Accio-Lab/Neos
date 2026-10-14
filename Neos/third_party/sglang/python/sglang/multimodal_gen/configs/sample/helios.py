# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HeliosT2VSamplingParams(SamplingParams):
    
    height: int = 384
    width: int = 640
    num_frames: int = 99
    fps: int = 24

    
    guidance_scale: float = 5.0
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    )
    num_inference_steps: int = 50

    
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (640, 384),  
            (384, 640),  
            (832, 480),  
            (480, 832),  
        ]
    )


@dataclass
class HeliosMidSamplingParams(HeliosT2VSamplingParams):
    """Sampling params for Helios-Mid (Stage 2 pyramid SR)."""

    num_inference_steps: int = 20


@dataclass
class HeliosDistilledSamplingParams(HeliosT2VSamplingParams):
    """Sampling params for Helios-Distilled (DMD, no CFG needed)."""

    guidance_scale: float = 1.0
    num_inference_steps: int = 10
