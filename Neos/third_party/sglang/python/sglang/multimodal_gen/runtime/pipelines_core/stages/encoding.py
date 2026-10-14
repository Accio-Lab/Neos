# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Encoding stage for diffusion pipelines.
"""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,  
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class EncodingStage(PipelineStage):
    """
    Stage for encoding pixel space representations into latent space.

    This stage handles the encoding of pixel-space video/images into latent
    representations for further processing in the diffusion pipeline.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae

    @torch.no_grad()
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        # Input video/images for VAE encoding: [batch_size, channels, frames, height, width]
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel space representations into latent space.



        Returns:
            The batch with encoded latents.
        """
        assert batch.latents is not None and isinstance(batch.latents, torch.Tensor)

        self.vae = self.vae.to(get_local_torch_device())

        
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        
        latents = (batch.latents * 2.0 - 1.0).clamp(-1, 1)

        
        latents = latents.to(get_local_torch_device())

        
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            
            
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            latents = self.vae.encode(latents).mean

        
        batch.latents = latents

        
        self.maybe_free_model_hooks()

        if server_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch
