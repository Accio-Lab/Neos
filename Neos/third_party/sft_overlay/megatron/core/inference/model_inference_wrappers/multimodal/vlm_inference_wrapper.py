# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from typing import Any, Dict, Optional

import torch

from megatron.core.inference.communication_utils import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)



class VLMInferenceWrapper(GPTInferenceWrapper):
    """Inference wrapper for VLMs"""

    def prep_model_for_inference(self, prompts_tokens: Optional[torch.Tensor] = None):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop.
        It puts the model in eval mode.

        Args:
            prompts_tokens (torch.Tensor): Deprecated, will be removed in `megatron-core` 0.13
        """
        if prompts_tokens is not None:
            warnings.warn(
                "Passing `prompts_tokens` is deprecated and this argument will be ignored."
                "This parameter will be removed in `megatron-core` 0.13."
            )

        super().prep_model_for_inference()

        
        
        self.model_is_pipeline_parallel = not (
            is_pipeline_first_stage(self.pp_group) and is_pipeline_last_stage(self.pp_group)
        )

        self._recv_only_vision_embeds = False
        pp_rank = self.pp_group.rank()
        
        
        
        if pp_rank > 0:
            self._recv_only_vision_embeds = False  

        
        self._encoder_only = False  

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        num_img_embeddings_per_tile: int,
        images: torch.Tensor,
        num_tiles: torch.Tensor,
        decoder_seq_length: int,
    ):
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
            num_img_embeddings_per_tile (int): The number of image embeddings per tile
            images (torch.Tensor): The image embeddings
            num_tiles (torch.Tensor): The number of tiles for each input image
            decoder_seq_length (int): The decoder sequence length
        """
        inference_input = super().prep_inference_input(prompts_tokens)

        total_num_tiles = torch.sum(num_tiles).item()
        num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles

        batch_size, max_sequence_length = prompts_tokens.shape
        self.inference_context = StaticInferenceContext(
            batch_size, max_sequence_length + num_img_embeddings
        )

        inference_input["images"] = images
        inference_input["num_tiles"] = num_tiles
        inference_input["num_img_embeddings"] = num_img_embeddings
        inference_input["decoder_seq_length"] = decoder_seq_length

        return inference_input

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length.

        Returns:
            Dict[str, Any]: A dict of inputs that will be used by your model in the forward step
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        images = inference_input["images"]
        num_tiles = inference_input["num_tiles"]
        num_img_embeddings = inference_input["num_img_embeddings"]
        decoder_seq_length = inference_input["decoder_seq_length"]

        tokens2use = tokens[:, context_start_position:context_end_position]
        positions2use = position_ids[:, context_start_position:context_end_position]

        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "images": images,
            "num_tiles": num_tiles,
            "num_img_embeddings": num_img_embeddings,
            "decoder_seq_length": decoder_seq_length,
        }

    def _forward(self, inference_input: Dict[str, Any]):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        images = inference_input["images"]
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        num_image_tiles = inference_input["num_tiles"]

        output = self.model(
            images,
            tokens,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=self.inference_context,
            num_image_tiles=num_image_tiles,
            runtime_gather_output=True,
        )
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        return logits

    def run_one_forward_step(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """The forward pass of the model for inference

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the VLM model

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size].
            The logits are returned only in the last pipeline stage for PP models.
        """
        tokens = inference_input["tokens"]
        num_image_tokens = (tokens == self.model.module.image_token_index).sum().item()
        num_img_embeddings = inference_input["num_img_embeddings"]
        decoder_seq_length = inference_input["decoder_seq_length"]
        num_tokens = tokens.size(1)
        recv_buffer_seq_len = None
        if num_image_tokens > 0:
            
            
            
            
            
            
            if self._recv_only_vision_embeds:
                recv_buffer_seq_len = num_img_embeddings
            else:
                recv_buffer_seq_len = min(
                    num_img_embeddings + num_tokens - num_image_tokens, decoder_seq_length
                )
        elif self._recv_only_vision_embeds:
            
            
            recv_buffer_seq_len = 0

        
        
        if not (self._encoder_only and num_image_tokens == 0):
            output = super().run_one_forward_step(
                inference_input, recv_buffer_seq_len=recv_buffer_seq_len
            )
        else:
            output = None
        logits = output

        
        
        
        if num_tokens > 1 and num_image_tokens > 0:
            if "image_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["image_tokens_count"] = (
                    num_img_embeddings
                )

            if num_img_embeddings + num_tokens - num_image_tokens > decoder_seq_length:
                self.inference_context.sequence_len_offset += decoder_seq_length - num_tokens
            else:
                self.inference_context.sequence_len_offset += (
                    self.inference_context.key_value_memory_dict["image_tokens_count"]
                    - num_image_tokens
                )

        return logits
