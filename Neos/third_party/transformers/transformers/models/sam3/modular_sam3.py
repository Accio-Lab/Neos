# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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


import torch

from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from ..sam2.image_processing_sam2 import Sam2ImageProcessor


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


class Sam3ImageProcessor(Sam2ImageProcessor):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1008, "width": 1008}
    mask_size = {"height": 288, "width": 288}

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple] | None = None, threshold: float = 0.5
    ):
        """
        Converts the output of [`Sam3Model`] into semantic segmentation maps.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry is a binary mask (0 or 1).
        """
        
        
        semantic_logits = outputs.semantic_seg

        if semantic_logits is None:
            raise ValueError(
                "Semantic segmentation output is not available in the model outputs. "
                "Make sure the model was run with semantic segmentation enabled."
            )

        
        semantic_probs = semantic_logits.sigmoid()

        
        if target_sizes is not None:
            if len(semantic_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []

            for idx in range(len(semantic_logits)):
                resized_probs = torch.nn.functional.interpolate(
                    semantic_probs[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                
                semantic_map = (resized_probs[0, 0] > threshold).to(torch.long)
                semantic_segmentation.append(semantic_map)
        else:
            
            semantic_segmentation = (semantic_probs[:, 0] > threshold).to(torch.long)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_object_detection(self, outputs, threshold: float = 0.3, target_sizes: list[tuple] | None = None):
        """
        Converts the raw output of [`Sam3Model`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, and optionally presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep object detection predictions.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted box on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
        """
        pred_logits = outputs.pred_logits  
        pred_boxes = outputs.pred_boxes  
        presence_logits = outputs.presence_logits  

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  
            batch_scores = batch_scores * presence_scores  

        
        batch_boxes = pred_boxes

        
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "boxes": boxes})

        return results

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: list[tuple] | None = None,
    ):
        """
        Converts the raw output of [`Sam3Model`] into instance segmentation predictions with bounding boxes and masks.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, pred_masks, and optionally
                presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep instance predictions.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the predicted masks.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted instance on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
                - **masks** (`torch.Tensor`): Binary segmentation masks for each instance, shape (num_instances,
                  height, width).
        """
        pred_logits = outputs.pred_logits  
        pred_boxes = outputs.pred_boxes  
        pred_masks = outputs.pred_masks  
        presence_logits = outputs.presence_logits  

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  
            batch_scores = batch_scores * presence_scores  

        
        batch_masks = pred_masks.sigmoid()

        
        batch_boxes = pred_boxes

        
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks)):
            
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]  

            
            if target_sizes is not None:
                target_size = target_sizes[idx]
                if len(masks) > 0:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0),  
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  

            
            masks = (masks > mask_threshold).to(torch.long)

            results.append({"scores": scores, "boxes": boxes, "masks": masks})

        return results


__all__ = ["Sam3ImageProcessor"]
