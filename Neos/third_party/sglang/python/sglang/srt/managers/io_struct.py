# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
The definition of objects transferred between different
processes (TokenizerManager, DetokenizerManager, Scheduler).
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import BaseFinishReason
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData


if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class BaseBatchReq(ABC):
    rids: Optional[List[str]] = field(default=None, kw_only=True)
    http_worker_ipcs: Optional[List[str]] = field(default=None, kw_only=True)

    def regenerate_rids(self):
        """Generate new request IDs and return them."""
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class SpeculativeDecodingMetricsMixin:
    """
    Mixin class containing speculative decoding metrics.

    This class consolidates speculative decoding metrics that are shared across
    batch output types that support speculative decoding to avoid code duplication.
    """

    
    spec_verify_ct: List[int]

    
    spec_accepted_tokens: List[int]

    
    
    
    
    spec_acceptance_histogram: List[List[int]]



@dataclass
class SessionParams:
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None
    drop_previous_output: Optional[bool] = None




ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]

MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]

MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput(BaseReq):
    
    text: Optional[Union[List[str], str]] = None
    
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    
    
    
    
    
    
    image_data: Optional[MultimodalDataInputFormat] = None
    
    video_data: Optional[MultimodalDataInputFormat] = None
    
    audio_data: Optional[MultimodalDataInputFormat] = None
    
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    
    return_logprob: Optional[Union[List[bool], bool]] = None
    
    
    logprob_start_len: Optional[Union[List[int], int]] = None
    
    top_logprobs_num: Optional[Union[List[int], int]] = None
    
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    
    return_text_in_logprobs: bool = False
    
    stream: bool = False
    
    log_metrics: bool = True
    
    return_hidden_states: Union[List[bool], bool] = False
    
    return_routed_experts: bool = False
    
    routed_experts_start_len: int = 0

    
    modalities: Optional[List[str]] = None
    
    session_params: Optional[Union[List[Dict], Dict]] = None

    
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    
    
    
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
    bootstrap_pair_key: Optional[Union[List[str], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None

    
    require_reasoning: bool = False

    
    routed_dp_rank: Optional[int] = None
    
    disagg_prefill_dp_rank: Optional[int] = None
    
    data_parallel_rank: Optional[int] = None

    
    background: bool = False

    
    conversation_id: Optional[str] = None

    
    priority: Optional[int] = None

    
    extra_key: Optional[Union[List[str], str]] = None

    
    routing_key: Optional[str] = None

    
    no_logs: bool = False

    
    custom_labels: Optional[Dict[str, str]] = None

    
    return_bytes: bool = False

    
    return_entropy: bool = False

    
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    
    need_wait_for_image: Optional[bool] = None
    num_items_assigned: Optional[List] = None

    
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):
        """
        Normalize the batch size and arguments for the request.

        This method resolves various input formats and ensures all parameters
        are properly formatted as either single values or batches depending on the input.
        It also handles parallel sampling expansion and sets default values for
        unspecified parameters.

        Raises:
            ValueError: If inputs are not properly specified (e.g., none or all of
                       text, input_ids, input_embeds are provided)
        """
        if self.data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.routed_dp_rank is None:
                self.routed_dp_rank = self.data_parallel_rank
            self.data_parallel_rank = None

        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

    def _determine_batch_size(self):
        """Determine if this is a single example or a batch and the batch size."""
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):
        """Handle parallel sampling parameters and adjust batch size if needed."""
        
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):
        """Normalize inputs for a single example."""
        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:  
            self.token_ids_logprob = None

    def _normalize_batch_inputs(self):
        """Normalize inputs for a batch of examples, including parallel sampling expansion."""
        
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            
            num = self.batch_size * self.parallel_sample_num

        
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):
        """Expand the main inputs (text, input_ids, input_embeds) for parallel sampling."""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):
        """Normalize image data for batch processing."""
        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):
            
            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):
            
            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):
                
                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:
                        
                        self.modalities.append(None)
                
                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:
                
                wrapped_images = [[img] for img in self.image_data]
                
                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_video_data(self, num):
        """Normalize video data for batch processing."""
        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):
        """Normalize audio data for batch processing."""
        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):
        """Normalize sampling parameters for batch processing."""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):
        """Normalize request IDs for batch processing."""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            
            
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):
        """Normalize logprob-related parameters for batch processing."""

        
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param

        
        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )

        
        if not self.token_ids_logprob:  
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_custom_logit_processor(self, num):
        """Normalize custom logit processor for batch processing."""
        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_bootstrap_params(self, num):
        """Normalize bootstrap parameters for batch processing."""
        
        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num

        
        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num

        
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num

        
        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num

    def _validate_session_params(self):
        """Validate that session parameters are properly formatted."""
        if self.session_params is not None:
            if not isinstance(self.session_params, dict) and not isinstance(
                self.session_params[0], dict
            ):
                raise ValueError("Session params must be a dict or a list of dicts.")

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            priority=self.priority,
            extra_key=self.extra_key,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            external_trace_header=self.external_trace_header,
            http_worker_ipc=self.http_worker_ipc,
            received_time=self.received_time,
        )


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    
    input_text: str
    
    input_ids: List[int]
    
    mm_inputs: dict
    
    sampling_params: SamplingParams
    
    return_logprob: bool
    
    logprob_start_len: int
    
    top_logprobs_num: int
    
    token_ids_logprob: List[int]
    
    stream: bool

    
    return_hidden_states: bool = False

    
    return_routed_experts: bool = False
    
    routed_experts_start_len: int = 0

    
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    
    session_params: Optional[SessionParams] = None

    
    lora_id: Optional[str] = None  

    
    
    
    custom_logit_processor: Optional[str] = None

    
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    
    require_reasoning: bool = False

    
    routed_dp_rank: Optional[int] = None
    
    disagg_prefill_dp_rank: Optional[int] = None

    
    priority: Optional[int] = None

    
    extra_key: Optional[str] = None

    
    routing_key: Optional[str] = None

    
    no_logs: bool = False

    
    return_bytes: bool = False

    
    return_entropy: bool = False

    need_wait_for_image: bool = False
    num_items_assigned: Optional[List] = None

    
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput(BaseReq):
    
    text: Optional[Union[List[List[str]], List[str], str]] = None
    
    
    
    
    
    
    image_data: Optional[MultimodalDataInputFormat] = None
    
    video_data: Optional[MultimodalDataInputFormat] = None
    
    audio_data: Optional[MultimodalDataInputFormat] = None
    
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    
    log_metrics: bool = True
    
    modalities: Optional[List[str]] = None
    
    is_cross_encoder_request: bool = False
    
    priority: Optional[int] = None
    
    routing_key: Optional[str] = None

    
    background: bool = False

    
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    
    dimensions: Optional[int] = None

    
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    def normalize_batch_and_arguments(self):
        
        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )

        
        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")

        
        self.batch_size = 0
        self.is_single = True

        
        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
                self.is_single = False
            else:
                self.batch_size += 1

        
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1

        
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

            self._normalize_lora_paths(self.batch_size)

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def __getitem__(self, i):
        if self.is_cross_encoder_request:
            return EmbeddingReqInput(
                text=[self.text[i]] if self.text is not None else None,
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                is_cross_encoder_request=True,
                http_worker_ipc=self.http_worker_ipc,
            )

        return EmbeddingReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i] if self.image_data is not None else None,
            audio_data=self.audio_data[i] if self.audio_data is not None else None,
            video_data=self.video_data[i] if self.video_data is not None else None,
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            external_trace_header=self.external_trace_header,
            dimensions=self.dimensions,
            http_worker_ipc=self.http_worker_ipc,
            received_time=self.received_time,
        )


@dataclass
class TokenizedEmbeddingReqInput(BaseReq):
    
    input_text: str
    
    input_ids: List[int]
    
    image_inputs: dict
    
    token_type_ids: List[int]
    
    sampling_params: SamplingParams
    
    routed_dp_rank: Optional[int] = None
    
    priority: Optional[int] = None
    
    dimensions: Optional[int] = None
    
    lora_id: Optional[str] = None  
    
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedEmbeddingReqInput(BaseBatchReq):
    
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class BatchTokenIDOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    
    finished_reasons: List[BaseFinishReason]
    
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    
    output_ids: Optional[List[int]]
    
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]
    output_token_entropy_val: List[float]

    
    output_hidden_states: List[List[float]]

    
    
    routed_experts: List[Optional[torch.Tensor]]

    
    
    
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    
    retraction_counts: List[int]

    
    token_steps: List[List[int]] = None

    
    load: GetLoadReqOutput = None
    
    customized_info: Optional[Dict[str, List[Any]]] = None
    
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    
    dp_ranks: Optional[List[int]] = None

    
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchMultimodalDecodeReq(BaseBatchReq):
    decoded_ids: List[int]
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    read_offsets: List[int]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    image_resolutions: List[List[int]]
    resize_image_resolutions: List[List[int]]

    finished_reasons: List[BaseFinishReason]

    
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    
    
    
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    return_bytes: List[bool]

    
    token_steps: List[List[int]] = None


@dataclass
class BatchStrOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    
    finished_reasons: List[dict]
    
    output_strs: List[str]
    
    output_ids: Optional[List[int]]

    
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]
    output_token_entropy_val: List[float]

    
    output_hidden_states: List[List[float]]

    
    
    routed_experts: List[Optional[torch.Tensor]]

    
    
    
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    
    retraction_counts: List[int]

    
    token_steps: List[List[int]] = None

    
    load: GetLoadReqOutput = None

    
    customized_info: Optional[Dict[str, List[Any]]] = None
    
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    
    dp_ranks: Optional[List[int]] = None

    
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchMultimodalOutput(BaseBatchReq):
    
    finished_reasons: List[dict]
    decoded_ids: List[List[int]]
    
    outputs: Union[List[str | bytes], List[List[Dict]]]

    
    input_token_logprobs_val: List[List[float]]
    input_token_logprobs_idx: List[List[int]]
    output_token_logprobs_val: List[List[float]]
    output_token_logprobs_idx: List[List[int]]

    
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    return_bytes: List[bool]
    
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None

    
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchEmbeddingOutput(BaseBatchReq):
    
    finished_reasons: List[BaseFinishReason]
    
    embeddings: Union[List[List[float]], List[Dict[int, float]]]
    
    prompt_tokens: List[int]
    cached_tokens: List[int]
    
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    
    retraction_counts: List[int]
    
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None

    
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class ClearHiCacheReqInput(BaseReq):
    pass


@dataclass
class ClearHiCacheReqOutput(BaseReq):
    success: bool


@dataclass
class FlushCacheReqInput(BaseReq):
    pass


@dataclass
class FlushCacheReqOutput(BaseReq):
    success: bool


@dataclass
class AttachHiCacheStorageReqInput(BaseReq):
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)
    """

    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is None:
            pass
        else:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )

        if self.hicache_write_policy is None:
            return
        allowed = ["write_back", "write_through", "write_through_selective"]
        if self.hicache_write_policy not in allowed:
            raise ValueError(
                f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                f"Expected one of {allowed}."
            )


@dataclass
class AttachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class DetachHiCacheStorageReqInput(BaseReq):
    """Dynamically detach (disable) HiCache storage backend at runtime."""

    pass


@dataclass
class DetachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class PinPrefixReqInput(BaseReq):
    """Pin a prefix by token_ids to resist eviction."""

    token_ids: List[int] = field(default_factory=list)
    ttl_seconds: int = 300  


@dataclass
class PinPrefixReqOutput(BaseReq):
    success: bool
    nodes_pinned: int = 0
    message: str = ""


@dataclass
class PauseGenerationReqInput(BaseReq):
    """
    Note that the PauseGenerationRequests is only supported in SGLang Server.
    abort: Abort and return all requests currently being processed.

    in_place: Pause the scheduler's event_loop from performing inference;
            only non-inference requests (e.g., control commands) will be handled.
            The requests in the engine will be paused and stay in the event_loop,
            then continue generation after continue_generation with the old kv cache.
            Note: In 'inplace' mode, flush_cache will fail if there are any requests
            in the running_batch.

    retract: Pause the scheduler's event loop from performing inference;
            only non-inference requests will be handled, and all currently running
            requests will be retracted back to the waiting_queue.
            Note: The KV cache can be flushed in this mode and will be automatically
            recomputed after continue_generation.
    """

    mode: Literal["abort", "retract", "in_place"] = "abort"

    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(
                f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}."
            )


@dataclass
class ContinueGenerationReqInput(BaseReq):
    pass


@dataclass
class UpdateWeightFromDiskReqInput(BaseReq):
    
    model_path: str
    
    load_format: Optional[str] = None
    
    abort_all_requests: bool = False
    
    weight_version: Optional[str] = None
    
    is_async: bool = False
    
    torch_empty_cache: bool = False
    
    keep_pause: bool = False
    
    recapture_cuda_graph: bool = False
    
    token_step: int = 0
    
    flush_cache: bool = True
    
    manifest: Optional[Dict[str, Any]] = None


@dataclass
class UpdateWeightFromDiskReqOutput(BaseReq):
    success: bool
    message: str
    
    num_paused_requests: Optional[int] = 0


@dataclass
class UpdateWeightsFromDistributedReqInput(BaseReq):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    
    group_name: str = "weight_update_group"
    
    flush_cache: bool = True
    
    abort_all_requests: bool = False
    
    weight_version: Optional[str] = None
    
    load_format: Optional[str] = None


@dataclass
class UpdateWeightsFromDistributedReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput(BaseReq):
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[Union[str, bytes]]
    
    load_format: Optional[str] = None
    
    flush_cache: bool = True
    
    abort_all_requests: bool = False
    
    weight_version: Optional[str] = None
    
    disable_draft_model: Optional[bool] = None


@dataclass
class UpdateWeightsFromTensorReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq):
    
    master_address: str
    
    ports: str
    
    group_rank: int
    
    world_size: int
    
    group_name: str = "weight_send_group"
    
    backend: str = "nccl"




@dataclass
class UpdateWeightsFromIPCReqInput(BaseReq):
    
    zmq_handles: Dict[str, str]
    
    flush_cache: bool = True
    
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class SendWeightsToRemoteInstanceReqInput(BaseReq):
    
    master_address: str
    
    ports: str
    
    group_name: str = "weight_send_group"


@dataclass
class SendWeightsToRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateExpertBackupReq(BaseReq):
    pass


@dataclass
class BackupDramReq(BaseReq):
    rank: int
    weight_pointer_map: Dict[str, Any]
    session_id: str
    buffer_size: int


@dataclass
class InitWeightsUpdateGroupReqInput(BaseReq):
    
    master_address: str
    
    master_port: int
    
    rank_offset: int
    
    world_size: int
    
    group_name: str = "weight_update_group"
    
    backend: str = "nccl"


@dataclass
class InitWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class DestroyWeightsUpdateGroupReqInput(BaseReq):
    group_name: str = "weight_update_group"


@dataclass
class DestroyWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightVersionReqInput(BaseReq):
    
    new_version: str
    
    abort_all_requests: bool = True


@dataclass
class GetWeightsByNameReqInput(BaseReq):
    name: str
    truncate_size: int = 100


@dataclass
class GetWeightsByNameReqOutput(BaseReq):
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput(BaseReq):
    
    
    tags: Optional[List[str]] = None


@dataclass
class ReleaseMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class ResumeMemoryOccupationReqInput(BaseReq):
    
    
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class CheckWeightsReqInput(BaseReq):
    action: str


@dataclass
class CheckWeightsReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class SlowDownReqInput(BaseReq):
    forward_sleep_time: Optional[float]


@dataclass
class SlowDownReqOutput(BaseReq):
    pass


@dataclass
class AbortReq(BaseReq):
    
    abort_all: bool = False
    
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None

    def __post_init__(self):
        
        if self.rid is None:
            self.rid = ""


@dataclass
class ActiveRanksOutput(BaseReq):
    status: List[bool]


@dataclass
class GetInternalStateReq(BaseReq):
    pass


@dataclass
class GetInternalStateReqOutput(BaseReq):
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq(BaseReq):
    server_args: Dict[str, Any]


@dataclass
class SetInternalStateReqOutput(BaseReq):
    updated: bool
    server_args: Dict[str, Any]


@dataclass
class ProfileReqInput(BaseReq):
    
    output_dir: Optional[str] = None
    
    start_step: Optional[int] = None
    
    
    
    num_steps: Optional[int] = None
    
    activities: Optional[List[str]] = None
    
    profile_by_stage: bool = False
    
    with_stack: Optional[bool] = None
    
    record_shapes: Optional[bool] = None
    
    merge_profiles: bool = False
    
    profile_prefix: Optional[str] = None
    
    profile_stages: Optional[List[str]] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq(BaseReq):
    type: ProfileReqType
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    merge_profiles: bool = False
    profile_prefix: Optional[str] = None
    profile_stages: Optional[List[str]] = None


@dataclass
class ProfileReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class FreezeGCReq(BaseReq):
    pass


@dataclass
class ConfigureLoggingReq(BaseReq):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None


@dataclass
class OpenSessionReqInput(BaseReq):
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


@dataclass
class CloseSessionReqInput(BaseReq):
    session_id: str


@dataclass
class OpenSessionReqOutput(BaseReq):
    session_id: Optional[str]
    success: bool


@dataclass
class HealthCheckOutput(BaseReq):
    pass


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReq(BaseReq):
    action: ExpertDistributionReqType


@dataclass
class ExpertDistributionReqOutput(BaseReq):
    pass


@dataclass
class Function:
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[object] = None


@dataclass
class Tool:
    function: Function
    type: Optional[str] = "function"


@dataclass
class ParseFunctionCallReq(BaseReq):
    text: str  
    tools: List[Tool] = field(
        default_factory=list
    )  
    tool_call_parser: Optional[str] = (
        None  
    )


@dataclass
class SeparateReasoningReqInput(BaseReq):
    text: str  
    reasoning_parser: str  


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None


@dataclass
class RpcReqInput(BaseReq):
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class LoadLoRAAdapterReqInput(BaseReq):
    
    lora_name: str
    
    lora_path: str
    
    pinned: bool = False
    
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


@dataclass
class UnloadLoRAAdapterReqInput(BaseReq):
    
    lora_name: str
    
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


@dataclass
class LoadLoRAAdapterFromTensorsReqInput(BaseReq):
    lora_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


@dataclass
class LoRAUpdateOutput(BaseReq):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


@dataclass
class BlockReqInput(BaseReq):
    type: BlockReqType


@dataclass
class GetLoadReqInput(BaseReq):
    pass


@dataclass
class GetLoadReqOutput(BaseReq):
    dp_rank: int
    num_reqs: int
    num_waiting_reqs: int
    num_tokens: int
    ts_tic: float


@dataclass
class MemoryMetrics:
    """Memory breakdown metrics."""

    weight_gb: float = field(
        metadata={"metric": ("gauge", "Model weight memory in GB")}
    )
    kv_cache_gb: float = field(metadata={"metric": ("gauge", "KV cache memory in GB")})
    graph_gb: float = field(metadata={"metric": ("gauge", "CUDA graph memory in GB")})
    token_capacity: int = field(
        metadata={"metric": ("gauge", "Max tokens in KV cache")}
    )


@dataclass
class SpeculativeMetrics:
    """Speculative decoding metrics."""

    accept_length: float = field(
        metadata={"metric": ("gauge", "Avg accepted tokens per step")}
    )
    accept_rate: float = field(
        metadata={"metric": ("gauge", "Speculative acceptance rate")}
    )


@dataclass
class LoRAMetrics:
    """LoRA adapter pool metrics."""

    slots_used: int = field(metadata={"metric": ("gauge", "LoRA adapter slots in use")})
    slots_total: int = field(metadata={"metric": ("gauge", "Total LoRA adapter slots")})
    utilization: float = field(
        metadata={"metric": ("gauge", "LoRA pool utilization ratio")}
    )


@dataclass
class DisaggregationMetrics:
    """PD disaggregation metrics."""

    mode: str  
    prefill_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill prealloc queue requests")}
    )
    prefill_inflight_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill inflight queue requests")}
    )
    decode_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode prealloc queue requests")}
    )
    decode_transfer_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode transfer queue requests")}
    )
    decode_retracted_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode retracted queue requests")}
    )
    kv_transfer_speed_gb_s: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer speed in GB/s")}
    )
    kv_transfer_latency_ms: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer latency in ms")}
    )


@dataclass
class QueueMetrics:
    """Detailed queue breakdown."""

    waiting: int = field(metadata={"metric": ("gauge", "Main waiting queue size")})
    grammar: int = field(
        metadata={"metric": ("gauge", "Grammar compilation queue size")}
    )
    paused: int = field(
        metadata={"metric": ("gauge", "Requests paused by weight sync")}
    )
    retracted: int = field(metadata={"metric": ("gauge", "Retracted requests count")})


@dataclass
class GetLoadsReqInput(BaseReq):
    """Request for /v1/loads endpoint."""

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    include: List[str] = field(default_factory=lambda: ["all"])
    dp_rank: Optional[int] = None

    def __post_init__(self):
        """Validate include sections."""
        if self.include:
            invalid = set(self.include) - self.VALID_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid include sections: {invalid}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )


@dataclass
class GetLoadsReqOutput(BaseReq):
    """Per-DP-rank load metrics for /v1/loads endpoint."""

    dp_rank: int
    timestamp: float

    num_running_reqs: int = field(
        metadata={"metric": ("gauge", "Number of running requests")}
    )
    num_waiting_reqs: int = field(
        metadata={"metric": ("gauge", "Number of waiting requests")}
    )
    num_used_tokens: int = field(
        metadata={"metric": ("gauge", "Number of tokens in use")}
    )
    max_total_num_tokens: int = field(
        metadata={"metric": ("gauge", "Maximum token capacity")}
    )
    token_usage: float = field(metadata={"metric": ("gauge", "Token pool usage ratio")})
    gen_throughput: float = field(
        metadata={"metric": ("gauge", "Generation throughput tokens/sec")}
    )
    cache_hit_rate: float = field(
        metadata={"metric": ("gauge", "Prefix cache hit rate")}
    )
    utilization: float = field(
        metadata={"metric": ("gauge", "Overall utilization ratio")}
    )
    max_running_requests: int = field(
        metadata={"metric": ("gauge", "Maximum running requests capacity")}
    )

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None


@dataclass
class WatchLoadUpdateReq(BaseReq):
    loads: List[GetLoadReqOutput]


@dataclass
class SetInjectDumpMetadataReqInput(BaseReq):
    dump_metadata: Dict[str, Any]


@dataclass
class SetInjectDumpMetadataReqOutput(BaseReq):
    success: bool


@dataclass
class LazyDumpTensorsReqInput(BaseReq):
    pass


@dataclass
class LazyDumpTensorsReqOutput(BaseReq):
    success: bool


@dataclass
class DumperControlReqInput(BaseReq):
    method: str
    body: Dict[str, Any]


@dataclass
class DumperControlReqOutput(BaseReq):
    success: bool
    response: List[Dict[str, Any]]
    error: str = ""


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        
        name = class_type[0]
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = issubclass(class_type[1], BaseReq) or issubclass(
            class_type[1], BaseBatchReq
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if is_base_req and not is_io_struct:
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()
