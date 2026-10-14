# Copyright 2025 The HuggingFace Inc. team
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
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum

import torch

from ...utils import is_psutil_available, is_torch_xpu_available
from ...utils.logging import logging


if is_psutil_available():
    import psutil



TMP_TOKEN_ID = -1



logger = logging.getLogger("ContinuousBatchingLogger")

if logger.propagate:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = total_memory - free_memory
    elif is_torch_xpu_available():
        device = torch.device("xpu")
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
        total_memory = torch.xpu.get_device_properties(device).total_memory
        reserved_memory = torch.xpu.memory_reserved(device)
        allocated_memory = torch.xpu.memory_allocated(device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        total_memory = torch.mps.recommended_max_memory()
        allocated_memory = torch.mps.current_allocated_memory()
        reserved_memory = torch.mps.driver_allocated_memory()
    else:
        device = torch.device("cpu")
        if is_psutil_available():
            total_memory = psutil.virtual_memory().total
            allocated_memory = psutil.Process().memory_info().rss
            reserved_memory = allocated_memory
        else:
            logger.error(
                "Cannot get memory breakdown on CPU without psutil: returning 0 for all memory values. Please install "
                "psutil to get an actual memory breakdown."
            )
            total_memory = 0
            reserved_memory = 0
            allocated_memory = 0

    return device, total_memory, reserved_memory, allocated_memory


class RequestStatus(IntEnum):
    """Status of a generation request through its lifecycle."""

    PENDING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3
    FAILED = 4


@dataclass
class GenerationOutput:
    """Tracks the output of a generation request.

    Attributes:
        request_id (str): The ID of the generation request.
        prompt_ids (list[int]): The IDs of the prompt tokens.
        generated_tokens (list[int]): The generated tokens.
        logprobs (list[float]): The log probabilities of the generated tokens.
        error (Optional[str]): Any error message associated with the request. When None, the request was successful.
        status (RequestStatus): The status of the request.
        created_time (float): The time the request was created.
        lifespan (tuple[float, float]): The time the request was no longer pending and the time the request finished.
    """

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    error: str | None = None
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.perf_counter)
    lifespan: tuple[float, float] = (-1, -1)  
    timestamps: list[float] | None = None  

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        request_id (str): The ID of the generation request.
        initial_tokens (list[int]): The initial prompt tokens.
        num_children (int): The number of children requests
        full_prompt_ids (list[int] | None): The tokens IDs of the full prompt.
        prompt_ids (list[int] | None): The tokens IDs currently being processed.
        remaining_prompt_ids (list[int]): The initial tokens IDs remaining to be processed.
        static_outputs (list[int]): The generated tokens.
        allocated_blocks (int): The number of blocks allocated to the request.
        position_offset (int): The current position in the sequence for position_ids.
        status (RequestStatus): The status of the request: can be one of PENDING, PREFILLING, PREFILLING_SPLIT,
                                SPLIT_PENDING_REMAINDER, DECODING, FINISHED, FAILED
        max_new_tokens (int | None): The maximum number of new tokens to generate.
        eos_token_id (None | int | list[int]): The ID(s) of the end-of-sequence tokens. Only used in post-init.
        _eos_token_ids (set[int]): The IDs of the end-of-sequence tokens, formatted as a set.
        streaming (bool): Whether to stream tokens as they're generated
        created_time (float): The time the request was created.
        error (Optional[str]): Any error message associated with the request. When None, has had no error yet.
    """

    
    request_id: str
    initial_tokens: list[int]  

    
    streaming: bool = False  
    record_timestamps: bool = False  

    
    max_new_tokens: int | None = 20  
    eos_token_id: int | list[int] | None = None  
    num_children: int = 0  
    logit_processor_kwargs: dict = field(default_factory=dict)  

    
    tokens_to_process: list[int] = field(default_factory=list)  
    generated_tokens: list[int] = field(default_factory=list)  
    logprobs: list[float] = field(default_factory=list)  
    position_offset: int = 0  
    allocated_blocks: int = 0  

    _status: RequestStatus = RequestStatus.PENDING  
    _eos_token_ids: set[int] = field(default_factory=set)  

    
    created_time: float = field(default_factory=time.perf_counter)  
    error: str | None = None  
    lifespan: tuple[float, float] = (-1, -1)  
    _timestamps: list[float] = field(default_factory=list)  
    _true_initial_tokens: int = 0  
    

    
    _new_tokens_limit: int = 2147483647  
    remaining_prefill_tokens: list[int] = field(default_factory=list)  
    is_cpu_offloaded: bool = False  

    def __post_init__(self):
        
        self._new_tokens_limit = 2147483647 if self.max_new_tokens is None else self.max_new_tokens
        
        self.remaining_prefill_tokens = self.initial_tokens[:]
        
        if self.eos_token_id is None:
            pass
        
        elif isinstance(self.eos_token_id, int):
            if self.eos_token_id >= 0:
                self._eos_token_ids.add(self.eos_token_id)
        
        else:
            for token_id in self.eos_token_id:
                if token_id >= 0:
                    self._eos_token_ids.add(token_id)

    @property
    def status(self) -> RequestStatus:
        return self._status

    @status.setter
    def status(self, value: RequestStatus):
        if self._status == RequestStatus.PENDING:
            self.lifespan = (time.perf_counter(), -1)
        elif value == RequestStatus.FINISHED:
            self.lifespan = (self.lifespan[0], time.perf_counter())
            self.log_end_of_request()
        self._status = value

    @property
    def timestamps(self) -> list[float] | None:
        return self._timestamps if self.record_timestamps else None

    def log_end_of_request(self):
        prefill_len = len(self.initial_tokens)
        decode_len = self.generated_len()
        start_time = self.lifespan[0] - self.created_time
        end_time = self.lifespan[1] - self.created_time
        logger.info(
            f"Request {self.request_id} finished: {prefill_len = } {decode_len = } {start_time = } {end_time = }"
        )

    def current_len(self) -> int:
        """Get the current length of the sequence (prompt + generated tokens)."""
        return self.position_offset

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.generated_tokens)

    
    def update_and_check_completion(self, token_id: int, logprob: float | None) -> bool:
        """Update the request with a newly generated token (and optional log probability of the token) and check for
        completion. Returns True if the request is now complete, False otherwise."""
        
        if self.status != RequestStatus.DECODING:
            return False

        
        if self.record_timestamps:
            self._timestamps.append(time.perf_counter())

        
        is_eos = token_id in self._eos_token_ids
        current_len = self.generated_len()

        
        
        if is_eos or (current_len < self._new_tokens_limit):
            self.generated_tokens.append(token_id)
            self.tokens_to_process = [token_id]  
            current_len += 1
            if logprob is not None:
                self.logprobs.append(logprob)
        else:
            logger.warning(f"Request {self.request_id} generated a useless token: {token_id}")

        if is_eos or current_len >= self._new_tokens_limit:
            self.status = RequestStatus.FINISHED
            return True
        return False  

    def __repr__(self):
        msg = [
            f"request_id={self.request_id}",
            f"status={self._status}",
            f"out_tokens={self.generated_len()}",
            f"query_length={len(self.tokens_to_process)}",
            f"remaining_tokens={len(self.remaining_prefill_tokens)}",
            f"kv_length={self.position_offset}",
            f"full_prompt_length={len(self.initial_tokens)}",
            f"allocated_blocks={self.allocated_blocks}",
            f"generated_tokens={self.generated_tokens}",
            f"logit_processor_kwargs={self.logit_processor_kwargs}",
        ]
        return "RequestState(\n\t" + ",\n\t".join(msg) + "\n)"

    def to_generation_output(self):
        """Convert the request state to a GenerationOutput object."""
        if self._true_initial_tokens:
            self.generated_tokens = self.initial_tokens[self._true_initial_tokens :] + self.generated_tokens
            self.initial_tokens = self.initial_tokens[: self._true_initial_tokens]
        return GenerationOutput(
            request_id=self.request_id,
            prompt_ids=self.initial_tokens,
            generated_tokens=self.generated_tokens,
            logprobs=self.logprobs,
            error=self.error,
            status=self.status,
            created_time=self.created_time,
            lifespan=self.lifespan,
            timestamps=self.timestamps,
        )

    def fork(self, new_request_id: str) -> "RequestState":
        """Fork the request into a new request with the same state except for request_id, created_time and lifespan."""
        new_request = deepcopy(self)
        
        new_request.request_id = new_request_id
        new_request.created_time = time.perf_counter()
        new_request.lifespan = (new_request.created_time, -1)
        new_request._timestamps = []
        
        new_request.remaining_prefill_tokens = self.remaining_prefill_tokens[:]
        return new_request

    def get_request_config(self) -> dict:
        """Get all the fields necessary to create a request that would have the same configuration."""
        return {
            "streaming": self.streaming,
            "record_timestamps": self.record_timestamps,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "num_children": self.num_children,
            "logit_processor_kwargs": deepcopy(self.logit_processor_kwargs),
        }

    def create_equivalent_initial_request(self) -> "RequestState":
        """Creates an equivalent new request by removing the generated tokens and adding them to the initial prompt. The
        created request has THE SAME request_id. Notably, we can retrieve the original request from the created one with
        the _true_initial_tokens attribute. The logprobs of the generated tokens are kept in the new request."""

        request_config = self.get_request_config()
        
        if self.max_new_tokens is not None:
            request_config["max_new_tokens"] = self.max_new_tokens - len(self.generated_tokens)
        
        new_state = RequestState(
            request_id=self.request_id,
            initial_tokens=self.initial_tokens + self.generated_tokens,
            logprobs=self.logprobs[:],
            _true_initial_tokens=self._true_initial_tokens + len(self.initial_tokens),
            **request_config,
        )
        
        if self._true_initial_tokens:
            new_state._true_initial_tokens = self._true_initial_tokens
        
        else:
            new_state._true_initial_tokens = len(self.initial_tokens)
        return new_state


class FutureRequestState:
    """Tracks the current state of a request and the relevant information to update it."""

    
    __slots__ = ("state", "has_new_token", "complete_blocks", "query_length")

    def __init__(self, state: RequestState, has_new_token: bool, complete_blocks: int, query_length: int) -> None:
        self.state = state
        self.has_new_token = has_new_token
        self.complete_blocks = complete_blocks
        self.query_length = query_length
