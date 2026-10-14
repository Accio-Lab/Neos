from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SamplingBatchInfo:
    
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    
    is_all_greedy: bool

    
    need_top_p_sampling: bool

    
    need_top_k_sampling: bool

    
    need_min_p_sampling: bool

    
    vocab_size: int
    grammars: Optional[List] = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    acc_linear_penalties: torch.Tensor = None  

    
    has_custom_logit_processor: bool = False
    
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    
    sampling_seed: Optional[torch.Tensor] = None

    
    device: str = "cuda"

    
    logit_bias: Optional[torch.Tensor] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        global_server_args = get_global_server_args()
        enable_deterministic = global_server_args.enable_deterministic_inference

        reqs = batch.reqs
        device = batch.device
        temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        )
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32, device=device
        )
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float, device=device
        )
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                device=device,
            )
            if enable_deterministic
            else None
        )

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value

        
        has_custom_logit_processor = (
            global_server_args.enable_custom_logit_processor
            and any(r.custom_logit_processor for r in reqs)  
        )  

        if has_custom_logit_processor:
            
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)

            merged_custom_logit_processor = {
                hash(processor_str): (
                    
                    CustomLogitProcessor.from_str(processor_str),
                    
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        
        
        
        
        
        
        
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
            },
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret

    
    def adjusted_from_schedule_batch(self, batch: ScheduleBatch, vocab_size: int):
        pass

    
    def adjusted_merge_batch(self, other: "SamplingBatchInfo"):
        pass

    def __len__(self):
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = (
            first_grammar.apply_vocab_mask
        )  

        
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)

        
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.acc_linear_penalties = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            self.penalizer_orchestrator.apply(self.acc_linear_penalties)
        else:
            self.acc_linear_penalties = None

    def apply_logits_bias(self, logits: torch.Tensor):
        if self.acc_linear_penalties is not None:
            
            logits.add_(self.acc_linear_penalties)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        self.penalizer_orchestrator.filter(keep_indices_device)

        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices_device])

        if self.logit_bias is not None:
            self.logit_bias = self.logit_bias[keep_indices_device]

    def _filter_batch_custom_logit_processor(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        """Filter the custom logit processor and custom params"""
        self.custom_logit_processor = {
            k: (p, mask[keep_indices_device])
            for k, (p, mask) in self.custom_logit_processor.items()
            if torch.any(
                mask[keep_indices_device]
            )  
        }
        self.custom_params = [self.custom_params[i] for i in keep_indices]

        
        
        if len(self.custom_logit_processor) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    @staticmethod
    def merge_custom_logit_processor(
        lhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        rhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        bs1: int,
        bs2: int,
        device: str,
    ):
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            
            left_mask = (
                lhs[k][1]
                if k in lhs
                else torch.zeros(bs1, dtype=torch.bool, device=device)
            )
            right_mask = (
                rhs[k][1]
                if k in rhs
                else torch.zeros(bs2, dtype=torch.bool, device=device)
            )
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

            assert merged_dict[k][1].shape[0] == bs1 + bs2, (
                f"The batch size of merged mask ({merged_dict[k][1].shape[0]}) does not match "
                f"the sum of the batch sizes of the two masks ({bs1 + bs2})"
                f"\n{left_mask=}\n{right_mask=}\n{bs1=}\n{bs2=}"
                f"\n{lhs=}\n{rhs=}"
            )

        return merged_dict

    def merge_batch(self, other: "SamplingBatchInfo"):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            
            self.custom_logit_processor = (
                SamplingBatchInfo.merge_custom_logit_processor(
                    self.custom_logit_processor,
                    other.custom_logit_processor,
                    len(self),
                    len(other),
                    self.device,
                )
            )
            
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            
            self.has_custom_logit_processor = True

        
        
        self.logit_bias = merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device, 0.0
        )

        
        
        
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            if self_val is not None and other_val is not None:
                setattr(self, item, torch.cat([self_val, other_val]))

        self.is_all_greedy &= other.is_all_greedy
        self.need_top_p_sampling |= other.need_top_p_sampling
        self.need_top_k_sampling |= other.need_top_k_sampling
        self.need_min_p_sampling |= other.need_min_p_sampling

        self.adjusted_merge_batch(other)

    def copy_for_forward(self):
        
        self.update_penalties()
        return dataclasses.replace(self, penalizer_orchestrator=None)


def merge_bias_tensor(
    lhs: Optional[torch.Tensor],
    rhs: Optional[torch.Tensor],
    bs1: int,
    bs2: int,
    device: str,
    default: float,
):
    """Merge two bias tensors for batch merging.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        bs1: Batch size of left-hand side tensor
        bs2: Batch size of right-hand side tensor
        device: Device to place the merged tensor on
        default: Default value for missing tensor elements

    Returns:
        Merged tensor or None if both inputs are None
    """
    if lhs is None and rhs is None:
        return None

    if lhs is not None and rhs is not None:
        return torch.cat([lhs, rhs])
    else:
        if lhs is not None:
            shape, dtype = lhs.shape[1:], lhs.dtype
        else:
            shape, dtype = rhs.shape[1:], rhs.dtype

        if lhs is None:
            lhs = torch.empty((bs1, *shape), device=device, dtype=dtype).fill_(default)
        if rhs is None:
            rhs = torch.empty((bs2, *shape), device=device, dtype=dtype).fill_(default)
        return torch.cat([lhs, rhs])
