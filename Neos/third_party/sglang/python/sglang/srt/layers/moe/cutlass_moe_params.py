from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class CutlassMoEType(Enum):
    """
    Enum for the different types of cutlass moe operations
    that are currently supported in SGLang.
    """

    BlockscaledFP8 = auto()
    BlockscaledFP4 = auto()


@dataclass
class CutlassMoEParams:
    """
    Parameters for the cutlass moe operation.
    """

    
    cutlass_moe_type: CutlassMoEType

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    ab_strides_13: torch.Tensor
    ab_strides_2: torch.Tensor
    c_strides_13: torch.Tensor
    c_strides_2: torch.Tensor

    
    
    
    
    
    m: int
    intermediate_size_per_partition: int
    hidden_size: int
    num_experts: int
    device: torch.device

    
    
    a_ptrs: torch.Tensor

    
    
    b_ptrs: torch.Tensor

    
    
    out_ptrs: torch.Tensor
    
    
    
    a_scales_ptrs: torch.Tensor
    b_scales_ptrs: torch.Tensor

    
    
    
    expert_offsets: torch.Tensor

    
    
    
    
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor
    
    blockscale_offsets: Optional[torch.Tensor] = None

    def __init__(
        self,
        cutlass_moe_type: CutlassMoEType,
        device: torch.device,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ):
        self.cutlass_moe_type = cutlass_moe_type
        self.device = device
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.hidden_size = hidden_size
        self.n = self.intermediate_size_per_partition
        self.k = self.hidden_size
        self.e = self.num_experts
        self.ab_strides_13 = torch.full(
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.ab_strides_2 = torch.full(
            (self.e,), self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_13 = torch.full(
            (self.e,), 2 * self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_2 = torch.full(
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.expert_offsets = torch.empty(
            (self.e + 1,), dtype=torch.int32, device=self.device
        )
        self.problem_sizes1 = torch.empty(
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        self.problem_sizes2 = torch.empty(
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        if self.cutlass_moe_type == CutlassMoEType.BlockscaledFP4:
            self.blockscale_offsets = torch.empty(
                (self.e + 1,), dtype=torch.int32, device=self.device
            )
        else:
            self.blockscale_offsets = None
        self.a_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.b_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.out_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.a_scales_ptrs = torch.empty(
            (self.e,), dtype=torch.int64, device=self.device
        )
        self.b_scales_ptrs = torch.empty(
            (self.e,), dtype=torch.int64, device=self.device
        )

    def to_gemm1_args(self) -> dict:
        return {
            "ab_strides": self.ab_strides_13,
            "c_strides": self.c_strides_13,
            "problem_sizes": self.problem_sizes1,
            "expert_offsets": self.expert_offsets[:-1],
            "blockscale_offsets": self.blockscale_offsets[:-1],
            
            
            
            
            
        }

    def to_gemm2_args(self) -> dict:
        return {
            "ab_strides": self.ab_strides_2,
            "c_strides": self.c_strides_2,
            "problem_sizes": self.problem_sizes2,
            "expert_offsets": self.expert_offsets[:-1],
            "blockscale_offsets": self.blockscale_offsets[:-1],
            
            
            
            
            
        }
