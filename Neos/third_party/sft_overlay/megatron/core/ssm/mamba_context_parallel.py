# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import all_to_all
from megatron.core.utils import is_te_min_version

try:
    from einops import repeat

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    
    import transformer_engine  

    
    import transformer_engine_torch as tex
except ImportError:
    
    tex = None


class MambaContextParallel:
    """
    This class provides the following functionality related to Mamba "all-to-all" context parallel:
    1. Error checking, and creation of, relevant parameters (e.g. nheads_local_tpcp)
    2. Collective operations on activations, on each context parallel rank, before and after the
       convolution and SSM
    3. A convolution operator that uses the correct slices of trainable variables on the current
       context parallel rank
    4. Sliced views of relevant trainable variables for the current context parallel rank

    This class is intentionally not a sub-class of MegatronModule. This class does not contain any
    trainable variables of its own and should not be involved in any checkpoint loading or saving.

    Args:
        cp_group (torch.distributed.ProcessGroup):
            The process group to use for context parallel.
        d_inner_local_tp (int): d_inner on the current tp rank
        nheads_local_tp (int): nheads on the current tp rank
        ngroups_local_tp (int): ngroups on the current tp rank
        d_state (int): Mamba d_state
        conv1d_cp1 (nn.Conv1d):
            The conv1d op which would be applied on this tp rank if cp_size was 1
        dt_bias_cp1 (torch.Tensor):
            The dt_bias parameter which would be used on this tp rank if cp_size was 1
        A_log_cp1 (torch.Tensor):
            The A_log parameter which would be used on this tp rank if cp_size was 1
        D_cp1 (torch.Tensor): The D parameter which would be used on this tp rank if cp_size was 1
        D_has_hdim (bool): D parameter is sized to hidden dimension, rather than being per-head
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        d_inner_local_tp: int,
        nheads_local_tp: int,
        ngroups_local_tp: int,
        d_state: int,
        conv1d_cp1: nn.Conv1d,
        dt_bias_cp1: torch.Tensor,
        A_log_cp1: torch.Tensor,
        D_cp1: torch.Tensor,
        D_has_hdim: bool,
    ) -> None:
        if not HAVE_EINOPS:
            raise ImportError("einops is required by the Mamba model but cannot be imported")

        self.cp_group = cp_group
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.conv1d_cp1 = conv1d_cp1
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1
        self.D_has_hdim = D_has_hdim

        self.cp_size = self.cp_group.size()

        if self.cp_size == 1:
            self.d_inner_local_tpcp = self.d_inner_local_tp
            self.nheads_local_tpcp = self.nheads_local_tp
            self.ngroups_local_tpcp = self.ngroups_local_tp
            return

        self.cp_rank = self.cp_group.rank()

        
        assert (
            self.nheads_local_tp % self.cp_size == 0
        ), "nheads must be evenly divisible by tp_size * cp_size"
        
        self.nheads_local_tpcp = self.nheads_local_tp // self.cp_size

        
        
        
        
        self.d_inner_local_tpcp = self.d_inner_local_tp // self.cp_size

        
        if self.ngroups_local_tp < self.cp_size:
            assert (
                self.cp_size % self.ngroups_local_tp == 0
            ), "cp_size must be evenly divisible by ngroups/tp_size"
            
            self.group_repeat_count = self.cp_size // self.ngroups_local_tp
            self.ngroups_local_tpcp = 1
        else:
            assert (
                self.ngroups_local_tp % self.cp_size == 0
            ), "ngroups must be evenly divisible by tp_size * cp_size"
            
            
            self.group_repeat_count = 1
            self.ngroups_local_tpcp = self.ngroups_local_tp // self.cp_size

        
        
        
        

    def pre_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """Method to be applied before the convolution and SSM"""
        if self.cp_size == 1:
            return input_

        z, x, B, C, dt = torch.split(
            input_,
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp,  
                self.ngroups_local_tp * self.d_state,  
                self.ngroups_local_tp * self.d_state,  
                self.nheads_local_tp,  
            ],
            dim=-1,
        )

        

        
        z = _all_to_all_cp2hp(z, self.cp_group)

        
        x = _all_to_all_cp2hp(x, self.cp_group)

        
        
        
        
        B = repeat(
            B,
            "l b (g n) -> l b (g r n)",
            g=self.ngroups_local_tp,
            n=self.d_state,
            r=self.group_repeat_count,
        )
        C = repeat(
            C,
            "l b (g n) -> l b (g r n)",
            g=self.ngroups_local_tp,
            n=self.d_state,
            r=self.group_repeat_count,
        )

        
        B = _all_to_all_cp2hp(B, self.cp_group)

        
        C = _all_to_all_cp2hp(C, self.cp_group)

        
        dt = _all_to_all_cp2hp(dt, self.cp_group)

        output = torch.cat([z, x, B, C, dt], dim=-1)
        
        output = _undo_attention_load_balancing(output, self.cp_size, packed_seq_params)

        return output

    def post_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """Method to be applied after the convolution and SSM"""
        if self.cp_size == 1:
            return input_
        else:
            return _all_to_all_hp2cp(
                _redo_attention_load_balancing(input_, self.cp_size, packed_seq_params),
                self.cp_group,
            )

    def conv1d(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Performs a conv1d on one context parallel rank, using slices of the weight and bias from
        the convolution that would be run when cp_size=1
        """
        if self.cp_size == 1:
            return self.conv1d_cp1(input_)
        else:
            return F.conv1d(
                input=input_,
                weight=self.get_conv1d_weight(),
                bias=self.get_conv1d_bias(),
                stride=self.conv1d_cp1.stride,
                padding=self.conv1d_cp1.padding,
                dilation=self.conv1d_cp1.dilation,
                groups=self.conv1d_channels(),  
            )

    
    def conv1d_channels(self):
        """Returns the number of convolution channels on the current context parallel rank"""
        
        
        return self.d_inner_local_tpcp + 2 * self.ngroups_local_tpcp * self.d_state

    def get_conv1d_weight(self) -> torch.Tensor:
        """Returns a slice of the conv1d weight relevant to the current context parallel rank"""
        
        return self._slice_conv_param(self.conv1d_cp1.weight)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Returns a slice of the conv1d bias relevant to the current context parallel rank"""
        
        return self._slice_conv_param(self.conv1d_cp1.bias)

    def get_dt_bias(self) -> torch.Tensor:
        """Returns a slice of dt_bias relevant to the current context parallel rank"""
        return self._slice_vector_param(self.dt_bias_cp1)

    def get_A_log(self) -> torch.Tensor:
        """Returns a slice of A_log relevant to the current context parallel rank"""
        return self._slice_vector_param(self.A_log_cp1)

    def get_D(self) -> torch.Tensor:
        """Returns a slice of D relevant to the current context parallel rank"""
        return self._slice_vector_param(self.D_cp1, has_hdim=self.D_has_hdim)

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """
        Slices a cp_size=1 conv1d parameter (either weight or bias) along the first dimension,
        returning the parts of the parameter needed for convolution on the current context parallel
        rank. Parameter slicing is done in the forward path so that gradients will backpropagate to
        the cp_size=1 parameters.
        """
        if self.cp_size == 1:
            return param

        x, B, C = torch.split(
            param,
            [
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
            ],
            dim=0,
        )

        
        size = self.d_inner_local_tpcp
        start = self.cp_rank * size
        end = start + size
        x_sliced = x[start:end, ...]

        
        size = self.ngroups_local_tpcp * self.d_state
        start = (self.cp_rank // self.group_repeat_count) * size
        end = start + size
        B_sliced = B[start:end, ...]
        C_sliced = C[start:end, ...]

        return torch.cat([x_sliced, B_sliced, C_sliced], dim=0).contiguous()

    def _slice_vector_param(self, param: torch.Tensor, has_hdim: bool = False) -> torch.Tensor:
        """
        Slices a cp_size=1 vector parameter along the first dimension, returning the part of the
        parameter needed on the current context parallel rank. Parameter slicing is done in the
        forward path so that gradients will backpropagate to the cp_size=1 parameters.
        """
        if self.cp_size == 1:
            return param

        size = self.d_inner_local_tpcp if has_hdim else self.nheads_local_tpcp
        start = self.cp_rank * size
        end = start + size
        return param[start:end]



def _all_to_all_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence/context-parallel-size, batch, local-hidden] to
    [global-sequence, batch, local-hidden/context-parallel-size].

    Args:
        input_ (torch.Tensor):
            The input tensor, which is partitioned along the sequence dimension
        cp_group (torch.distributed.ProcessGroup):
            Process group to use for context parallel

    Returns:
        torch.Tensor: The output tensor with shape
            [global-sequence, batch, local-hidden/context-parallel-size].
    """
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    
    input_ = input_.reshape(-1, h_in)
    
    world_size = cp_group.size()
    h_out = h_in // world_size
    split_tensors = torch.split(input_, split_size_or_sections=h_out, dim=1)
    
    concat_tensor = torch.cat(split_tensors, dim=0)
    
    
    
    output = all_to_all(cp_group, concat_tensor)
    
    output = output.reshape(s_in * world_size, b_in, h_out)
    return output



def _all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence, batch, local-hidden/context-parallel-size] to
    [global-sequence/context-parallel-size, batch, local-hidden].

    Args:
        input_ (torch.Tensor):
            The input tensor, which is partitioned along the hidden dimension
        cp_group (torch.distributed.ProcessGroup):
            Process group to use for context parallel

    Returns:
        torch.Tensor: The output tensor with shape
            [global-sequence/context-parallel-size, batch, local-hidden].
    """
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    
    input_ = input_.reshape(-1, h_in)
    
    input_exchanged = all_to_all(cp_group, input_)
    
    world_size = cp_group.size()
    s_out = s_in // world_size
    split_tensors = torch.split(input_exchanged, split_size_or_sections=s_out * b_in, dim=0)
    
    output = torch.cat(split_tensors, dim=-1)
    
    output = output.reshape(s_out, b_in, h_in * world_size)
    return output


def _undo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Undoes the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 162534 to 123456 for
    sequential processing by the convolution and SSM.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [2 * i for i in range(num_chunks_div_2)] + [
            num_chunks - 2 * i - 1 for i in range(num_chunks_div_2)
        ]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        output = torch.empty_like(input_)
        for cp_rank in range(cp_size):
            start = cp_rank * seqlen_per_rank
            end = start + seqlen_per_rank
            index = tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            output[index] = input_[start:end]
        return output


def _redo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Redo the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 123456 to 162534 for
    efficient processing by attention.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [None] * num_chunks
        order[::2] = range(num_chunks_div_2)  
        order[1::2] = reversed(range(num_chunks_div_2, num_chunks))  
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        index = torch.empty(total_tokens, device=input_.device, dtype=torch.int32)
        for cp_rank in range(cp_size):
            start = cp_rank * seqlen_per_rank
            end = start + seqlen_per_rank
            index[start:end] = tex.thd_get_partitioned_indices(
                cu_seqlens, total_tokens, cp_size, cp_rank
            )
        return input_.index_select(0, index)
