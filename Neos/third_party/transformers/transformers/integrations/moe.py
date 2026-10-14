# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import (
    is_torch_available,
    is_torch_greater_or_equal,
    is_torch_less_or_equal,
    is_torchdynamo_compiling,
)
from .deepgemm import deepgemm_bf16_experts_forward
from .sonicmoe import sonicmoe_experts_forward


if is_torch_available():
    import torch

    
    
    
    is_torch_greater_or_equal = torch._dynamo.assume_constant_result(is_torch_greater_or_equal)
    is_torch_less_or_equal = torch._dynamo.assume_constant_result(is_torch_less_or_equal)


logger = logging.get_logger(__name__)










































def _batched_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Batched linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (batch_size, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (batch_size, output_dim, input_dim) if transposed is `False`,
            else of shape (batch_size, input_dim, output_dim).
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (batch_size, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (batch_size, output_dim).
    """
    if is_transposed:
        
        out = torch.bmm(input.unsqueeze(1), weight).squeeze(1)
    else:
        
        out = torch.bmm(weight, input.unsqueeze(-1)).squeeze(-1)

    if bias is not None:
        out.add_(bias)

    return out


def batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    
    
    selected_hidden_states = hidden_states.repeat_interleave(num_top_k, dim=0)
    sample_weights = top_k_weights.reshape(-1)  
    expert_ids = top_k_index.reshape(-1)  

    
    
    
    
    expert_ids = expert_ids.clamp(0, self.num_experts - 1)

    
    if self.has_gate:
        selected_weights = self.gate_up_proj[expert_ids]
        selected_biases = self.gate_up_proj_bias[expert_ids] if self.has_bias else None
    else:
        selected_weights = self.up_proj[expert_ids]
        selected_biases = self.up_proj_bias[expert_ids] if self.has_bias else None

    
    proj_out = _batched_linear(
        selected_hidden_states, selected_weights, bias=selected_biases, is_transposed=self.is_transposed
    )  

    
    if self.has_gate:
        
        proj_out = self._apply_gate(proj_out)  
    else:
        
        proj_out = self.act_fn(proj_out)  

    
    selected_weights = self.down_proj[expert_ids]
    selected_biases = self.down_proj_bias[expert_ids] if self.has_bias else None

    
    proj_out = _batched_linear(
        proj_out, selected_weights, bias=selected_biases, is_transposed=self.is_transposed
    )  

    
    weighted_out = proj_out * sample_weights.unsqueeze(-1)  

    
    
    
    
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)





def _grouped_mm_fallback(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """
    Fallback grouped matrix multiplication used when `torch.nn.functional.grouped_mm` and `torch._grouped_mm`
    are unavailable or incompatible with `torch.compile` (e.g. non-bfloat16 weights).

    Args:
        input (`torch.Tensor`): Input of shape (S, input_dim), sorted by expert id.
        weight (`torch.Tensor`): Expert weights of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`): Cumulative token counts per expert of shape (num_experts,).
    Returns:
        `torch.Tensor`: Output of shape (S, output_dim).
    """
    output = torch.zeros(input.size(0), weight.size(2), device=input.device, dtype=input.dtype)  

    start = 0
    
    
    for i, end in enumerate(offs.tolist()):
        if start == end:
            continue
        torch.mm(input[start:end], weight[i], out=output[start:end])
        start = end

    return output


def _grouped_mm_fallback_fake(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Shape/dtype inference stub for `_grouped_mm_fallback` required by `torch.compile`."""
    assert input.dim() == 2, f"input must be 2D (S, input_dim), got shape {tuple(input.shape)}"
    assert weight.dim() == 3, (
        f"weight must be 3D (num_experts, input_dim, output_dim), got shape {tuple(weight.shape)}"
    )
    assert offs.dim() == 1, f"offs must be 1D (num_experts,), got shape {tuple(offs.shape)}"
    assert offs.size(0) == weight.size(0), f"offs length {offs.size(0)} must match number of experts {weight.size(0)}"
    assert input.size(1) == weight.size(1), (
        f"input_dim mismatch: input has {input.size(1)}, weight has {weight.size(1)}"
    )
    assert offs.dtype in (torch.int32, torch.int64), f"offs must be an integer tensor, got {offs.dtype}"
    return torch.empty(input.size(0), weight.size(2), device=input.device, dtype=input.dtype)


def _grouped_mm_fallback_setup_context(ctx, inputs, output):
    """Saves input and weight for backward; offs is stored directly as it is a non-differentiable integer tensor."""
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.offs = inputs[2]


def _grouped_mm_fallback_backward(ctx, grad_output):
    """Backward pass for `_grouped_mm_fallback`. Computes grad_input and grad_weight per expert group; offs has no gradient."""
    input, weight = ctx.saved_tensors
    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)

    start = 0
    
    
    for i, end in enumerate(ctx.offs.tolist()):
        if start == end:
            continue
        torch.mm(grad_output[start:end], weight[i].T, out=grad_input[start:end])
        torch.mm(input[start:end].T, grad_output[start:end], out=grad_weight[i])
        start = end

    return grad_input, grad_weight, None


if is_torch_available():
    torch.library.custom_op(
        "transformers::grouped_mm_fallback",
        _grouped_mm_fallback,
        mutates_args=(),
        schema="(Tensor input, Tensor weight, Tensor offs) -> Tensor",
    )
    torch.library.register_fake("transformers::grouped_mm_fallback", _grouped_mm_fallback_fake)
    torch.library.register_autograd(
        "transformers::grouped_mm_fallback",
        _grouped_mm_fallback_backward,
        setup_context=_grouped_mm_fallback_setup_context,
    )


def _can_use_grouped_mm(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> bool:
    """
    Check if torch.nn.functional.grouped_mm or torch._grouped_mm can be used based on availability and compatibility with torch.compile.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
    Returns:
        `bool`: True if grouped_mm can be used, False otherwise.
    """
    if (is_torchdynamo_compiling() and weight.dtype != torch.bfloat16) or (
        weight.device.type == "cpu"
        
        and is_torch_less_or_equal("2.10.0", accept_dev=True)
        and (weight.data_ptr() % 16 != 0 or input.data_ptr() % 16 != 0)
    ):
        
        
        
        
        
        
        return False

    
    
    
    if weight.device.type == "cuda":
        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.cuda.get_device_capability(weight.device) >= (8, 0)
        if hasattr(torch, "_grouped_mm"):
            if is_torch_greater_or_equal("2.9", accept_dev=True):
                return torch.cuda.get_device_capability(weight.device) >= (8, 0)
            else:
                return torch.cuda.get_device_capability(weight.device) >= (9, 0)

        return False

    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")


def _grouped_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Grouped matrix multiplication dispatcher that uses torch.nn.functional.grouped_mm if available, else falls back to torch._grouped_mm.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """

    if _can_use_grouped_mm(input, weight, offs):
        
        
        
        
        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.nn.functional.grouped_mm(input.to(weight.dtype), weight, offs=offs)
        elif hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(input.to(weight.dtype), weight, offs=offs)

    return torch.ops.transformers.grouped_mm_fallback(input, weight, offs=offs)


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Grouped linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim) if `is_transposed`,
            else of shape (num_experts, output_dim, input_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (num_experts, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """
    if is_transposed:
        
        out = _grouped_mm(input, weight, offs=offs)
    else:
        
        out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)

    if bias is not None:
        
        out.add_(bias)

    return out


def grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    
    sample_weights = top_k_weights.reshape(-1)  
    expert_ids = top_k_index.reshape(-1)  

    
    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    
    
    
    
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=self.num_experts - 1)

    
    
    
    
    
    
    if self.has_gate:
        selected_weights = self.gate_up_proj
        selected_biases = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    else:
        selected_weights = self.up_proj
        selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

    
    selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)

    
    proj_out = _grouped_linear(
        selected_hidden_states_g, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  

    
    if self.has_gate:
        
        proj_out = self._apply_gate(proj_out)  
    else:
        
        proj_out = self.act_fn(proj_out)  

    
    selected_weights = self.down_proj
    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    
    proj_out = _grouped_linear(
        proj_out, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  

    
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  

    
    weighted_out.masked_fill_(sentinel_mask, 0.0)

    
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]  

    
    
    
    
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_bf16_experts_forward,
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe": sonicmoe_experts_forward,
    }

    def get_interface(self, experts_implementation: str, default: Callable) -> Callable:
        """Return the requested `experts_implementation`. Also strictly check its validity, and raise if invalid."""
        if experts_implementation is None:
            logger.warning_once(
                "You tried to access the `ExpertsInterface` with a `config._experts_implementation` set to `None`. This "
                "is expected if you use an Expert Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._experts_implementation`"
            )
        elif experts_implementation != "eager" and experts_implementation not in self:
            raise KeyError(
                f"`{experts_implementation}` is not a valid experts implementation registered in the `ExpertsInterface`"
            )
        return super().get(experts_implementation, default)


ALL_EXPERTS_FUNCTIONS = ExpertsInterface()


def _default_apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
    """
    Default gating mechanism: splits the gate_up_out into gate and up parts,
    applies the activation function to the gate part, and multiplies it with the up part.
    Args:
        gate_up_out (`torch.Tensor`):
            The output tensor from the gate and up projection of shape (S, 2 * intermediate_dim).
    Returns:
        `torch.Tensor`: The gated output tensor of shape (S, intermediate_dim).
    """
    gate, up = gate_up_out.chunk(2, dim=-1)  
    return self.act_fn(gate) * up  


def use_experts_implementation(
    experts_class: type[torch.nn.Module] | None = None,
    *,
    experts_interface: ExpertsInterface = ALL_EXPERTS_FUNCTIONS,
    is_concatenated: bool = True,
    is_transposed: bool = False,
    has_bias: bool = False,
    has_gate: bool = True,
) -> type[torch.nn.Module]:
    """Decorator to modify experts class to support different experts implementations.

    Args:
        experts_class (`type[torch.nn.Module]`, *optional*):
            The experts class to modify. If not provided, returns a decorator that can be applied to the class.
        experts_interface (`ExpertsInterface`, *optional*, defaults to `ALL_EXPERTS_FUNCTIONS`):
            The experts interface to use for dispatching the forward method.
        is_concatenated (`bool`, *optional*, defaults to `True`):
            Whether the expert weights are stored in concatenated layout [gate;up]
            or interleaved layout [gate0, up0, gate1, up1, ...].
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the expert weights are stored in transposed format.
        has_bias (`bool`, *optional*, defaults to `False`):
            Whether the expert layers include bias terms or not.
        has_gate (`bool`, *optional*, defaults to `True`):
            Whether the experts use a gating mechanism or not.
            Whether it has gate_up_proj weights or just up_proj weights.

    Returns:
        `type[torch.nn.Module]`: The modified experts class.
    """

    def wrapper(experts_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
        original_init = experts_class.__init__
        original_forward = experts_class.forward

        @wraps(original_init)
        def __init__(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            self.config = config
            self.has_gate = has_gate
            self.has_bias = has_bias
            self.is_transposed = is_transposed
            self.is_concatenated = is_concatenated

        @wraps(original_forward)
        def forward(self, *args, **kwargs):
            experts_forward = experts_interface.get_interface(self.config._experts_implementation, original_forward)
            return experts_forward(self, *args, **kwargs)

        if not hasattr(experts_class, "_apply_gate"):
            experts_class._apply_gate = _default_apply_gate

        experts_class.__init__ = __init__
        experts_class.forward = forward
        return experts_class

    if experts_class is not None:
        return wrapper(experts_class)

    return wrapper
