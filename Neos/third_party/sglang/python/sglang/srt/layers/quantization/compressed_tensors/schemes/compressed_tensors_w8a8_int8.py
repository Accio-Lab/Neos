# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import is_cuda

__all__ = ["CompressedTensorsW8A8Int8", "NPUCompressedTensorsW8A8Int8"]

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm


class CompressedTensorsW8A8Int8(CompressedTensorsLinearScheme):

    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            layer.register_parameter("input_scale", input_scale)

            if not self.input_symmetric:
                
                
                
                input_zero_point = PerTensorScaleParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )
                layer.register_parameter("input_zero_point", input_zero_point)

    @classmethod
    def get_min_capability(cls) -> int:
        
        return 80

    def process_weights_after_loading(self, layer) -> None:
        
        
        
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight
            weight_scale = layer.weight_scale.data

            layer.weight = Parameter(weight.t(), requires_grad=False)
            
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            if self.input_symmetric:
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )
            else:
                input_scale = layer.input_scale
                input_zero_point = layer.input_zero_point

                
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)

                
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)

                layer.input_scale = Parameter(scale, requires_grad=False)
                layer.input_zero_point = Parameter(azp, requires_grad=False)
        else:
            layer.input_scale = None
            layer.input_zero_point = None

        
        
        
        
        
        if not self.input_symmetric:
            weight = layer.weight
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.is_static_input_scheme:
                
                
                azp_adj = layer.input_zero_point * azp_adj
            layer.azp_adj = Parameter(azp_adj, requires_grad=False)
        else:
            layer.azp_adj = None

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        x_q, x_scale = per_token_quant_int8(x)

        return int8_scaled_mm(
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
        )


class NPUCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8):

    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        super().__init__(strategy, is_static_input_scheme, input_symmetric)
        
        
        if self.is_static_input_scheme:
            raise NotImplementedError(
                "Static compressed-tensors scheme is not yet supported on NPU."
            )
        self.kernel = NPUW8A8Int8DynamicLinearMethod()

    @classmethod
    def get_min_capability(cls) -> int:
        return NotImplementedError

    def process_weights_after_loading(self, layer):
        return self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer, x, bias):
        return self.kernel.apply(layer, x, bias)
