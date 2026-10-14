# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch



class MXFP4QuantizeUtil:
    E2M1_max = 6.0

    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: Optional[int]) -> tuple:
        """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported.
        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_sizes (dict | None): The block sizes for quantization.
        """

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            
            
            left_side = x[..., 0::2]  
            right_side = x[..., 1::2]  
            new_data = (
                right_side.clone() << 4
            )  
            new_data[
                ..., : left_side.shape[-1]
            ] += left_side  
            return new_data

        if block_size is None:
            block_size = 32

        original_shape = input.shape
        original_dtype = input.dtype
        input = input.view(-1, block_size)
        
        input_amax = input.abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

        input = (input / torch.exp2(e8m0_scale)).view(original_shape)
        input_q = cast_fp4(input)
        input_q = fuse_uint4_to_uint8(input_q)
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
        return cls(original_shape, original_dtype, input_q), e8m0_scale

    @classmethod
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale, block_sizes):
        """Dequantze MXFP4 packed tensor to a target dtype."""

        def unfuse_uint8_to_uint4(x):
            """Unfuse uint8 values back to uint4 values.
            This is the inverse operation of fuse_uint4_to_uint8.
            """
            
            left_side = x & 0x0F

            
            right_side = (x >> 4) & 0x0F

            
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)

            
            result[..., 0::2] = left_side  
            result[..., 1::2] = right_side  

            return result

        e8m0_scale = scale
        block_size = block_sizes[-1]

        
        x_unfused = unfuse_uint8_to_uint4(quantized_data)
        
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(
            torch.float32
        )  
        magnitude = x_unfused & 0b0111  
        magnitude = magnitude.to(torch.long)

        
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device)

        
        
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)

        
        x_float = sign.float() * x_float

        
        x_float = x_float.reshape(-1, block_size)

        
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)  

        
        x_float = x_float * scale_factor

        
        return x_float.reshape(original_shape).to(dtype)
