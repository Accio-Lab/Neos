# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch

from megatron.core.jit import jit_fuser




def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool) -> Tensor
    
    
    
    
    

    x, bias = x_with_bias  

    
    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    
    
    
    
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)

    
    
    
    
    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out


def bias_dropout_add_unfused(training):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def get_bias_dropout_add(training, fused):
    if fused:
        
        
        
        
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
