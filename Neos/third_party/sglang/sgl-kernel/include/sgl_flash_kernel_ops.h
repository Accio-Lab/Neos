/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <vector>

#include "sgl_kernel_torch_shim.h"

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

  
                       
   
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    at::Tensor q,                                                               
    at::Tensor k,                                                                                               
                                                     
    at::Tensor v,                                                                                      
                                                                 
    std::optional<at::Tensor> k_new_,                                                                               
    std::optional<at::Tensor> v_new_,                                                                                 
    std::optional<at::Tensor> q_v_,                                                                       
    std::optional<at::Tensor> out_,                                                                   
    std::optional<at::Tensor> cu_seqlens_q_,            
    std::optional<at::Tensor> cu_seqlens_k_,            
    std::optional<at::Tensor> cu_seqlens_k_new_,        
    std::optional<at::Tensor>
        seqused_q_,                                                                                               
    std::optional<at::Tensor>
        seqused_k_,                                                                                
    std::optional<int64_t> max_seqlen_q_,
                                          
    std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,                                     
    std::optional<at::Tensor> kv_batch_idx_,                                            
    std::optional<at::Tensor> leftpad_k_,           
    std::optional<at::Tensor> rotary_cos_,                                     
    std::optional<at::Tensor> rotary_sin_,                                     
    std::optional<at::Tensor> seqlens_rotary_,      
    std::optional<at::Tensor> q_descale_,                              
    std::optional<at::Tensor> k_descale_,                  
    std::optional<at::Tensor> v_descale_,                  
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,                                                                            
    std::optional<at::Tensor> scheduler_metadata_,            
    int64_t num_splits,
    std::optional<bool> pack_gqa_,
    int64_t sm_margin,
    std::optional<const at::Tensor>& sinks_);        
