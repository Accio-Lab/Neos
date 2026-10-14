/*Adapt from:
https://github.com/neuralmagic/vllm-flash-attention/blob/90eacc1af2a7c3de62ea249e929ed5faccf38954/csrc/common/pytorch_shim.h
  Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <torch/library.h>

   
                                                                              
                                                               
                                                                        
                                                                         
                                                                               
                                                                            
  
                                                                       
                                                                               
                                                                        
           
             
                          
                                         
                                        
               
              
                               
                                        
   

template <typename T>
struct pytorch_library_compatible_type {
  using type = T;
  static T convert_from_type(T arg) {
    return arg;
  }
};

template <typename T>
using pytorch_library_compatible_type_t = typename pytorch_library_compatible_type<T>::type;

template <typename T>
T convert_from_pytorch_compatible_type(pytorch_library_compatible_type_t<T> arg) {
  return pytorch_library_compatible_type<T>::convert_from_type(arg);
}

                                                        
                                                                     
                            
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>&> {
  using type = const c10::optional<T>&;
  static c10::optional<T>& convert_from_type(const c10::optional<T>& arg) {
    return const_cast<c10::optional<T>&>(arg);
  }
};

                            
                                                                 
                                                                       
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>> {
  using type = c10::optional<pytorch_library_compatible_type_t<T>>;
  static c10::optional<pytorch_library_compatible_type_t<T>> convert_from_type(c10::optional<T> arg) {
    return arg;
  }
};

                                                                               
template <>
struct pytorch_library_compatible_type<c10::optional<const at::Tensor>&> {
  using type = const c10::optional<at::Tensor>&;
  static c10::optional<const at::Tensor>& convert_from_type(const c10::optional<at::Tensor>& arg) {
    return const_cast<c10::optional<const at::Tensor>&>(reinterpret_cast<const c10::optional<const at::Tensor>&>(arg));
  }
};

                         
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "int64_t value is too large to be converted to int");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "int64_t value is too small to be converted to int");
    return arg;
  }
};

                          
template <>
struct pytorch_library_compatible_type<float> {
  using type = double;
  static float convert_from_type(double arg) {
    TORCH_CHECK(
        std::abs(arg) <= std::numeric_limits<float>::max(), "double value is too large to be converted to float");
    return arg;
  }
};

  
              
  

template <typename Ret, typename... Args>
auto make_pytorch_shim(Ret (*fun)(Args... args)) {
  return [fun](pytorch_library_compatible_type_t<Args>... args) {
    return fun(convert_from_pytorch_compatible_type<Args>(args)...);
  };
}
