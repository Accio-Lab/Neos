#pragma once

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"

   
                                                                      
                                                                          
                                          
                                                                               
                                                                     
   
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
