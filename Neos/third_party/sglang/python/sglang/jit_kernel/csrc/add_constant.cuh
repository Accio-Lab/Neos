#include <sgl_kernel/tensor.h>                                                    
#include <sgl_kernel/utils.h>                                

#include <sgl_kernel/utils.cuh>                     

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

                                                               
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

                              
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                                                  
      .with_dtype<int32_t>()                          
      .with_device<kDLCUDA>(device_)                           
      .verify(dst)                                       
      .verify(src);                                      

                                                              
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  [[maybe_unused]]                             
  const size_t dynamic_smem = 0;
  [[maybe_unused]]                                                                 
  const cudaStream_t stream = LaunchKernel::resolve_device(device);
                                                       
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

                                                                    
  LaunchKernel(grid_size, kBlockSize, device                   )(
                        
      add_constant_kernel<kConstant>,
                         
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}              
