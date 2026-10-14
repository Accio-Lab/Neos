#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace host::runtime {

                                                                         
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

                                                
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

                                                           
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
  return cc_major;
}

                             
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

                                                                          
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

}                            
