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

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/be1788106245496872d18e702978e59b6bfd50e0/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/device/gemm_universal_base_compat.h
#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/trace.h>

                                                                                

namespace cutlass {
namespace gemm {
namespace device {

                                                                                                 

  
                                                                                               
                                                                                              
                                                                                                 

                                                                                                   
                                     
    

template <typename GemmKernel_>
class GemmUniversalBaseCompat {
 public:
  using GemmKernel = GemmKernel_;
  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  using ElementAccumulator = typename GemmKernel::Mma::Policy::Operator::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

                        
  using Arguments = typename GemmKernel::Arguments;

 protected:
                              
  typename GemmKernel::Params params_;

 protected:
                                                                          
  static void get_grid_shape_(gemm::GemmCoord& grid_tiled_shape, int& gemm_k_size, Arguments const& args) {
                           
    ThreadblockSwizzle threadblock_swizzle;

    grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, args.batch_count);

    gemm_k_size = args.problem_size.k();

    if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      int const kAlignK =
          const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

      gemm_k_size = round_up(ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

      if (gemm_k_size) {
        grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
      }
    }
  }

 public:
                          
  GemmUniversalBaseCompat() {}

                                                                
  static Status can_implement(Arguments const& args) {
                           
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

    uint32_t const kGridYZMax = ((1 << (sizeof(uint16_t) * 8)) - 1);

    if (!(grid.y <= kGridYZMax && grid.z <= kGridYZMax)) {
      return Status::kErrorInvalidProblem;
    }

    return GemmKernel::can_implement(args);
  }

                             
  static size_t get_workspace_size(Arguments const& args) {
    CUTLASS_TRACE_HOST("GemmUniversalBaseCompat::get_workspace_size()");

    size_t workspace_bytes = 0;

                           
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
                                                               
      workspace_bytes = sizeof(ElementC) * size_t(args.batch_stride_D) * size_t(grid_tiled_shape.k());
    } else if (args.mode == GemmUniversalMode::kGemm && grid_tiled_shape.k() > 1) {
                                                                                                 
                                              
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    workspace_bytes += GemmKernel::get_extra_workspace_size(args, grid_tiled_shape);

    return workspace_bytes;
  }

                             
  static dim3 get_grid_shape(Arguments const& args) {
    CUTLASS_TRACE_HOST("GemmUniversalBaseCompat::get_grid_shape()");

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);
    dim3 result = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

    CUTLASS_TRACE_HOST(
        "  grid_tiled_shape: " << grid_tiled_shape << "\n"
                               << "  result = {" << result << "}");

    return result;
  }

                                                                     
  static int maximum_active_blocks(int smem_capacity = -1) {
    CUTLASS_TRACE_HOST("GemmUniversalBaseCompat::maximum_active_blocks()");

    int max_active_blocks = -1;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    if (smem_size <= (48 << 10)) {
      cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_active_blocks, Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size);

      if (result == cudaSuccess) {
        CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
      }
    } else {
                                                                                     
      cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_active_blocks, Kernel<GemmKernel>, GemmKernel::kThreadCount, 0);

      if (result != cudaSuccess) {
        CUTLASS_TRACE_HOST(
            "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error " << cudaGetErrorString(result));

        return -1;
      }

      if (smem_capacity < 0) {
        int device_idx = 0;
        result = cudaGetDevice(&device_idx);

        if (result != cudaSuccess) {
          return -1;
        }

        cudaDeviceProp properties;
        result = cudaGetDeviceProperties(&properties, device_idx);

        if (result != cudaSuccess) {
          return -1;
        }

        smem_capacity = static_cast<int>(properties.sharedMemPerMultiprocessor);
      }

      int occupancy = std::min(max_active_blocks, smem_capacity / smem_size);

      CUTLASS_TRACE_HOST("  occupancy: " << occupancy);

      return occupancy;
    }

    CUTLASS_TRACE_HOST("  returning internal error");

    return -1;
  }

                                            
  Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST(
        "GemmUniversalBaseCompat::initialize() - workspace " << workspace
                                                             << ", stream: " << (stream ? "non-null" : "null"));

    size_t workspace_bytes = get_workspace_size(args);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    if (workspace_bytes) {
      if (!workspace) {
        CUTLASS_TRACE_HOST("  error: device workspace must not be null");

        return Status::kErrorWorkspaceNull;
      }

      if (args.mode == GemmUniversalMode::kGemm) {
        CUTLASS_TRACE_HOST("  clearing device workspace");
        cudaError_t result = cudaMemsetAsync(workspace, 0, workspace_bytes, stream);

        if (result != cudaSuccess) {
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));

          return Status::kErrorInternal;
        }
      }
    }

                          
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int gemm_k_size = 0;

    get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

                                      
    params_ = typename GemmKernel::Params(args, grid_tiled_shape, gemm_k_size, static_cast<int*>(workspace));

                                                 
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result =
          cudaFuncSetAttribute(Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

                                                    
  Status update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("GemmUniversalBaseCompat()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_.update(args, workspace);

    return Status::kSuccess;
  }

                                              
  Status run(cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("GemmUniversalBaseCompat::run()");

      
                                          
      

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

      
                    
      

    CUTLASS_TRACE_HOST("  grid: (" << grid << "),  block: (" << block << "),  SMEM: " << smem_size << " bytes");

             
    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

      
                       
      
    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

                                              
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

                                              
  Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

                                                                                                 

}                     
}                   
}                      

                                                                                                 
