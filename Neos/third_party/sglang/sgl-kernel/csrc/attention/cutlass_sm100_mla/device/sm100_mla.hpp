/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file
  \brief An universal device layer for cutlass 3.x-style kernels.
*/

// clang-format off
#pragma once

         
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif                            

#include "../kernel/sm100_fmha_mla_tma_warpspecialized.hpp"
#include "../kernel/sm100_fmha_mla_reduction.hpp"

                                                                                

namespace cutlass::fmha::device {

using namespace cute;
using namespace cutlass::fmha::kernel;


                                                                                
                                                                                
                                                                                

template<
    class Kernel_
>
class MLA {
public:

  using Kernel = Kernel_;

  using ReductionKernel = cutlass::fmha::kernel::Sm100FmhaMlaReductionKernel<
      typename Kernel::ElementOut,
      typename Kernel::ElementAcc,
      typename Kernel::ElementAcc,
      Kernel::TileShapeH::value,
      Kernel::TileShapeL::value,
      256              
  >;

                                  
  using KernelArguments = typename Kernel::Arguments;
  using ReductionArguments = typename ReductionKernel::Arguments;

  using Arguments = KernelArguments;

                                    
  using KernelParams = typename Kernel::Params;
  using ReductionParams = typename ReductionKernel::Params;
  struct Params {
    KernelParams fmha_params;
    ReductionParams reduction_params;
  };

private:

                                  
  Params params_;

  bool is_initialized(bool set = false) {
    static bool initialized = false;
    if (set) initialized = true;
    return initialized;
  }

  static ReductionArguments to_reduction_args(Arguments const& args) {
    auto [H, K, D, B] = args.problem_shape;
    return ReductionArguments{
      nullptr, args.epilogue.ptr_o, nullptr, args.epilogue.ptr_lse,
      args.mainloop.softmax_scale, B, args.split_kv, K, args.mainloop.ptr_seq,
      args.ptr_split_kv, Kernel::TileShapeS::value
    };
  }

public:

                                 
  Params const& params() const {
    return params_;
  }

  static void set_split_kv (KernelArguments& args) {
    if (args.split_kv >= 1) return;
    auto [H, K, D, B] = args.problem_shape;
    int sm_count = args.hw_info.sm_count;
    int max_splits = ceil_div(K, 128);
    int sms_per_batch = max(1, sm_count / B);
    int split_heur = min(max_splits, sms_per_batch);
    int waves = ceil_div(B * split_heur, sm_count);
    int k_waves = ceil_div(max_splits, split_heur);
    int split_wave_aware = ceil_div(max_splits, k_waves);
    args.split_kv = split_wave_aware;
  }

                                                                
  static Status
  can_implement(Arguments const& args) {
    if (! Kernel::can_implement(args)) {
      return Status::kInvalid;
    }
    if (! ReductionKernel::can_implement(to_reduction_args(args))) {
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

                             
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    workspace_bytes += ReductionKernel::get_workspace_size(to_reduction_args(args));
    return workspace_bytes;
  }

                                                                     
  static int maximum_active_blocks(int                     = -1) {
    CUTLASS_TRACE_HOST("MLA::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = Kernel::SharedStorageSize;

                                                         
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();                          
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

                                              
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<Kernel>,
        Kernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError();                          
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

                                            
  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

                               
    Status status = Kernel::initialize_workspace(args, workspace, stream);
    if (status != Status::kSuccess) {
      return status;
    }
    status = ReductionKernel::initialize_workspace(to_reduction_args(args), workspace, stream);
    if (status != Status::kSuccess) {
      return status;
    }
    KernelParams kernel_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum   = kernel_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = kernel_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace);
                                      
    params_ = Params {kernel_params, reduction_params};

    if (is_initialized()) return Status::kSuccess;

                                                  
                                                     
    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result = cudaFuncSetAttribute(
          device_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();                          
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    is_initialized(true);

    return Status::kSuccess;
  }

                                                                                            
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("MLA()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    auto fmha_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum   = fmha_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = fmha_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace);
                                      
    params_ = Params {fmha_params, reduction_params};

    return Status::kSuccess;
  }

                                                                                                        
                                                                                          
  static Status
  run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::run()");
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params.fmha_params);

                                       
    int smem_size = Kernel::SharedStorageSize;

    Status launch_result;
                                                             
    if constexpr(Kernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
                   cute::size<1>(typename Kernel::ClusterShape{}),
                   cute::size<2>(typename Kernel::ClusterShape{}));
      void const* kernel = (void const*) device_kernel<Kernel>;
      void* kernel_params[] = {&params.fmha_params};
      launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
    }
    else {
      launch_result = Status::kSuccess;
      device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params.fmha_params);
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess != result or Status::kSuccess != launch_result) {
                                
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
    if (params.reduction_params.split_kv > 1) {
                                
      dim3 const block = ReductionKernel::get_block_shape();
      dim3 const grid  = ReductionKernel::get_grid_shape(params.reduction_params);
      device_kernel<ReductionKernel><<<grid, block, 0, stream>>>(params.reduction_params);
      cudaError_t result = cudaGetLastError();
      if (cudaSuccess == result) {
        return Status::kSuccess;
      }
      else {
        CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
        return Status::kErrorInternal;
      }
    }
    else {
      return Status::kSuccess;
    }
  }

    
                                                                                                            
    

                                                                                                 
  Status
  run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

                                                                                                 
  Status
  operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

                                                                                                       
  Status
  run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

                                                                                                       
  Status
  operator()(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }
};

                                                                                

}                                   

                                                                                
