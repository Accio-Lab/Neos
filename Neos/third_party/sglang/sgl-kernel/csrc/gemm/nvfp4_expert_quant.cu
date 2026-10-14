#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include "nvfp4_quant.cuh"
#include "utils.h"

                                                            
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                          
  auto localMax = __habs2(vec.elts[0]);

                       
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

                                                                
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
                                           
  float vecMax = float(__hmax(localMax.x, localMax.y));

                                                              
                                 
                                         
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
                                     
  uint8_t fp8SFVal;
                                           
  if constexpr (UE8M0_SF) {
                                                
                                                                      
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
                            
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
                                                                     
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
                            
    SFValue = float(tmp);
  }
                          
                                                                        
                                                  
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  if (SFout) {
                                             
    *SFout = fp8SFVal;
  }

                                
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

                            
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

                                            
  return e2m1Vec;
#else
  return 0;
#endif
}

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

template <class Type>
inline __device__ void silu_and_mul(PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  float2 x[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 y[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      x[i] = __half22float2(x_vec.elts[i]);
      y[i] = __half22float2(y_vec.elts[i]);
      x[i].x = silu(x[i].x) * y[i].x;
      x[i].y = silu(x[i].y) * y[i].y;
      x_vec.elts[i] = __float22half2_rn(x[i]);
    } else {
      x[i] = __bfloat1622float2(x_vec.elts[i]);
      y[i] = __bfloat1622float2(y_vec.elts[i]);
      x[i].x = silu(x[i].x) * y[i].x;
      x[i].y = silu(x[i].y) * y[i].y;
      x_vec.elts[i] = __float22bfloat162_rn(x[i]);
    }
  }
}

                        
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint32_t* input_offset_by_experts,
    uint32_t* output_scale_offset_by_experts,
    int32_t* mask,
    int n_experts,
    bool low_latency) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

                                
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
                                                                       
                                                                              
                                           
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

                                             
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
                                                                       
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

                                                                               
            
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
                                                                          
                                                              
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

                                      
#pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

                                   
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

                                    
                                                                        
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

                                                                      
                                                              
                                 
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    int factor = CVT_FP4_SF_VEC_SIZE * 4;
                                                                        
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

                        
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_expert(
#else
cvt_fp16_to_fp4_expert(
#endif
    int32_t numRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    int32_t* mask,
    bool use_silu_and_mul,
    int n_experts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

                                
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;
  int remainder = (gridDim.x * blockDim.x) % n_experts;
  int expert_idx;
  int tid_in_expert;
  int actual_stride;
  if (remainder > 0) {
    int bound = remainder * (stride + 1);
    if (tid < bound) {
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  int m = numRows / n_experts;
  int padded_m = (m + (128 - 1)) / 128 * 128;

  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
                                                                       
                                                                              
                                           
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_silu_and_mul ? colsPerRow * 2 : colsPerRow;

                                             
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow; globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
                                                                       
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

                                    
    int rowIdx_in_expert = rowIdx - expert_idx * m;

                                   
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      break;
    }

    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    if (use_silu_and_mul) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

                                    
                                                                        
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

                                                                      
                                                              
                                 
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    int factor = CVT_FP4_SF_VEC_SIZE * 4;
                                                                        
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;

    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

                                                                  
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint32_t* input_offset_by_experts,
    uint32_t* output_scale_offset_by_experts,
    int32_t* mask,
    int n_experts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");
  extern __shared__ uint32_t shared_input_offsets[];

                                           
                                                                             
                                                   
  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) = *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

                                             
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
                                                                       
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

                                                                               
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

                                                        
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
                                                            
                                   
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;

    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

template <typename T>
void quant_impl(
    void* output,
    void* output_scale,
    void* input,
    void* input_global_scale,
    void* input_offset_by_experts,
    void* output_scale_offset_by_experts,
    void* mask,
    bool use_silu_and_mul,
    int m_topk,
    int k,
    int n_experts,
    cudaStream_t stream) {
                                                     
  int device;
  cudaGetDevice(&device);
  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

                      
                                   
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
                                                                      
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x), multiProcessorCount * numBlocksPerSM));
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

                                                                   
  if (mask != nullptr) {
    grid.x = (grid.x + n_experts - 1) / n_experts * n_experts;
    cvt_fp16_to_fp4_expert<T, false><<<grid, block, 0, stream>>>(
        m_topk,
        k,
        reinterpret_cast<T*>(input),
        reinterpret_cast<float*>(input_global_scale),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_scale),
        reinterpret_cast<int32_t*>(mask),
        use_silu_and_mul,
        n_experts);
    return;
  }

  int const blockRepeat = (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  if (blockRepeat > 1) {
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    if (n_experts >= 4) {
      cvt_fp16_to_fp4<T, false, false><<<grid, block, shared_mem_size, stream>>>(
          m_topk,
          k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    } else {
      cvt_fp16_to_fp4<T, false, true><<<grid, block, shared_mem_size, stream>>>(
          m_topk,
          k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    }
  } else {
    if (n_experts >= 16) {
      cvt_fp16_to_fp4<T, false, false><<<grid, block, 0, stream>>>(
          m_topk,
          k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
                                 true);
    } else {
      cvt_fp16_to_fp4<T, false, true><<<grid, block, 0, stream>>>(
          m_topk,
          k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
                                 true);
    }
  }
}

                              
#undef CHECK_CONTIGUOUS
#undef CHECK_TH_CUDA
#undef CHECK_INPUT

                                                   
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

                                                      
constexpr auto HALF = at::ScalarType::Half;
constexpr auto BF16 = at::ScalarType::BFloat16;
constexpr auto FLOAT = at::ScalarType::Float;
constexpr auto INT = at::ScalarType::Int;
constexpr auto UINT8 = at::ScalarType::Byte;

void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version >= 100, "fp4_quant is only supported on sm100+");

  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts must be a CUDA tensor");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts must be a CUDA tensor");

  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
                                                                 
                                                                      
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  const int BLOCK_SIZE = 16;
  auto m_topk = input.size(0);
  auto k = input.size(1);
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
                                                     
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
                                                   
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,         
        false,                       
        m_topk,
        k,
        n_experts,
        stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,         
        false,                       
        m_topk,
        k,
        n_experts,
        stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}

void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& mask,
    bool use_silu_and_mul) {
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version >= 100, "fp4_quant is only supported on sm100+");

  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(mask, "mask must be a CUDA tensor");

  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);

  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(mask.scalar_type() == INT);
                                                                 
                                                                      
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  const int BLOCK_SIZE = 16;
  auto m_topk = input.size(0);
  auto k_by_2 = input.size(1);
  auto k = k_by_2;
  if (use_silu_and_mul) {
    TORCH_CHECK(k_by_2 % 2 == 0, "k must be a multiple of 2");
    k = k_by_2 / 2;
  }
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(mask.size(0) == n_experts);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
                                                     
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
                                                   
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,                            
        nullptr,                                   
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk,
        k,
        n_experts,
        stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,                            
        nullptr,                                   
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk,
        k,
        n_experts,
        stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}
