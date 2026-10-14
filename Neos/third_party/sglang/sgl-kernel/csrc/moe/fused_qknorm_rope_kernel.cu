/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/cuda.h>

#include <cmath>

#define CHECK_TYPE(x, st) \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st) \
  CHECK_TH_CUDA(x);        \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)

#define FINAL_MASK 0xffffffff

namespace tensorrt_llm::common {
template <typename T, int num>
struct packed_as;

                                                    
template <>
struct packed_as<uint, 1> {
  using type = uint;
};

template <>
struct packed_as<uint, 2> {
  using type = uint2;
};

template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask,
                           32);                                              
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}                                   
namespace tensorrt_llm::kernels {

__device__ inline float compute_freq_yarn(float base, int head_dim, int half_dim, float factor, float low, float high) {
  float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (fabsf(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float linear_func = (static_cast<float>(half_dim) - low) / (high_adj - low);
    float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float inv_freq_extrapolation_factor = 1.0f - ramp_func;

    freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor) +
           inv_freq_extrapolation * inv_freq_extrapolation_factor;
  }

  return freq;
}

                                                                                                    

                                                        
                                       
                                   
template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,                                                                                                
    int const num_heads_q,                                  
    int const num_heads_k,                                
    int const num_heads_v,                                  
    float const eps,                                                
    __nv_bfloat16 const* q_weight,                              
    __nv_bfloat16 const* k_weight,                            
    float const base,                                           
    int const* position_ids,                                
    int const num_tokens,                              
                          
    float factor,                                                                                                 
    float low,                                    
    float high,                                  
    float attention_factor,                                            
    int const rotary_dim) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

                                                                                  
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

                                              
  int const total_qk_heads = num_heads_q + num_heads_k;

                                                                     
  int const tokenIdx = globalWarpIdx / total_qk_heads;
  int const localHeadIdx = globalWarpIdx % total_qk_heads;

                                                              
  if (tokenIdx >= num_tokens) return;

  bool const isQ = localHeadIdx < num_heads_q;
  int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
  int const num_heads = num_heads_q + num_heads_k + num_heads_v;
  static_assert(
      head_dim % (32 * 2) == 0,
      "head_dim must be divisible by 64 (each warp processes one head, and each thread gets even number of "
      "elements)");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "numSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;                                                            
  using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

  int offsetWarp;                        
  if (isQ) {
                                                             
    offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
  } else {
                                                                                
    offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
  }
  int offsetThread = offsetWarp + laneId * numElemsPerThread;

                               
  float sumOfSquares = 0.0f;

          
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
      sumOfSquares += vals.x * vals.x;
      sumOfSquares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

                                                      
  sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);

                                     
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

                       
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    float weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }
                                      
  float elements2[numElemsPerThread];                                         
  float cos_vals[numElemsPerThread];
  float sin_vals[numElemsPerThread];
  float pos_id = static_cast<float>(position_ids[tokenIdx]);
  int const rotary_lanes = rotary_dim / numElemsPerThread;                 
  bool const applyRotary = (laneId < rotary_lanes);
  if (applyRotary) {
    if constexpr (interleave) {
                                                          
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];

        int dim_idx = laneId * numElemsPerThread + i;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }

    } else {
                   
                                                            
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
                                                                                       
                                                                             
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          elements2[i] = -elements2[i];
        }

        int dim_idx = laneId * numElemsPerThread + i;
        dim_idx = (dim_idx * 2) % rotary_dim;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }
                                                                       
      __syncwarp();
    }

    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
    }
  }

           
  {
    vec_T vec;
    for (int i = 0; i < vecSize; i++) {
      __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
      reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
    }
    vec_T* outputPtr = reinterpret_cast<vec_T*>(&qkv[offsetThread]);
    *outputPtr = vec;
  }
}
                
                                                                                                                                
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }
void launchFusedQKNormRope(
    void* qkv,
    int const num_tokens,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    int const head_dim,
    float const eps,
    void const* q_weight,
    void const* k_weight,
    float const base,
    bool const interleave,
    int const* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int const rotary_dim,
    cudaStream_t stream) {
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;
  int const gridSize = common::divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);
                                               
                             
  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<64, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor,
            rotary_dim);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<128, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor,
            rotary_dim);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<256, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor,
            rotary_dim);
      });
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fusedQKNormRope: ", head_dim);
  }
}
}                                    

                                      
                                                                                               
                                                               
void fused_qk_norm_rope(
    torch::Tensor& qkv,                                                                                              
    int64_t num_heads_q,                                  
    int64_t num_heads_k,                                
    int64_t num_heads_v,                                  
    int64_t head_dim,                                  
    double eps,                                                   
    torch::Tensor& q_weight,                                             
    torch::Tensor& k_weight,                                           
    double base,                                              
    bool is_neox,                                                         
    torch::Tensor& position_ids,                                       
                          
    double factor,                                                                                                 
    double low,                                    
    double high,                                  
    double attention_factor,                                            
    int64_t rotary_dim) {
                     
  TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
  TORCH_CHECK(rotary_dim % (head_dim / 32) == 0, "rotary_dim must be divisible by numElemsPerThread");
  if (is_neox) {
    int64_t half_rotary_lanes = rotary_dim / (head_dim / 32) / 2;
    TORCH_CHECK(
        half_rotary_lanes >= 1 && (half_rotary_lanes & (half_rotary_lanes - 1)) == 0,
        "half_rotary_lanes must be a power of 2 for neox style, got ",
        half_rotary_lanes);
  }
  CHECK_INPUT(qkv, torch::kBFloat16);
  CHECK_INPUT(position_ids, torch::kInt32);
  CHECK_INPUT(q_weight, torch::kBFloat16);
  CHECK_INPUT(k_weight, torch::kBFloat16);

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  tensorrt_llm::kernels::launchFusedQKNormRope(
      reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
      static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q),
      static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v),
      static_cast<int>(head_dim),
      static_cast<float>(eps),
      reinterpret_cast<__nv_bfloat16*>(q_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k_weight.data_ptr()),
      static_cast<float>(base),
      !is_neox,               
      reinterpret_cast<int const*>(position_ids.data_ptr()),
      static_cast<float>(factor),
      static_cast<float>(low),
      static_cast<float>(high),
      static_cast<float>(attention_factor),
      static_cast<int>(rotary_dim),
      stream);
}
