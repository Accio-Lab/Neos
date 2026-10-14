#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

static constexpr int kWarpSize = 32;
static constexpr int DEFAULT_SHARED_MEM_THRESHOLD_KB = 48;                                      

                                                                              
                                                
                                   
                                        
                                                                   
                                                                              
template <typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16, bool USE_SMEM = true>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;                         
  const int lane_id = threadIdx.x & (kWarpSize - 1);         
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

                                  
  const T* token_input = input + token_id * hidden_dim;
  DST_DTYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  extern __shared__ char smem_buffer[];
  const int smem_padding = 32;                                                          
  const int warp_smem_stride = (hidden_dim * sizeof(T) + smem_padding - 1) / smem_padding * smem_padding;
  const int warp_smem_offset = warp_id * warp_smem_stride;
  T* shared_input = reinterpret_cast<T*>(smem_buffer + warp_smem_offset);

    
                                            
    
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

                                              
    if constexpr (USE_SMEM) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        shared_input[i * kVecSize + j] = input_vec[j];
      }
    }

                                    
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

                                                                          
  if constexpr (USE_SMEM) {
    __syncwarp();
  }

  float warp_max = warpReduceMax(max_value);

                                                                               
                                                                                
  const float scale = warp_max / FP8_E4M3_MAX;
                    
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    
                                    
    
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;

    if constexpr (USE_SMEM) {
                                
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = shared_input[i * kVecSize + j];
      }
    } else {
                                  
      input_vec.cast_load(token_input + i * kVecSize);
    }

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
                                                                      
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

                                                                              
                                                        
                                                                              
template <typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  DST_DTYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

                                           
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

                                    
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = 1.0f / scale;

                                    
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }

    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
                                                                      
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

template <bool USE_SMEM, typename scalar_t, int TOKENS_PER_CTA>
static inline void launch_per_token_quant_fp8_warp_kernel(
    const dim3& grid,
    const dim3& block,
    size_t dynamicSmemSz,
    cudaStream_t stream,
    bool use_vec16,
    bool use_vec8,
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const size_t smem_size = USE_SMEM ? dynamicSmemSz : 0;

  if (use_vec16) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 16, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  } else if (use_vec8) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 8, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  } else {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 4, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  }
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];
  TORCH_CHECK(hidden_dim % 4 == 0, "Hidden dimension must be divisible by 4, but got ", hidden_dim);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int TOKENS_PER_CTA = 8;
  const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
  const bool use_vec16 = (hidden_dim % 16 == 0);
  const bool use_vec8 = (hidden_dim % 8 == 0);

  const int sizeof_T = input.scalar_type() == torch::kFloat16 ? 2 : (input.scalar_type() == torch::kBFloat16 ? 2 : 4);
  const int smem_padding = 32;                                            
  const int warp_smem_stride = (hidden_dim * sizeof_T + smem_padding - 1) / smem_padding * smem_padding;
  const size_t dynamicSmemSz = warp_smem_stride * TOKENS_PER_CTA;

  bool use_smem = (hidden_dim < 2048);

  if (dynamicSmemSz >= DEFAULT_SHARED_MEM_THRESHOLD_KB) {
    use_smem = false;                                                                  
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_warp_kernel) {
                                                                                
      constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;
      dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
      dim3 block(THREADS);

      if (use_smem) {
        launch_per_token_quant_fp8_warp_kernel<             true, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      } else {
        launch_per_token_quant_fp8_warp_kernel<             false, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      }
    } else {
                                                                                
      constexpr int THREADS = 256;
      dim3 grid(num_tokens);
      dim3 block(THREADS);

      if (use_vec16) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 16><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else if (use_vec8) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 8><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 4><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      }
    }
    return true;
  });
}
