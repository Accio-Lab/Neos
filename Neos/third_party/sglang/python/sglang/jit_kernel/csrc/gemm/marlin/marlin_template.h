/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */
#include <sgl_kernel/scalar_type.hpp>

#include "dequant.h"
#include "marlin.cuh"
#include "marlin_dtypes.cuh"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)                                        \
  static_assert(                                                                         \
      std::is_same<scalar_t, half>::value || std::is_same<scalar_t, nv_bfloat16>::value, \
      "only float16 and bfloat16 is supported");

namespace device::marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <
    typename scalar_t,                                                       
    const host::ScalarTypeId w_type_id,                         
    const int threads,                                                        
    const int thread_m_blocks,                                             
                                                                        
                                                       
    const int thread_n_blocks,                                           
    const int thread_k_blocks,                                              
    const bool m_block_size_8,                                       
                                                                                
    const int stages,                                                                    
                                                          
    const bool has_act_order,                                           
    const int group_blocks,                                                   
                                                                              
    const bool is_zp_float                                                
    >
__global__ void Marlin(
    const int4* __restrict__ A,                                            
    const int4* __restrict__ B,                                                       
    int4* __restrict__ C,                                                   
    int4* __restrict__ C_tmp,                                                   
    const int4* __restrict__ scales_ptr,                                      
                                                            
    const int* __restrict__ g_idx,                                         
    int num_groups,                                                                   
    int prob_m,                                               
    int prob_n,                                                
    int prob_k,                                                   
    int* locks,                                                                              
    bool use_fp32_reduce                                                      
) {}

}                             

#else

                                                                 
                       
template <typename scalar_t>
__device__ inline void
mma(const typename ScalarType<scalar_t>::FragA& a_frag,
    const typename ScalarType<scalar_t>::FragB& frag_b,
    typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

template <typename scalar_t>
__device__ inline void mma_trans(
    const typename ScalarType<scalar_t>::FragA& a_frag,
    const typename ScalarType<scalar_t>::FragB& frag_b,
    const typename ScalarType<scalar_t>::FragB& frag_b2,
    typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]),
          "r"(b2[0]),
          "r"(b[1]),
          "r"(b2[1]),
          "r"(a[0]),
          "r"(a[1]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]),
          "r"(b2[0]),
          "r"(b[1]),
          "r"(b2[1]),
          "r"(a[0]),
          "r"(a[1]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

                                                                                
                                          
template <int count, typename scalar_t>
__device__ inline void ldsm(typename ScalarType<scalar_t>::FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(a[0]) : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

                                                                            
                                 
template <typename scalar_t>
__device__ inline void
scale(typename ScalarType<scalar_t>::FragB& frag_b, typename ScalarType<scalar_t>::FragS& frag_s, int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s = ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t>
__device__ inline void scale_and_sub(typename ScalarType<scalar_t>::FragB& frag_b, scalar_t s, scalar_t zp) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s2 = ScalarType<scalar_t>::num2num2(s);
  scalar_t2 zp2 = ScalarType<scalar_t>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <typename scalar_t>
__device__ inline void
sub_zp(typename ScalarType<scalar_t>::FragB& frag_b, typename ScalarType<scalar_t>::scalar_t2& frag_zp, int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 zp = ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

                                                                       
template <typename scalar_t>
__device__ inline void scale4(
    typename ScalarType<scalar_t>::FragB& frag_b,
    typename ScalarType<scalar_t>::FragS& frag_s_1,
    typename ScalarType<scalar_t>::FragS& frag_s_2,
    typename ScalarType<scalar_t>::FragS& frag_s_3,
    typename ScalarType<scalar_t>::FragS& frag_s_4,
    int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

                                               
template <typename scalar_t>
__device__ inline void scale_float(float* c, typename ScalarType<scalar_t>::FragS& s) {
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], ScalarType<scalar_t>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], ScalarType<scalar_t>::num2float(s_ptr[1]));
}

                                                                         
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
                                                                             
                  
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

                                                  
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
                                                                         
                                             
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

                                                          
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
                                                                             
                  
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <
    typename scalar_t,                                                       
    const host::ScalarTypeId w_type_id,                         
    const int threads,                                                        
    const int thread_m_blocks,                                             
                                                                        
                                                       
    const int thread_n_blocks,                                           
    const int thread_k_blocks,                                              
    const bool m_block_size_8,                                       
                                                                                
    const int stages,                                                                    
                                                          
    const int group_blocks,                                                   
                                                                              
    const bool is_zp_float                                                
    >
__global__ void Marlin(
    const int4* __restrict__ A,                                                
    const int4* __restrict__ B,                                                           
    int4* __restrict__ C,                                                       
    int4* __restrict__ C_tmp,                                                       
    const int4* __restrict__ scales_ptr,                                          
                                                                
    const uint16_t* __restrict__ scale2_ptr,                                 
                                                      
    const int4* __restrict__ zp_ptr,                                             
                                                                              
    const int* __restrict__ g_idx,                                             
    int num_groups,                                                                       
    int prob_m,                                                   
    int prob_n,                                                    
    int prob_k,                                                       
    int lda,                                                                                    
    int* locks,                                                                                  
    bool use_atomic_add,                                                            
    bool use_fp32_reduce,                                                         
    int max_shared_mem) {
                                                                               
                                                                           
                                                                            
             
            
            
            
                                                                               
                                                                        
                                                                         
                            
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  using FragA = typename ScalarType<scalar_t>::FragA;
  using FragB = typename ScalarType<scalar_t>::FragB;
  using FragC = typename ScalarType<scalar_t>::FragC;
  using FragS = typename ScalarType<scalar_t>::FragS;
  using FragZP = typename ScalarType<scalar_t>::FragZP;

  static constexpr auto w_type = host::ScalarType::from_id(w_type_id);
  constexpr bool has_zp = w_type == host::kU4 || w_type == host::kU8;
  constexpr bool is_int_type =
      w_type == host::kU4 || w_type == host::kU8 || w_type == host::kU4B8 || w_type == host::kU8B128;
                                               
  constexpr bool dequant_skip_flop = !is_int_type ||
                                     has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
                                     has_zp && !is_zp_float && !(w_type == host::kU8);

  scalar_t2 global_scale;

  if constexpr (w_type == host::kFE2M1f) {
    uint16_t val = scale2_ptr[0];
    global_scale = Dtype::num2num2(*reinterpret_cast<scalar_t*>(&val));
  }

  constexpr bool has_act_order = group_blocks == 0;
  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);

  constexpr int pack_factor = 32 / w_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);

                                                                             
                                             
  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
                                                                            
                                                                              
                                
      iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;                                                         
  int slice_count = 0;                                                             
  int slice_idx;                                                                    
                              

  int par_id = 0;
  int locks_off = 0;

                                                                         
                                          
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * lda / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    slice_col = slice_col_par % n_tiles;
    par_id = slice_col_par / n_tiles;
  }
  if (parallel * n_tiles >= gridDim.x) {
                                     
                                                        
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

                                                                          
                     
  auto init_slice = [&](bool first_init = false) {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (parallel * n_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread = div_ceil(thread_m_blocks * 16, threads / threads_per_m);
      if (m_block_size_8) m_per_thread = div_ceil(8, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < prob_m) {
          int col = slice_col * 16 * thread_n_blocks / 8 + threadIdx.x % threads_per_m;
          C[row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
                                                                    
                                                              
                                                        
                                                                           
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * lda / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
  };
  init_slice(true);

                    

                                            
  int a_gl_stride = lda / 8;
                                                
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
                                                      
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
                                              
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
                                 
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
                                     
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
                                
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
                           
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
                                                 
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

                    
  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = w_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

                                          
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_tb_groups = !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks
                                  ? thread_k_blocks / group_blocks / (w_type == host::kFE2M1f ? 2 : 1)
                                  : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

                                      
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
                                             
                                                                         
  constexpr int act_s_max_num_groups = 32;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;

  int tb_n_warps = thread_n_blocks / 4;
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

                              
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float ? 16 * thread_n_blocks / 8 : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

                                           
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
                                          
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
                       
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
                (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_wr = threadIdx.x * b_thread_vecs;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;

                  
  constexpr int k_iter_size = tb_k / b_sh_wr_iters;
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

                 
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
    } else {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) / (w_type == host::kFE2M1f ? 2 : 1) +
                s_sh_stride * slice_col + threadIdx.x;
    }
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

                
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + zp_sh_stride * slice_col + threadIdx.x;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = threadIdx.x < zp_sh_stride;

                                                                                
                                                                        
                                  
  int s_sh_rd;
  if constexpr (group_blocks != -1 && w_type == host::kFE2M1f) {
    auto warp_id = threadIdx.x / 32;
    int n_warps = thread_n_blocks / 4;
    int warp_row = warp_id / n_warps;

    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
    s_sh_rd = s_sh_rd * 2 + warp_row % 2;

  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 && (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

                                                        
                               
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
      }
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

                                                                                
                                                                             
                                                
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

                                                                          
                                                                             
                                                                          
                                                                             
                                                                               
                                                            
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
                                                                                
                                                                              
                                       
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

                                                                          
                                                                              
                                                                    
                  
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
                                                      
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_g_idx = sh_b + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
  constexpr int sh_s_size = has_act_order ? (act_s_max_num_groups * s_sh_stride) : (stages * s_sh_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
                                                             
                                  
  static_assert(thread_m_blocks * 16 * thread_n_blocks * 16 / 8 <= stages * b_sh_stage);
  int4* sh_a = sh_s + sh_s_size;
                                  
                                                           
                                                             

                                                               
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];                                   
  FragS act_frag_s[2][4][4];                             
  int frag_qzp[2][num_ints_per_thread];                
  FragZP frag_zp;                                              
  FragZP frag_zpf[2];                                                 

                       
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;

  auto fetch_act_order_scales_to_shared = [&](bool is_async, int first_group_id, int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups > act_s_max_num_groups) {
      sh_num_groups = act_s_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(
              &sh_s[(i * s_sh_stride) + threadIdx.x],
              &scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x];
        }
      }
    }
  };
                                                                          
                                     
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);
        }

        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (has_act_order) {
                                           
        int full_pipe = a_off;
        int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
        if (cur_k < prob_k && cur_k < slice_k_finish) {
          int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

          int4 const* cur_g_idx_stage_ptr = reinterpret_cast<int4 const*>(&g_idx[cur_k]);

          if (threadIdx.x < g_idx_stage) {
            cp_async4_pred(&sh_g_idx_stage[threadIdx.x], &cur_g_idx_stage_ptr[threadIdx.x]);
          }
        }
      } else {
        if constexpr (group_blocks != -1) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
                                                                
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < s_tb_groups; i++) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr], &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          }
        }

        if constexpr (has_zp && group_blocks != -1) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
                                                                     
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < zp_tb_groups; i++) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[i * zp_sh_stride + zp_sh_wr], &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          }
        }
      }
    }
                                                                               
                                             
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

                                                                      
  auto wait_for_stage = [&]() {
                                                                             
                                                                                
                                                                   
                    
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

                                                                               
                                      
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, scalar_t>(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

#pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (!has_act_order) {
                          
      if constexpr (group_blocks == -1) {
                                              
        if (k == 0 && full_pipe == 0) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      } else if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_s_stage =
                sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
          } else {
            reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          int cur_group_id = k_blocks / (group_blocks * (w_type == host::kFE2M1f ? 2 : 1));

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (w_type_id != host::kFE2M1f.id()) {
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
          } else {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
          }
        }
      }

      return;
    }

                     

                                                
    int cur_k = slice_k_start + tb_k * full_pipe;
    if (cur_k >= prob_k || cur_k >= slice_k_finish) {
      return;
    }

                                                                           
                    
    cur_k = 0;

                                    
    cur_k += k_iter_size * (k % b_sh_wr_iters);

                                                                      
                 
    auto warp_id = threadIdx.x / 32;
    int n_warps = thread_n_blocks / 4;                                               

    int warp_row = warp_id / n_warps;
    int warp_col = warp_id % n_warps;

    cur_k += warp_row * 16;

    auto th_id = threadIdx.x % 32;
    cur_k += (th_id % 4) * 2;                                                

    int s_col_shift =
                             (act_s_col_warp_stride * warp_col) + (th_id / 4) * act_s_col_stride;

    if (is_same_group[pipe]) {
      if (k % 2 == 0) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride + s_col_shift];
      } else {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
      }

      for (int i = 1; i < 4; i++) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) = *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
      }
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    constexpr int k_frag_offsets[4] = {0, 1, 8, 9};                                   

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int actual_k = cur_k + k_frag_offsets[i];

      int group_id = sh_g_idx_int_ptr[actual_k];
      int rel_group_id = group_id - sh_first_group_id;

      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) = sh_s[rel_group_id * s_sh_stride + s_col_shift];
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
                                                   
                                 
                                                        
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
                                              
        if (k == 0 && full_pipe == 0) {
#pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }

      } else if constexpr (group_blocks >= thread_k_blocks) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_zp_stage =
              sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
#pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks = cur_k / 16;
        int cur_group_id = 0;

                                                               
#pragma nv_diagnostic push
#pragma nv_diag_suppress divide_by_zero
        cur_group_id = k_blocks / group_blocks;
#pragma nv_diagnostic pop

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

#pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage =
                sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] = sh_zp_stage[zp_sh_rd];
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
                                                                 
#pragma nv_diagnostic push
#pragma nv_diag_suppress divide_by_zero
          int cur_group_id = k_blocks / group_blocks;
#pragma nv_diagnostic pop

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] = sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_t2* frag_b_ptr) {
    dequant<scalar_t2, w_type_id, dequant_skip_flop>(q, frag_b_ptr);
  };

                                                         
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k) {
    int k2 = k % 2;
    const bool is_new_zp = ((group_blocks != -1) && (group_blocks < thread_k_blocks || k == 0)) ||
                           (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        FragB frag_zp_0;
        FragB frag_zp_1;
        int zp_quant_0, zp_quant_1;

        if constexpr (w_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(w_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_t2*>(&frag_zp));
        dequant_data(zp_quant_1, reinterpret_cast<scalar_t2*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] = reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (w_type == host::kFE2M1f) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<scalar_t2>(s_quant_0, reinterpret_cast<scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<scalar_t2>(s_quant_1, reinterpret_cast<scalar_t2*>(&frag_s[k2]) + 2);
    }

                                                                              
                                        
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (w_type_id == host::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (w_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(w_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float) {
        sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
        sub_zp<scalar_t>(frag_b1, frag_zp[j], 1);
      }

                               
      if constexpr (has_act_order) {
        static_assert(group_blocks != -1);
        scale4<scalar_t>(
            frag_b0, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j], act_frag_s[k2][3][j], 0);
        scale4<scalar_t>(
            frag_b1, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j], act_frag_s[k2][3][j], 1);
      } else if constexpr (!dequant_skip_flop && has_zp && !is_zp_float && group_blocks == -1) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Dtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<scalar_t>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1) {
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<scalar_t>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1) {
        scale<scalar_t>(frag_b0, frag_s[k2][j], 0);
        scale<scalar_t>(frag_b1, frag_s[k2][j], 1);
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<scalar_t>(frag_a[k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
        } else {
          mma<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
          mma<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
        }
      }
    }
  };

                                                                             
                                                                                
                                                                         
                                                                               
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);

                                                                                
                                                                                
                                                     

#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

                                                                               
                                                                     
                                                                             
                                                                          
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
                                                                          
                                                                           
                                                            
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr;
      if constexpr (m_block_size_8) {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) + 4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      } else {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      }
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
                                                                           
                                                                          
                                                      
#pragma unroll
        for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
          if constexpr (m_block_size_8) {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i],
                (threadIdx.x % 4) * 2 + i < prob_m);
          } else {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          }
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

#pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        bool mask = (!m_block_size_8) && (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) ||
                    (m_block_size_8) && ((threadIdx.x % 4) * 2 + i < prob_m);
        if (mask) {
          if (!first) {
            int4 c_red = sh_red[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta] +=
                  Dtype::num2float(reinterpret_cast<scalar_t*>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<scalar_t*>(&c)[j] =
                  Dtype::float2num(reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta]);
            }
            if constexpr (m_block_size_8)
              C[c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i] = c;
            else
              C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

                                                                          
                                                            
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * thread_n_blocks / 4;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = locks_off * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] = C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
#pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

                                                                              
                                                                              
                        
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) + (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += 32 * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;
                                                                              
                            
    auto write = [&](int idx, float c0, float c1, FragS& s) {
      scalar_t2 res = Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

                                                                              
               
      if constexpr (
          !has_act_order && group_blocks == -1 && w_type.size_bits() == 4 && (has_zp && dequant_skip_flop || !has_zp)) {
        res = __hmul2(res, s[0]);
      }

      if constexpr (w_type == host::kFE2M1f) {
        res = __hmul2(res, global_scale);
      }

      if constexpr (m_block_size_8) {
        ((scalar_t*)sh_red)[idx] = res.x;
        ((scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(
                wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
            write(
                wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
            write(
                wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
            write(
                wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        if (use_atomic_add && slice_count > 1) {
          scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[c_gl_wr]);
          scalar_t2* sh_red_half2 = reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
#pragma unroll
          for (int a = 0; a < 4; a++) {
            atomicAdd(&C_half2[a], sh_red_half2[a]);
          }
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

                                                    
  auto start_pipes = [&]() {

#pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if (has_act_order && i == 0) {
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        fetch_act_order_scales_to_shared(true, g_idx[slice_k_start], g_idx[last_g_idx]);
      }

      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    if constexpr (has_act_order) {
      slice_k_start_shared_fetch += tb_k * (stages - 1);
    }
  };
  if (slice_iters) {
    start_pipes();
  }

               
  while (slice_iters) {
                                                                             
                                                                             
                                                                            
               

#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
          init_same_group(pipe % stages);
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    if constexpr (has_act_order) {
      slice_k_start += tb_k * stages;

      if (slice_k_start < prob_k) {
        slice_k_start_shared_fetch += tb_k * stages;
        int first_group_id = g_idx[slice_k_start];
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        int last_group_id = g_idx[last_g_idx];
        if (last_group_id >= sh_first_group_id + sh_num_groups) {
          fetch_act_order_scales_to_shared(false, first_group_id, last_group_id);
          __syncthreads();
        }
      }
    }

                                                                           
                                                                             
                                                                         
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
                                                                                
                  
      if constexpr (!has_act_order && group_blocks == -1 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();
      if constexpr (!has_act_order && group_blocks == -1 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              scalar_t2* frag_s_half2 = reinterpret_cast<scalar_t2*>(frag_s);
#pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Dtype::num2num2(reinterpret_cast<scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

                                                                              
                                                                          
                          
      if constexpr (
          !has_act_order && group_blocks == -1 && w_type.size_bits() == 8 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][0][0]), frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]), frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][0]), frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][2]), frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {
                                                                          
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }
      if (use_atomic_add && slice_count > 1 && slice_idx != 0) wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
                                                                    
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      is_first_matmul_in_slice = true;
      init_slice();

      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }

                                              
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;

        } else {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        }

        start_pipes();
      }
    }
  }
}

}                             

#endif
