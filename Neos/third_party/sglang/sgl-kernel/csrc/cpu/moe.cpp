#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

                                    
  
                                           
                               
                    
  
                                                              
                                                            
                                                     
                                                                  
                                                                 
                                                                
                                                                    
                                         
  
         
                                                       
                                                         
                                                                             
  

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec data_vec(val);
  at::vec::map<scalar_t>([data_vec](Vec out) { return out = data_vec; }, out, out, size);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
               
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

                            
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
                           
    copy_stub(out, input, K);
  } else {
                           
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

                               
template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts

                                           
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);

    for (int i = begin; i < end; ++i) {
      local_cnts[topk_ids[i]]++;
    }
  });

  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, T_INDEX(t + 1), T_INDEX(t + 1), T_INDEX(t), num_experts);
  }

                                            
  int32_t* total_cnts_t_1 = T_INDEX(num_threads);

  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
                                                                  
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;

    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
                                         
    int32_t* __restrict__ offsets = T_INDEX(tid);

    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      offsets[expert_id]++;
    }
  });

                                                                
  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e) {
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);
  }

                                        
  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };

                                                           
                                     
  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
    for (int mb = begin; mb < end; ++mb) {
      offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
    }
  });
                                         
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }
                                                       
  TORCH_CHECK(offsets[num_token_blocks] == numel);

  return num_tokens_post_pad;
}

                                               
                                        
                                        
                                    
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ input0,              
    const float* __restrict__ input1,              
    int64_t m_size,
    int64_t N) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const fVec one = fVec(1.f);

                 
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ x = input0 + m * BLOCK_N;
    const float* __restrict__ y = input1 + m * BLOCK_N;

    for (int64_t d = 0; d < BLOCK_N; d += bVec::size()) {
      fVec x0 = fVec::loadu(x + d);
      fVec x1 = fVec::loadu(x + d + fVec::size());
      fVec y0 = fVec::loadu(y + d);
      fVec y1 = fVec::loadu(y + d + fVec::size());
             
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
            
      x0 = x0 * y0;
      x1 = x1 * y1;
                
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(out + d);
    }
  }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B0,
      const scalar_t* __restrict__ B1,
      scalar_t* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B0,
      const at::BFloat16* __restrict__ B1,
      at::BFloat16* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    static_assert(COLS % 2 == 0);

                        
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb0[COLS];
    __m512bh vb1[COLS];
    __m512 vc0[ROWS * COLS];
    __m512 vc1[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc0[i] = _mm512_set1_ps(0.f);
      vc1[i] = _mm512_set1_ps(0.f);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;                  
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b0_ptr = reinterpret_cast<const float*>(B0);
    const float* b1_ptr = reinterpret_cast<const float*>(B1);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb0[col] = (__m512bh)(_mm512_loadu_si512(b0_ptr + k * ldb2 + col * 16));
        vb1[col] = (__m512bh)(_mm512_loadu_si512(b1_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b0_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          _mm_prefetch(b1_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc0[i] = _mm512_dpbf16_ps(vc0[i], va, vb0[col]);
      vc1[i] = _mm512_dpbf16_ps(vc1[i], va, vb1[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    using Vec = at::vec::Vectorized<float>;
    const Vec one = Vec(1.f);
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
                                         
      if constexpr (col % 2 == 0) {
        Vec x0 = vc0[row * COLS + col + 0];
        Vec x1 = vc0[row * COLS + col + 1];
        Vec y0 = vc1[row * COLS + col + 0];
        Vec y1 = vc1[row * COLS + col + 1];
               
        x0 = x0 / (one + x0.neg().exp_u20());
        x1 = x1 / (one + x1.neg().exp_u20());
              
        x0 = x0 * y0;
        x1 = x1 * y1;

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(__m512(x1), __m512(x0))));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)       \
  tinygemm_kernel_nn2<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B0 + nb_start * 2, B1 + nb_start * 2, C + mb_start * ldc + nb_start, K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B0,
    const scalar_t* __restrict__ B1,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
                           
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
                      
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
                      
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
                      
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
                      
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    static_assert(COLS % 2 == 0);

                        
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_ps(0.f); };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;                  
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN2(MB_SIZE, NB_SIZE)     \
  tinygemm_kernel_nn<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
                   
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
                      
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 32);
          break;
                      
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN2(2, 32);
          break;
                      
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN2(3, 32);
          break;
                      
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN2(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
                             
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

                                                            
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

                               
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  const bool use_brgemm = can_use_brgemm<scalar_t>(avg_M);

                                                                       
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
                         
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
                                                             
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);

                                           
      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb_upper * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb_lower * BLOCK_N * stride_n;

                   
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      if (use_brgemm) {
                                
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B0,
                        C0);

                                
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B1,
                        C1);

                           
        const int64_t offset = offsets[mb];
        silu_and_mul<scalar_t, BLOCK_N>(ic1 + offset * N + nb * BLOCK_N, C0, C1, m_size, N);
      } else {
                                                    
        const int64_t offset = offsets[mb];
        tinygemm_kernel(
                        A,
                        B0,
                        B1,
                        ic1 + offset * N + nb * BLOCK_N,
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

                                                            
                                    
  const int64_t OC = K;                   
  const int64_t IC = N;                   
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_oc = IC;

                           
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
                         
    int tid = get_thread_num();
                                     
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

                                                        
                                                  
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

                                            
      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;

                            
      if (use_brgemm) {
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        IC,
                        IC,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B,
                        C);
      } else {
        tinygemm_kernel(
                        A,
                        B,
                        C,
                        m_size,
                        n_size,
                        IC,
                        IC,
                        n_size,
                        BLOCK_N);
      }

                                                 
                                               
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

                                                  
                                  
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

template <typename scalar_t>
void shared_expert_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
                             
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

                                                            
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);
  const int64_t stride_n = K;

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

                                                                       
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
                         
    int tid = get_thread_num();
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
                                                             
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);

                            
      const scalar_t* A = input + mb * BLOCK_M * K;

                                           
      const scalar_t* __restrict__ B0 = packed_w1 + nb_upper * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B1 = packed_w1 + nb_lower * BLOCK_N * stride_n;

      if (use_brgemm) {
                                
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B0,
                        C0);

                                
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B1,
                        C1);

                           
        silu_and_mul<scalar_t, BLOCK_N>(ic1 + mb * BLOCK_M * N + nb * BLOCK_N, C0, C1, m_size, N);
      } else {
                                                    
        tinygemm_kernel(
                        A,
                        B0,
                        B1,
                        ic1 + mb * BLOCK_M * N + nb * BLOCK_N,
                        m_size,
                        n_size,
                        K,
                        K,
                        n_size,
                        N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

                                               
                              
  const int64_t OC = K;                   
  const int64_t IC = N;                   
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = IC;

                           
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
                         
    int tid = get_thread_num();
                                     
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

                             
      const scalar_t* __restrict__ A = ic1 + mb * BLOCK_M * N;

                                            
      const scalar_t* __restrict__ B = packed_w2 + nb * BLOCK_N * stride_oc;

                            
      if (use_brgemm) {
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        IC,
                        IC,
                        n_size,
                        BLOCK_N,
                        false,
                        A,
                        B,
                        C);
      } else {
        tinygemm_kernel(
                        A,
                        B,
                        C,
                        m_size,
                        n_size,
                        IC,
                        IC,
                        n_size,
                        BLOCK_N);
      }

                                                            
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N;
      for (int64_t m = 0; m < m_size; ++m) {
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}                        

                
static inline void check_moe_scales(
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size) {
  if (use_int8_w8a8) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for int8 w8a8.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for int8 w8a8.");
  }
  if (use_fp8_w8a16) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for fp8 w8a16.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for fp8 w8a16.");
    TORCH_CHECK(block_size.has_value(), "missing block_size for fp8 w8a16.");
    TORCH_CHECK(block_size.value().size() == 2, "expect block_size.size() to be 2.");
  }
}

#define CHECK_MOE_SCALES_FP8(DIM0, DIM1)                      \
  auto w1s = w1_scale.value();                                \
  auto w2s = w2_scale.value();                                \
  auto block_size_val = block_size.value();                   \
  int64_t block_size_N = block_size_val[0];                   \
  int64_t block_size_K = block_size_val[1];                   \
  TORCH_CHECK(w1s.size(DIM0) == div_up(2 * N, block_size_N)); \
  TORCH_CHECK(w1s.size(DIM1) == div_up(K, block_size_K));     \
  TORCH_CHECK(w2s.size(DIM0) == div_up(K, block_size_N));     \
  TORCH_CHECK(w2s.size(DIM1) == div_up(N, block_size_K))

                        
                 
                
                          
                                
  

at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    int64_t moe_comp_method,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w1_zero,
    const std::optional<at::Tensor>& w2_zero,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl-kernel::fused_experts_cpu", std::vector<c10::IValue>({hidden_states, w1, w2, topk_weights, topk_ids}));

  auto packed_w1 = is_vnni ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_vnni ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());
  CHECK_DIM(2, hidden_states);
  if (moe_comp_method == CPUQuantMethod::INT4_W4A8 && is_vnni) {
    CHECK_DIM(4, w1);
    CHECK_DIM(4, w2);
  } else {
    CHECK_DIM(3, w1);
    CHECK_DIM(3, w2);
  }
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);

  CHECK_EQ(topk_ids.scalar_type(), at::kInt);

                                                                 
                                                                                                  
                                                              
  auto topk_weights_ = topk_weights.to(at::kFloat);
  CHECK_EQ(topk_weights_.scalar_type(), at::kFloat);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = moe_comp_method == CPUQuantMethod::INT4_W4A8 ? w1_scale.value().size(1) * w1_scale.value().size(3) / 2
                                                           : w1.size(1) / 2;
  int64_t E = w1.size(0);
  int64_t topk = topk_weights_.size(1);

                                              
  int64_t packed_K = get_row_size(K, moe_comp_method == CPUQuantMethod::INT8_W8A8);
  int64_t packed_N = get_row_size(N, moe_comp_method == CPUQuantMethod::INT8_W8A8);

                        
  CHECK_EQ(w2.size(0), E);
  if (!(moe_comp_method == CPUQuantMethod::INT4_W4A8)) {
    CHECK_EQ(w2.size(1), K);
    CHECK_EQ(packed_w1.size(2), packed_K / (moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 2 : 1));
    CHECK_EQ(packed_w2.size(2), packed_N / (moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 2 : 1));
  }
                 
  check_moe_scales(
      moe_comp_method == CPUQuantMethod::INT8_W8A8,
      moe_comp_method == CPUQuantMethod::FP8_W8A16,
      w1_scale,
      w2_scale,
      block_size);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

                                                                    
                                                     
                                       
                                 
                              
                                           
    
  int num_threads = at::get_num_threads();
  int64_t max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int64_t max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  auto buffer = at::empty(
      {max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E + (E + 1) + (max_num_blocks + 1)},
      topk_ids.options());

  int32_t* __restrict__ sorted_ids = buffer.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);

                                                       
                                       
  int64_t numel = M * topk;
  at::parallel_for(0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int64_t begin, int64_t end) {
    int64_t m_start = begin * BLOCK_M;
    int64_t m_size = std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
    fill_stub(sorted_ids + m_start, (int32_t)numel, m_size);
    fill_stub(expert_ids + begin, (int32_t)E, end - begin);
  });
                                
  at::parallel_for(0, (num_threads + 1) * E + (E + 1), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    fill_stub(total_cnts + begin, 0, end - begin);
  });

                        
  int64_t num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
      sorted_ids, expert_ids, topk_ids.data_ptr<int32_t>(), total_cnts, cumsums, offsets, E, numel, num_threads);

                                                                                      
                                             
                                             
                                  
                                            
    
                   
                                          
                             
    
                   
                                              
                                                                    
    
  int64_t buffer_size_nbytes =
      M * topk * N * 2 + M * topk * K * 2 +
      num_threads * BLOCK_M * K *
          (moe_comp_method == CPUQuantMethod::INT8_W8A8 | moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 1 : 2) +
      num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);

  if (moe_comp_method == CPUQuantMethod::INT8_W8A8) {
    buffer_size_nbytes += std::max(M * K, M * topk * N) + M * topk * sizeof(float);
  }
  if (moe_comp_method == CPUQuantMethod::FP8_W8A16) {
    buffer_size_nbytes += M * topk * 2 * N * 2 + num_threads * MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N) * 2;
  }
  if (moe_comp_method == CPUQuantMethod::INT4_W4A8) {
    buffer_size_nbytes += M * topk * 2 * N * 2 + std::max(M * K, M * topk * N) + M * topk * sizeof(float) +
                          num_threads * 2 * get_4bit_block_k_size(K / w1_scale.value().size(2)) * BLOCK_N;
  }
  auto buffer2 = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer2.data_ptr<int8_t>()));
    scalar_t* __restrict__ intermediate_cache2 = intermediate_cache1 + M * topk * N;

    if (moe_comp_method == CPUQuantMethod::INT8_W8A8) {
      uint8_t* __restrict__ A_tmp = (uint8_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * topk * N)));

      auto w1s = w1_scale.value();
      auto w2s = w2_scale.value();
      TORCH_CHECK(w1s.numel() == E * 2 * N);
      TORCH_CHECK(w2s.numel() == E * K);

      fused_experts_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          C_tmp,
          Aq_tmp,
          As_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<int8_t>(),
          packed_w2.data_ptr<int8_t>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    } else if (moe_comp_method == CPUQuantMethod::FP8_W8A16) {
                                                    
      scalar_t* __restrict__ A_tmp = (scalar_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      scalar_t* __restrict__ B_tmp = (scalar_t*)((void*)(intermediate_cache0 + M * topk * 2 * N));

      CHECK_MOE_SCALES_FP8(1, 2);
      fused_experts_fp8_kernel_impl(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<at::Float8_e4m3fn>(),
          packed_w2.data_ptr<at::Float8_e4m3fn>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          block_size_N,
          block_size_K,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    } else if (moe_comp_method == CPUQuantMethod::INT4_W4A8) {
      uint8_t* __restrict__ A_tmp = (uint8_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(intermediate_cache0 + M * topk * 2 * N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * topk * N)));
      int8_t* __restrict__ dqB_tmp = (int8_t*)((void*)(As_tmp + M * topk));

                                                                                                
                                                  
      int64_t num_groups = w1_scale.value().size(2);
      const int group_size = K / num_groups;
                                     
      fused_experts_int4_w4a8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          Aq_tmp,
          As_tmp,
          nullptr,
          C_tmp,
          dqB_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<uint8_t>(),
          packed_w2.data_ptr<uint8_t>(),
          w1_zero.value().data_ptr<int8_t>(),
          w2_zero.value().data_ptr<int8_t>(),
          w1_scale.value().data_ptr<float>(),
          w2_scale.value().data_ptr<float>(),
          group_size,
          topk_weights.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    } else {
      scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));

      fused_experts_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    }
  });
  return out_hidden_states;
}

                       
  
                        
              
             
                    
at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::shared_expert_cpu", std::vector<c10::IValue>({hidden_states, w1, w2}));

  auto packed_w1 = is_vnni ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_vnni ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(fused_experts_out);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w1);
  CHECK_DIM(2, w2);
  CHECK_EQ(hidden_states.sizes(), fused_experts_out.sizes());
  CHECK_EQ(hidden_states.scalar_type(), st);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(0) / 2;

                                              
  int64_t packed_K = get_row_size(K, use_int8_w8a8);
  int64_t packed_N = get_row_size(N, use_int8_w8a8);

                        
  CHECK_EQ(w2.size(0), K);
  CHECK_EQ(packed_w1.size(1), packed_K);
  CHECK_EQ(packed_w2.size(1), packed_N);

                 
  check_moe_scales(use_int8_w8a8, use_fp8_w8a16, w1_scale, w2_scale, block_size);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

                                                                                      
                                      
                                            
    
                   
                                   
                      
    
                   
                                       
                                                              
    
  int num_threads = at::get_num_threads();
  int64_t buffer_size_nbytes = M * N * 2 + num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);

  if (use_int8_w8a8) {
    buffer_size_nbytes += std::max(M * K, M * N) + M * sizeof(float);
  }
  if (use_fp8_w8a16) {
    buffer_size_nbytes += M * 2 * N * 2 + num_threads * MAX_CACHE_BLOCK_SIZE * BLOCK_M * std::max(K, N) * 2;
  }

  auto buffer = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "share_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer.data_ptr<int8_t>()));
    float* __restrict__ C_tmp = (float*)((void*)(intermediate_cache1 + M * N));

    if (use_int8_w8a8) {
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * N)));

      auto w1s = w1_scale.value();
      auto w2s = w2_scale.value();
      TORCH_CHECK(w1s.numel() == 2 * N);
      TORCH_CHECK(w2s.numel() == K);

      shared_expert_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          Aq_tmp,
          As_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<int8_t>(),
          packed_w2.data_ptr<int8_t>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          fused_experts_out.data_ptr<scalar_t>(),
          routed_scaling_factor,
          M,
          N,
          K);
    } else if (use_fp8_w8a16) {
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      scalar_t* __restrict__ B_tmp = (scalar_t*)((void*)(intermediate_cache0 + M * 2 * N));

      CHECK_MOE_SCALES_FP8(0, 1);
      shared_expert_fp8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<at::Float8_e4m3fn>(),
          packed_w2.data_ptr<at::Float8_e4m3fn>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          block_size_N,
          block_size_K,
          fused_experts_out.data_ptr<scalar_t>(),
          routed_scaling_factor,
          M,
          N,
          K);
    } else {
      shared_expert_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          fused_experts_out.data_ptr<scalar_t>(),
          routed_scaling_factor,
          M,
          N,
          K);
    }
  });
  return out_hidden_states;
}
