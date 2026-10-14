#include "common.h"
#include "flash_attn.h"
#include "gemm.h"

namespace {

                                   
                                                         
                                                       
                                                           
                                                           
  

template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int buffer_size_per_thread,
    bool is_prefix_skipped) {
            
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

                                                
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

                             
  int MB = div_up(max_len_extend, BLOCK_M);

                                         
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
                                          
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    scalar_t* __restrict__ s_delta = reinterpret_cast<scalar_t*>(s_i);

                                      
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

                                                   
    scalar_t* __restrict__ Btmp = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

                                                         
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
                                  
      int head_kv_id = head_id / num_groups;
      int seq_len = seq_lens[bs];
      int seq_len_extend = extend_seq_lens[bs];
      int seq_len_prefix = seq_len - seq_len_extend;
      int seq_extend_start_loc = extend_start_loc[bs];

      int req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      if (is_prefix_skipped) {
        TORCH_CHECK(seq_len_prefix == 0, "extend attention: expect seq_len_prefix to be 0, got ", seq_len_prefix);
      }

                              
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

                  
      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

                           
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

                                            
      for (int n = 0; n < seq_len_prefix; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, seq_len_prefix - n);

                                                    
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

                           
        pack_vnni<scalar_t, index_t>(
                         Btmp,
                         k_buffer + head_kv_id * k_strideH,
                         req_to_token + req_pool_id * max_context_len + n,
                         n_size,
                         head_size,
                         k_strideN,
                         BLOCK_N);

                                 
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        head_size,
                        q_strideM,
                        BLOCK_N,
                        BLOCK_N,
                        false,
                        q_ptr,
                        Btmp,
                        s_i);

        flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
            s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale);

                             
        pack_vnni2<scalar_t, index_t>(
                         Btmp,
                         v_buffer + head_kv_id * v_strideH,
                         req_to_token + req_pool_id * max_context_len + n,
                         n_size,
                         head_size_v,
                         v_strideN,
                         head_size_v);

                                           
        at::native::cpublas::brgemm(
                        m_size,
                        head_size_v,
                        padded_n_size,           
                        BLOCK_N,
                        head_size_v,
                        head_size_v,
                        true,
                        s_delta,
                        Btmp,
                        v_prime);
      }                             

                                           
      int num_keys = std::min(seq_len_extend, m + BLOCK_M);
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

                                                    
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

                           
        pack_vnni<scalar_t>(
                         Btmp,
                         k_extend + (seq_extend_start_loc + n) * ke_strideN + head_kv_id * ke_strideH,
                         n_size,
                         head_size,
                         ke_strideN,
                         BLOCK_N);

                                 
        at::native::cpublas::brgemm(
                        m_size,
                        n_size,
                        head_size,
                        q_strideM,
                        BLOCK_N,
                        BLOCK_N,
                        false,
                        q_ptr,
                        Btmp,
                        s_i);

                            
        if (num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            int last_col = m + row - n;
                                                  
            float* row_ptr = s_i + row * BLOCK_N;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
          }
        }

        flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
            s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale);

                             
        pack_vnni2<scalar_t>(
                         Btmp,
                         v_extend + (seq_extend_start_loc + n) * ve_strideN + head_kv_id * ve_strideH,
                         n_size,
                         head_size_v,
                         ve_strideN,
                         head_size_v);

                                           
        at::native::cpublas::brgemm(
                        m_size,
                        head_size_v,
                        padded_n_size,           
                        BLOCK_N,
                        head_size_v,
                        head_size_v,
                        true,
                        s_delta,
                        Btmp,
                        v_prime);
      }                             

      scalar_t* __restrict__ out_ptr = o_extend + (seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

                               
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}                        

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  static_assert(BLOCK_M <= BLOCK_N, "Make sure BLOCK_M <= BLOCK_N to prevent buffer overflows during causal masking");
  const int size_per_thread =
                    BLOCK_M * BLOCK_N * sizeof(float) +
                    BLOCK_M * head_size_v * sizeof(float) +
                    BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

#define LAUNCH_EXTEND_ATTENTION_KERNEL(BLOCK_M, BLOCK_N)                                   \
  do {                                                                                     \
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v); \
                                                                                           \
    extend_attention_kernel_impl<scalar_t, index_t, BLOCK_M, BLOCK_N>(                     \
        o_extend.data_ptr<scalar_t>(),                                                     \
        q_extend.data_ptr<scalar_t>(),                                                     \
        k_extend.data_ptr<scalar_t>(),                                                     \
        v_extend.data_ptr<scalar_t>(),                                                     \
        k_buffer.data_ptr<scalar_t>(),                                                     \
        v_buffer.data_ptr<scalar_t>(),                                                     \
        req_to_token.data_ptr<index_t>(),                                                  \
        req_pool_indices.data_ptr<int64_t>(),                                              \
        seq_lens.data_ptr<int64_t>(),                                                      \
        extend_seq_lens.data_ptr<index_t>(),                                               \
        extend_start_loc.data_ptr<index_t>(),                                              \
        buffer.data_ptr(),                                                                 \
        num_seqs,                                                                          \
        num_heads,                                                                         \
        num_heads_kv,                                                                      \
        head_size,                                                                         \
        head_size_v,                                                                       \
        q_strideM,                                                                         \
        q_strideH,                                                                         \
        ke_strideN,                                                                        \
        ke_strideH,                                                                        \
        ve_strideN,                                                                        \
        ve_strideH,                                                                        \
        k_strideN,                                                                         \
        k_strideH,                                                                         \
        v_strideN,                                                                         \
        v_strideH,                                                                         \
        sm_scale,                                                                          \
        max_num_reqs,                                                                      \
        max_context_len,                                                                   \
        max_total_num_tokens,                                                              \
        max_len_extend,                                                                    \
        sz,                                                                                \
        is_prefix_skipped);                                                                \
  } while (0)

                                                             
                                                               
  
                                               
                                                      
                                                      
                                               
                                                         
                                                         
                                                               
                                     
                             
                              
                               
  
void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::extend_attention_cpu",
      std::vector<c10::IValue>(
          {q_extend,
           k_extend,
           v_extend,
           o_extend,
           k_buffer,
           v_buffer,
           req_to_token,
           req_pool_indices,
           seq_lens,
           extend_seq_lens,
           extend_start_loc,
           max_len_extend}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

                                                
  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

                                      
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

                
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

                              
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

                           
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

                                                       
  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);

  int num_threads = at::get_num_threads();
  auto buffer = at::empty({}, q_extend.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_indices", [&] {
      if (max_len_extend <= 256) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(32, 64);
      } else if (max_len_extend <= 1024) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(128, 256);
      } else if (max_len_extend <= 4096) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(256, 768);
      } else {                          
        LAUNCH_EXTEND_ATTENTION_KERNEL(512, 768);
      }
    });
  });
}
