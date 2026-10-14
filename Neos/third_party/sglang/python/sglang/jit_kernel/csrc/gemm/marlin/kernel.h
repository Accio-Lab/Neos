
#include <sgl_kernel/scalar_type.hpp>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"

#define MARLIN_KERNEL_PARAMS                                                                                         \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
      const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, \
      const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks,        \
      bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace device::marlin {
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
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}                             
