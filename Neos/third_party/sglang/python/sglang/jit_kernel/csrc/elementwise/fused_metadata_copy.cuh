  
                                                                
                                                     
  
            
                                                                                  
                                                                               
                                                                             
                                             
  
                        
                                                                     
                                                   
                                                            
                                                         
  
          
                                                                                      
                                                                                  
                                                                                          
                                                                                            
  
         
                                                                                 
                                                                                            
   

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>                 
#include <cuda_runtime.h>

                                                                                                  
enum ForwardModeEnum { DECODE = 0, TARGET_VERIFY = 1, DRAFT_EXTEND = 2 };

   
                                                
                                                                   
                                                                            
   
struct SourcePointers {
  const int32_t* __restrict__ cache_seqlens;                                         
  const int32_t* __restrict__ cu_seqlens_k;                                              
  const int32_t* __restrict__ page_indices;                              
  const int32_t* __restrict__ nsa_cache_seqlens;                                 
  const int32_t* __restrict__ seqlens_expanded;                                                                   
  const int32_t* __restrict__ nsa_cu_seqlens_k;                                       
  const int32_t* __restrict__ real_page_table;                                 
  const int32_t* __restrict__ flashmla_num_splits;                                   
  const int32_t* __restrict__ flashmla_metadata;                                 
};

   
                                                     
                                                                        
                                                 
   
struct DestinationPointers {
  int32_t* __restrict__ cache_seqlens;                                         
  int32_t* __restrict__ cu_seqlens_k;                                              
  int32_t* __restrict__ page_table_1;                                                         
  int32_t* __restrict__ nsa_cache_seqlens;                                 
  int32_t* __restrict__ seqlens_expanded;                                                                   
  int32_t* __restrict__ nsa_cu_seqlens_k;                                       
  int32_t* __restrict__ real_page_table;                                 
  int32_t* __restrict__ flashmla_num_splits;                                   
  int32_t* __restrict__ flashmla_metadata;                                 
};

   
                                                                     
                                                                     
   
struct FusedMetadataCopyParams {
  SourcePointers src;                                
  DestinationPointers dst;                                

                      
  int forward_mode;                                                            
  int bs;                                       
  int max_len;                                                  
  int max_seqlen_k;                                                                     
  int seqlens_expanded_size;                                           
  int page_indices_rows;                                            
  int page_table_1_stride;                                   
  int real_page_table_cols;                                     
  int real_page_table_dst_stride;                                           
  int flashmla_metadata_size;                                  
};

   
                                                                    
                                                                                   
                                                              
   
struct FusedMetadataCopyMultiParams {
  SourcePointers src;                                                       
  DestinationPointers dst0;                                   
  DestinationPointers dst1;                                   
  DestinationPointers dst2;                                   

                      
  int bs;                                       
  int max_len;                                                     
  int seqlens_expanded_size;                                           
  int page_table_1_stride;                                   
  int real_page_table_cols;                                     
  int real_page_table_dst_stride;                                           
  int flashmla_metadata_size;                                  
};

   
                                                                              
                                                                         
                                                  
  
          
                                                               
                                                                              
                                                                                          
  
                                                                    
  
                                                                            
                                                                       
   
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_kernel(const FusedMetadataCopyParams __grid_constant__ params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

                                      
  const auto& src = params.src;
  const auto& dst = params.dst;
  const int forward_mode = params.forward_mode;
  const int bs = params.bs;
  const int max_len = params.max_len;
  const int max_seqlen_k = params.max_seqlen_k;
  const int seqlens_expanded_size = params.seqlens_expanded_size;
  const int page_indices_rows = params.page_indices_rows;
  const int page_table_1_stride = params.page_table_1_stride;
  const int real_page_table_cols = params.real_page_table_cols;
  const int real_page_table_dst_stride = params.real_page_table_dst_stride;
  const int flashmla_metadata_size = params.flashmla_metadata_size;

                                                           
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    dst.cache_seqlens[i] = src.cache_seqlens[i];
  }

                                                                 
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    dst.cu_seqlens_k[i + 1] = src.cu_seqlens_k[i + 1];
  }

                                                              
  if (forward_mode == 0) {           
    int page_table_elements = bs * max_len;
#pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
      int row = i / max_len;
      int col = i % max_len;
      dst.page_table_1[row * page_table_1_stride + col] = src.page_indices[i];
    }
  } else {                                  
    int page_table_elements = page_indices_rows * max_seqlen_k;
#pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
      int row = i / max_seqlen_k;
      int col = i % max_seqlen_k;
      dst.page_table_1[row * page_table_1_stride + col] = src.page_indices[i];
    }
  }

                                                                          
  if (forward_mode != 0) {                                  
#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.seqlens_expanded[i] = src.seqlens_expanded[i];
    }
  }

                                                                
  if (forward_mode == 0) {           
#pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
      dst.nsa_cache_seqlens[i] = src.nsa_cache_seqlens[i];
    }

#pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
      dst.nsa_cu_seqlens_k[i + 1] = src.nsa_cu_seqlens_k[i + 1];
    }
  } else {                                  
#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.nsa_cache_seqlens[i] = src.nsa_cache_seqlens[i];
    }

#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.nsa_cu_seqlens_k[i + 1] = src.nsa_cu_seqlens_k[i + 1];
    }
  }

                                               
  if constexpr (HAS_REAL_PAGE_TABLE) {
    int real_table_elements = (forward_mode == 0 ? bs : page_indices_rows) * real_page_table_cols;
#pragma unroll 2
    for (int i = tid; i < real_table_elements; i += total_threads) {
      int row = i / real_page_table_cols;
      int col = i % real_page_table_cols;
      dst.real_page_table[row * real_page_table_dst_stride + col] =
          src.real_page_table[row * real_page_table_cols + col];
    }
  }

                                                                
  if constexpr (HAS_FLASHMLA) {
    int flashmla_size = (forward_mode == 0) ? (bs + 1) : (seqlens_expanded_size + 1);

    if (forward_mode == 0) {
#pragma unroll 8
      for (int i = tid; i < flashmla_size; i += total_threads) {
        dst.flashmla_num_splits[i] = src.flashmla_num_splits[i];
      }
    } else {
#pragma unroll 4
      for (int i = tid; i < flashmla_size; i += total_threads) {
        dst.flashmla_num_splits[i] = src.flashmla_num_splits[i];
      }
    }

#pragma unroll 2
    for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
      dst.flashmla_metadata[i] = src.flashmla_metadata[i];
    }
  }
}

   
                                        
                                                                          
  
                                                                     
                                                           
                                                                             
                                         
  
                                                                           
  
                                                                            
                                                                       
   
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_multi_kernel(const FusedMetadataCopyMultiParams __grid_constant__ params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

                                      
  const auto& src = params.src;
  const auto& dst0 = params.dst0;
  const auto& dst1 = params.dst1;
  const auto& dst2 = params.dst2;
  const int bs = params.bs;
  const int max_len = params.max_len;
  const int seqlens_expanded_size = params.seqlens_expanded_size;
  const int page_table_1_stride = params.page_table_1_stride;
  const int real_page_table_cols = params.real_page_table_cols;
  const int real_page_table_dst_stride = params.real_page_table_dst_stride;
  const int flashmla_metadata_size = params.flashmla_metadata_size;

                                         
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.cache_seqlens[i];
    dst0.cache_seqlens[i] = val;
    dst1.cache_seqlens[i] = val;
    dst2.cache_seqlens[i] = val;
  }

                                                             
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.cu_seqlens_k[i + 1];
    dst0.cu_seqlens_k[i + 1] = val;
    dst1.cu_seqlens_k[i + 1] = val;
    dst2.cu_seqlens_k[i + 1] = val;
  }

                                                     
  int page_table_elements = bs * max_len;
#pragma unroll 4
  for (int i = tid; i < page_table_elements; i += total_threads) {
    int row = i / max_len;
    int col = i % max_len;
    int32_t val = src.page_indices[i];
    dst0.page_table_1[row * page_table_1_stride + col] = val;
    dst1.page_table_1[row * page_table_1_stride + col] = val;
    dst2.page_table_1[row * page_table_1_stride + col] = val;
  }

                                             
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.nsa_cache_seqlens[i];
    dst0.nsa_cache_seqlens[i] = val;
    dst1.nsa_cache_seqlens[i] = val;
    dst2.nsa_cache_seqlens[i] = val;
  }

                                          
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.nsa_cu_seqlens_k[i + 1];
    dst0.nsa_cu_seqlens_k[i + 1] = val;
    dst1.nsa_cu_seqlens_k[i + 1] = val;
    dst2.nsa_cu_seqlens_k[i + 1] = val;
  }

                                           
  if (src.real_page_table != nullptr && dst0.real_page_table != nullptr) {
    int real_table_elements = bs * real_page_table_cols;
#pragma unroll 2
    for (int i = tid; i < real_table_elements; i += total_threads) {
      int row = i / real_page_table_cols;
      int col = i % real_page_table_cols;
      int src_idx = row * real_page_table_cols + col;
      int dst_idx = row * real_page_table_dst_stride + col;
      int32_t val = src.real_page_table[src_idx];
      dst0.real_page_table[dst_idx] = val;
      dst1.real_page_table[dst_idx] = val;
      dst2.real_page_table[dst_idx] = val;
    }
  }

                                             
  if constexpr (HAS_FLASHMLA) {
    int flashmla_size = bs + 1;
#pragma unroll 8
    for (int i = tid; i < flashmla_size; i += total_threads) {
      int32_t val = src.flashmla_num_splits[i];
      dst0.flashmla_num_splits[i] = val;
      dst1.flashmla_num_splits[i] = val;
      dst2.flashmla_num_splits[i] = val;
    }

#pragma unroll 2
    for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
      int32_t val = src.flashmla_metadata[i];
      dst0.flashmla_metadata[i] = val;
      dst1.flashmla_metadata[i] = val;
      dst2.flashmla_metadata[i] = val;
    }
  }
}

                                                                               
                                                  
                                                                               

namespace {

                                 
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_GRID_SIZE = 1024;                                              

   
                                                                     
                                                                        
  
                                                      
                                                           
                                                           
                                           
   
template <typename T>
inline const T* unwrap_data_ptr(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  }
  return static_cast<const T*>(tensor.data_ptr());
}

   
                                                                             
                                                                        
  
                                                      
                                                           
                                                           
                                                   
   
template <typename T>
inline T* unwrap_data_ptr_mut(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  }
  return static_cast<T*>(tensor.data_ptr());
}

   
                                                                               
                                                                                  
  
                                                      
                                                                             
                                                           
                                                                                
   
template <typename T>
inline const T*
unwrap_optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& optional_tensor, const char* name) {
  using namespace host;
  if (!optional_tensor.has_value()) {
    return nullptr;
  }
  const auto& tensor = optional_tensor.value();
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  return static_cast<const T*>(tensor.data_ptr());
}

   
                                                                                       
                                                                                  
  
                                                      
                                                                             
                                                           
                                                                                        
   
template <typename T>
inline T*
unwrap_optional_data_ptr_mut(const tvm::ffi::Optional<tvm::ffi::TensorView>& optional_tensor, const char* name) {
  using namespace host;
  if (!optional_tensor.has_value()) {
    return nullptr;
  }
  const auto& tensor = optional_tensor.value();
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  return static_cast<T*>(tensor.data_ptr());
}

   
                                         
  
                                               
                                                                          
                                           
   
inline dim3 get_launch_config(int total_work, int threads_per_block = THREADS_PER_BLOCK) {
  int num_blocks = (total_work + threads_per_block - 1) / threads_per_block;
                                                                                
  num_blocks = std::min(num_blocks, MAX_GRID_SIZE);
  return dim3(num_blocks);
}

   
                                                             
  
                                                                                 
                                                                              
                               
  
                  
                                                  
                                                                                      
                                                             
                                                                          
  
                                                                               
                                                                          
                                                                     
   
template <int FORWARD_MODE, bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
struct FusedMetadataCopyKernel {
  static_assert(
      FORWARD_MODE >= 0 && FORWARD_MODE <= 2,
      "FORWARD_MODE must be 0 (DECODE), 1 (TARGET_VERIFY), or 2 (DRAFT_EXTEND)");

  static void
  run(const tvm::ffi::TensorView cache_seqlens_src,
      const tvm::ffi::TensorView cu_seqlens_k_src,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView nsa_cache_seqlens_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seqlens_expanded_src,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_src,
      const tvm::ffi::TensorView cache_seqlens_dst,
      const tvm::ffi::TensorView cu_seqlens_k_dst,
      const tvm::ffi::TensorView page_table_1_dst,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seqlens_expanded_dst,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst,
      int bs,
      int max_len,
      int max_seqlen_k,
      int seqlens_expanded_size) {
    using namespace host;

                                                                     
                                                                            
    const auto params = FusedMetadataCopyParams{
        .src =
            {
                .cache_seqlens = unwrap_data_ptr<int32_t>(cache_seqlens_src, "cache_seqlens_src"),
                .cu_seqlens_k = unwrap_data_ptr<int32_t>(cu_seqlens_k_src, "cu_seqlens_k_src"),
                .page_indices = unwrap_data_ptr<int32_t>(page_indices_src, "page_indices_src"),
                .nsa_cache_seqlens = unwrap_data_ptr<int32_t>(nsa_cache_seqlens_src, "nsa_cache_seqlens_src"),
                .seqlens_expanded = unwrap_optional_data_ptr<int32_t>(seqlens_expanded_src, "seqlens_expanded_src"),
                .nsa_cu_seqlens_k = unwrap_data_ptr<int32_t>(nsa_cu_seqlens_k_src, "nsa_cu_seqlens_k_src"),
                .real_page_table = unwrap_optional_data_ptr<int32_t>(real_page_table_src, "real_page_table_src"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr<int32_t>(flashmla_num_splits_src, "flashmla_num_splits_src"),
                .flashmla_metadata = unwrap_optional_data_ptr<int32_t>(flashmla_metadata_src, "flashmla_metadata_src"),
            },
        .dst =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst, "cache_seqlens_dst"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst, "cu_seqlens_k_dst"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst, "page_table_1_dst"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst, "nsa_cache_seqlens_dst"),
                .seqlens_expanded = unwrap_optional_data_ptr_mut<int32_t>(seqlens_expanded_dst, "seqlens_expanded_dst"),
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst, "nsa_cu_seqlens_k_dst"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst, "real_page_table_dst"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst, "flashmla_num_splits_dst"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst, "flashmla_metadata_dst"),
            },
        .forward_mode = FORWARD_MODE,
        .bs = bs,
        .max_len = max_len,
        .max_seqlen_k = max_seqlen_k,
        .seqlens_expanded_size = seqlens_expanded_size,
        .page_indices_rows = static_cast<int>(page_indices_src.shape()[0]),
        .page_table_1_stride = static_cast<int>(page_table_1_dst.shape()[1]),
        .real_page_table_cols =
            real_page_table_src.has_value() ? static_cast<int>(real_page_table_src.value().shape()[1]) : 0,
        .real_page_table_dst_stride =
            real_page_table_dst.has_value() ? static_cast<int>(real_page_table_dst.value().stride(0)) : 0,
        .flashmla_metadata_size =
            flashmla_metadata_src.has_value() ? static_cast<int>(flashmla_metadata_src.value().numel()) : 0,
    };

                                   
    int max_elements = std::max(
        {bs,
         params.page_indices_rows * max_seqlen_k,
         seqlens_expanded_size,
         HAS_FLASHMLA ? (seqlens_expanded_size + 1) : 0,
         HAS_FLASHMLA ? params.flashmla_metadata_size : 0});

    dim3 grid = get_launch_config(max_elements);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = cache_seqlens_src.device();

                                               
    host::LaunchKernel(grid, block, device)(fused_metadata_copy_kernel<HAS_REAL_PAGE_TABLE, HAS_FLASHMLA>, params);
  }
};

   
                                                            
  
                                                                               
                                                                              
                                                          
                                                           
                                                                             
                                                           
  
                                                                              
                                                                                
  
                                                                                
                                               
  
                  
                                                                                          
                                                                                
  
                                                                          
                                                                     
   
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
struct FusedMetadataCopyMultiKernel {
  static void
  run(const tvm::ffi::TensorView cache_seqlens_src,
      const tvm::ffi::TensorView cu_seqlens_k_src,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView nsa_cache_seqlens_src,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_src,
      const tvm::ffi::TensorView cache_seqlens_dst0,
      const tvm::ffi::TensorView cu_seqlens_k_dst0,
      const tvm::ffi::TensorView page_table_1_dst0,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst0,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst0,
      const tvm::ffi::TensorView cache_seqlens_dst1,
      const tvm::ffi::TensorView cu_seqlens_k_dst1,
      const tvm::ffi::TensorView page_table_1_dst1,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst1,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst1,
      const tvm::ffi::TensorView cache_seqlens_dst2,
      const tvm::ffi::TensorView cu_seqlens_k_dst2,
      const tvm::ffi::TensorView page_table_1_dst2,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst2,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst2,
      int bs,
      int max_len,
      int seqlens_expanded_size) {
    using namespace host;

                                                                     
                                                                            
    const auto params = FusedMetadataCopyMultiParams{
        .src =
            {
                .cache_seqlens = unwrap_data_ptr<int32_t>(cache_seqlens_src, "cache_seqlens_src"),
                .cu_seqlens_k = unwrap_data_ptr<int32_t>(cu_seqlens_k_src, "cu_seqlens_k_src"),
                .page_indices = unwrap_data_ptr<int32_t>(page_indices_src, "page_indices_src"),
                .nsa_cache_seqlens = unwrap_data_ptr<int32_t>(nsa_cache_seqlens_src, "nsa_cache_seqlens_src"),
                .seqlens_expanded = nullptr,                                          
                .nsa_cu_seqlens_k = unwrap_data_ptr<int32_t>(nsa_cu_seqlens_k_src, "nsa_cu_seqlens_k_src"),
                .real_page_table = unwrap_optional_data_ptr<int32_t>(real_page_table_src, "real_page_table_src"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr<int32_t>(flashmla_num_splits_src, "flashmla_num_splits_src"),
                .flashmla_metadata = unwrap_optional_data_ptr<int32_t>(flashmla_metadata_src, "flashmla_metadata_src"),
            },
        .dst0 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst0, "cache_seqlens_dst0"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst0, "cu_seqlens_k_dst0"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst0, "page_table_1_dst0"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst0, "nsa_cache_seqlens_dst0"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst0, "nsa_cu_seqlens_k_dst0"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst0, "real_page_table_dst0"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst0, "flashmla_num_splits_dst0"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst0, "flashmla_metadata_dst0"),
            },
        .dst1 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst1, "cache_seqlens_dst1"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst1, "cu_seqlens_k_dst1"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst1, "page_table_1_dst1"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst1, "nsa_cache_seqlens_dst1"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst1, "nsa_cu_seqlens_k_dst1"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst1, "real_page_table_dst1"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst1, "flashmla_num_splits_dst1"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst1, "flashmla_metadata_dst1"),
            },
        .dst2 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst2, "cache_seqlens_dst2"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst2, "cu_seqlens_k_dst2"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst2, "page_table_1_dst2"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst2, "nsa_cache_seqlens_dst2"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst2, "nsa_cu_seqlens_k_dst2"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst2, "real_page_table_dst2"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst2, "flashmla_num_splits_dst2"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst2, "flashmla_metadata_dst2"),
            },
        .bs = bs,
        .max_len = max_len,
        .seqlens_expanded_size = seqlens_expanded_size,
        .page_table_1_stride = static_cast<int>(page_table_1_dst0.shape()[1]),
        .real_page_table_cols =
            real_page_table_src.has_value() ? static_cast<int>(real_page_table_src.value().shape()[1]) : 0,
        .real_page_table_dst_stride =
            real_page_table_dst0.has_value() ? static_cast<int>(real_page_table_dst0.value().stride(0)) : 0,
        .flashmla_metadata_size =
            flashmla_metadata_src.has_value() ? static_cast<int>(flashmla_metadata_src.value().numel()) : 0,
    };

    dim3 grid = get_launch_config(bs * max_len);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = cache_seqlens_src.device();

                                                     
    host::LaunchKernel(grid, block, device)(
        fused_metadata_copy_multi_kernel<HAS_REAL_PAGE_TABLE, HAS_FLASHMLA>, params);
  }
};

}              
