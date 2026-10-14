#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

                
#define AT_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...) \
  [&] {                                          \
    if (BOOL_V) {                                \
      constexpr bool BOOL_NAME = true;           \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr bool BOOL_NAME = false;          \
      return __VA_ARGS__();                      \
    }                                            \
  }()

#define AT_DISPATCH_BOOL2(BOOL_V1, BOOL_NAME1, BOOL_V2, BOOL_NAME2, ...) \
  [&] {                                                                  \
    if (BOOL_V1) {                                                       \
      constexpr bool BOOL_NAME1 = true;                                  \
      if (BOOL_V2) {                                                     \
        constexpr bool BOOL_NAME2 = true;                                \
        return __VA_ARGS__();                                            \
      } else {                                                           \
        constexpr bool BOOL_NAME2 = false;                               \
        return __VA_ARGS__();                                            \
      }                                                                  \
    } else {                                                             \
      constexpr bool BOOL_NAME1 = false;                                 \
      if (BOOL_V2) {                                                     \
        constexpr bool BOOL_NAME2 = true;                                \
        return __VA_ARGS__();                                            \
      } else {                                                           \
        constexpr bool BOOL_NAME2 = false;                               \
        return __VA_ARGS__();                                            \
      }                                                                  \
    }                                                                    \
  }()

                                                
#define CPU_DISPATCH_PACKED_TYPES(TYPE, ...)                     \
  [&] {                                                          \
    switch (TYPE) {                                              \
      case at::ScalarType::BFloat16: {                           \
        using packed_t = at::BFloat16;                           \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Half: {                               \
        using packed_t = at::Half;                               \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Char: {                               \
        using packed_t = int8_t;                                 \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Float8_e4m3fn: {                      \
        using packed_t = at::Float8_e4m3fn;                      \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false, "Unsupported floating data type.\n"); \
    }                                                            \
  }()

                                             
                                                      
                                             
#define CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(TYPE1, TYPE2, ...) \
  [&] {                                                            \
    if (TYPE2 == at::kFloat) {                                     \
      switch (TYPE1) {                                             \
        case at::ScalarType::BFloat16: {                           \
          using scalar_t = at::BFloat16;                           \
          using param_t = float;                                   \
          return __VA_ARGS__();                                    \
        }                                                          \
        case at::ScalarType::Half: {                               \
          using scalar_t = at::Half;                               \
          using param_t = float;                                   \
          return __VA_ARGS__();                                    \
        }                                                          \
        default:                                                   \
          TORCH_CHECK(false, "Unsupported floating data type.\n"); \
      }                                                            \
    } else {                                                       \
      TORCH_CHECK(TYPE1 == TYPE2);                                 \
      switch (TYPE1) {                                             \
        case at::ScalarType::BFloat16: {                           \
          using scalar_t = at::BFloat16;                           \
          using param_t = at::BFloat16;                            \
          return __VA_ARGS__();                                    \
        }                                                          \
        case at::ScalarType::Half: {                               \
          using scalar_t = at::Half;                               \
          using param_t = at::Half;                                \
          return __VA_ARGS__();                                    \
        }                                                          \
        default:                                                   \
          TORCH_CHECK(false, "Unsupported floating data type.\n"); \
      }                                                            \
    }                                                              \
  }()

#define UNUSED(x) (void)(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

template <bool is_only_lastdim_contiguous>
static inline void CHECK_INPUT_SHAPE_DTYPE(const at::Tensor& tensor, const at::IntArrayRef sizes, at::ScalarType st) {
  TORCH_CHECK(tensor.sizes() == sizes, "Input tensor shape mismatch: expected ", sizes, ", got ", tensor.sizes());
  TORCH_CHECK(tensor.scalar_type() == st, "Input tensor dtype mismatch");
  if constexpr (is_only_lastdim_contiguous) {
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor);
  } else {
    CHECK_INPUT(tensor);
  }
}
#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

                         
  
                                                                                     
                                                                 
  
                                                                                       
                                    
  
                                                                      
                                                                              
  

                             
constexpr int GRAIN_SIZE = 1024;

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

                                                                
                                                                
inline int get_thread_num() {
#if defined(_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

                                     
template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
                               
                    
                             
                    
                 
            
                              
                      
                            
                                  
                                                                 
     
                     
#else
                                   
  T n_my = div_up(n, nth);
  n_start = ith * n_my;
  n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
  {
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
  }
#else
  f(0, n);
#endif
}

                                    
                                              
int inline adjust_num_threads(int m) {
  int actual_nth = at::get_num_threads();
  if (m == 1) {
    return actual_nth;
  }
  return std::max(1, (actual_nth >> 1) * 2);
}

template <typename func_t>
inline void parallel_2d(int m, int n, const func_t& f) {
                                       
  int nth = adjust_num_threads(m);

                            
    
                                        
                                      
                                     
    
               
                    
                          
            
                                   
    
  float r = float(m) / n;
  int nth_m = std::ceil(std::sqrt(r * nth));
  int nth_n = 1;
  for (; nth_m > 0; --nth_m) {
    nth_n = nth / nth_m;
    if (nth_m * nth_n == nth) {
      break;
    }
  }

#if defined(_OPENMP)
#pragma omp parallel num_threads(nth)
  {
    int ith = omp_get_thread_num();
    int ith_m = ith / nth_n;
    int ith_n = ith % nth_n;

    int thread_block_m = div_up(m, nth_m);
    int thread_block_n = div_up(n, nth_n);

    int begin_m = ith_m * thread_block_m;
    int end_m = std::min(m, begin_m + thread_block_m);
    int begin_n = ith_n * thread_block_n;
    int end_n = std::min(n, begin_n + thread_block_n);

    f(begin_m, end_m, begin_n, end_n);
  }
#else
  f(0, m, 0, n);
#endif
}

                         
                                                      
#define MAX_CACHE_BLOCK_SIZE 4

template <typename T>
inline int get_cache_blocks(int chunk_size) {
                            
  const int L2_size = 2048 * 1024 >> 1;
  return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
}

template <>
inline int get_cache_blocks<at::Float8_e4m3fn>(int chunk_size) {
                                     
  int cache_block_size = get_cache_blocks<at::BFloat16>(chunk_size);
  return std::min(MAX_CACHE_BLOCK_SIZE, cache_block_size);
}

                                                       
template <typename T, typename func_t>
inline void loop_2d(int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1, int64_t chunk_size, const func_t& f) {
                                                   
  int64_t cache_blocks_nb = get_cache_blocks<T>(chunk_size);

                                                            
                                                                                 
  for (int64_t nbb = nb0; nbb < nb1; nbb += cache_blocks_nb) {
    for (int64_t mb = mb0; mb < mb1; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, nb1); ++nb) {
        f(mb, nb, nb - nbb);
      }
    }
  }
}

                                       
template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

                                       

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

                                           
template <typename T>
inline T* conditional_data_ptr(const std::optional<at::Tensor>& opt) {
  return opt.has_value() ? opt.value().data_ptr<T>() : nullptr;
}

}                        
