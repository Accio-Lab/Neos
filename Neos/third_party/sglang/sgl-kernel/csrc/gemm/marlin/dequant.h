  
                                                               

                                                                     
                                                      

                                        
                                                 
                  

                                                                              
                                                                              
                                                  

                                                                            
                                      

                                    

                                           

                   

                                           
                                                     
                               

                                         

                         

                              
                                   
                                 

                                                                  
                                                                 

                       

                               
                                    
                                       
                                                  

                                                                        
                                        


               

                                                                 
                                       


                       

                               
                                          
                                                  

                                                                    

  

#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
                                                                    
                                                                                
             
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

                                                                           
        
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(res) : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, sglang::ScalarTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

  
                                                                          
                                                                                
                           
          
                                                                                                                                                                                   
          
                                                                                                                                                                                   
  
template <>
__device__ inline void dequant<half2, sglang::kU4B8.id(), true>(int q, half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
                                                           
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, sglang::kU4B8.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
                                                           
                     
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
                    
                                                                             
                                   
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(
      *reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, sglang::kU4.id(), true>(int q, half2* frag_b) {
  dequant<half2, sglang::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, sglang::kU4.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
                                                           
                     
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
                    
                                                                             
                                   
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(
      *reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4B8.id(), true>(int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

                                                           
                     
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
                    

  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4B8.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4B8.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43084308;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4.id(), true>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43004300;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

  
                                                                                
                  
          
                                                                                                                                                                                 
          
                                                                                                                                                                                   
  
template <>
__device__ inline void dequant<half2, sglang::kU8B128.id(), true>(int q, half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, sglang::kU8B128.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8B128.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<half2, sglang::kU8.id(), true>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8B128.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, sglang::kU8.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU8B128.id(), false>(int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU8.id(), false>(int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<half2, sglang::kFE4M3fn.id(), true>(int q, half2* frag_b) {
                                              
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

                                                
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, sglang::kFE4M3fn.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kFE4M3fn.id(), true>(q, frag_b);

                                              
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;

                                      
  constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

                                    
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE4M3fn.id(), true>(int q, nv_bfloat162* frag_b) {
                                              
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

  constexpr int MASK = 0x7F007F00;

                                                
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE4M3fn.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kFE4M3fn.id(), true>(q, frag_b);

                                              
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;

                                      
  constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
                                                                             
             
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

                                        
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<half2, sglang::kFE2M1f.id(), true>(int q, half2* frag_b) {
                                              
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

                                                
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, sglang::kFE2M1f.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kFE2M1f.id(), true>(q, frag_b);

                                              
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;

                                      
  constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

                                    
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE2M1f.id(), true>(int q, nv_bfloat162* frag_b) {
                                              
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

                                                
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE2M1f.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kFE2M1f.id(), true>(q, frag_b);

                                              
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;

                                      
  constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
                                                                             
             
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

                                    
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <typename scalar_t2>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

                                                                       
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

                                                
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
};

                                                               
template <typename scalar_t2, sglang::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, sglang::kFE4M3fn.id()>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

                                                                       
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, sglang::kFE4M3fn.id()>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

                                                
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

                                                                       
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, sglang::kFE8M0fnu.id()>(int q, nv_bfloat162* frag_b) {
                                                                     
                                                                            
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

                                                                       
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

#endif

}                                    
