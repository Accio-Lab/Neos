#pragma once

#include <hip/hip_runtime.h>

#include "quick_all_reduce_base.h"

namespace quickreduce {

struct CodecBase {
  const int thread;
  const int rank;
  const int group_leader;
  __quickreduce_device_inline__ CodecBase(int thread, int rank)
      : thread(thread), rank(rank), group_leader((threadIdx.x / kThreadGroupSize) * kThreadGroupSize) {
    set_fp16_ovfl(true);
  }
};

                                
template <typename T, int world_size>
struct CodecFP : public CodecBase {
  static constexpr int kWorldSize = world_size;
  static constexpr int kRankAtoms = kAtoms / kWorldSize;

                                               
                                                  
  static constexpr int kRankTransmittedTileSize = kBlockSize * kRankAtoms * sizeof(int32x4_t);
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTransmittedTileSize must be 16B aligned.");

                                                      
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

  __quickreduce_device_inline__ CodecFP(int thread, int rank) : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer, const int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      __builtin_nontemporal_store(data[i], send_buffer + thread);
      send_buffer += kAtomStride;
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer, int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      data[i] = __builtin_nontemporal_load(*recv_buffer + thread);
      *recv_buffer += kAtomStride;
    }
  }
};

                                     
                                                                  
                    
template <typename T, int world_size>
struct CodecQ4 : public CodecBase {
  static constexpr int kWorldSize = world_size;

                                               
                                                         
                                                                  
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1152;
  static constexpr int kRankTileScaleOffset = 1024;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride = kRankTileStride / sizeof(int32x4_t);

                                                      
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

                            

                                
  static constexpr int kScaleFactor = std::is_same<T, half>::value ? 0xB000B000 : 0xBE00BE00;

                          
  static constexpr int kScaleEpsilon = std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

                      
  static constexpr int kRangeMin = std::is_same<T, half>::value ? 0xC800C800 : 0xC100C100;

                      
  static constexpr int kRangeMax = std::is_same<T, half>::value ? 0x47004700 : 0x40E040E0;

                        
  static constexpr int kRangeBias = 0x00080008;

  __quickreduce_device_inline__ CodecQ4(int thread, int rank) : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer, const int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

                                                                     
                                                                 
      int wblockmax = group_abs_max<T>(atom);

                      
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

                                             
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

                                           
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

                                 
      int qw = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12);

                                            
                                                     
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer, int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
                                                      
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      int32_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

                               
      int32x4_t w;
      {
        static constexpr uint kMask000F = 0x000F000F;
        static constexpr uint kHalf2_1024 = 0x64006400;                               
        static uint constexpr kHalf2_1032 = 0xE408E408;                                 

        for (int i = 0; i < 4; i++) {
          if constexpr (std::is_same<T, half>::value) {
            int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
            w[i] = packed_add<half>(q4, kHalf2_1032);
          } else {
            int32_t int16_2 = (qw >> (i * 4)) & kMask000F;
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);
            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

                              
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

                                     
                                                                  
                    
template <typename T, int world_size>
struct CodecQ6 : public CodecBase {
  static constexpr int kWorldSize = world_size;

                                               
                                                         
                                                                       
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1664;
  static constexpr int kRankTileQ2Offset = 1024;
  static constexpr int kRankTileScaleOffset = 1536;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride = kRankTileStride / sizeof(int32x4_t);

                                                      
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

                            

                                   
  static constexpr int kScaleFactor = std::is_same<T, half>::value ? 0xA800A800 : 0xBD00BD00;

                           
  static constexpr int kScaleEpsilon = std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

                         
  static constexpr int kRangeMin = std::is_same<T, half>::value ? 0xD000D000 : 0xC200C200;

                         
  static constexpr int kRangeMax = std::is_same<T, half>::value ? 0x4FC04FC0 : 0x41F841F8;

                          
  static constexpr int kRangeBias = 0x00200020;

  __quickreduce_device_inline__ CodecQ6(int thread, int rank) : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer, const int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

                                                                     
                                                                 
      int wblockmax = group_abs_max<T>(atom);

                      
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

                                             
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

                                           
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

                                           
      uint32_t q4w;
      uint16_t q2w = 0;
      q4w = (q[0] & 0x000F000F) | ((q[1] & 0x000F000F) << 4) | ((q[2] & 0x000F000F) << 8) | ((q[3] & 0x000F000F) << 12);
      {
        int16_t* tw = reinterpret_cast<int16_t*>(&q);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          q2w |= (tw[i] >> 4) << (i * 2);
        }
      }
                                            
                                                     
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr = reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      __builtin_nontemporal_store(q4w, q4w_ptr);
      __builtin_nontemporal_store(q2w, q2w_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer, int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
                                                      
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr = reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      uint32_t q4w = __builtin_nontemporal_load(q4w_ptr);
      uint16_t q2w = __builtin_nontemporal_load(q2w_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

                                
      int32x4_t w;
      {
        static uint constexpr kMask000F = 0x000F000F;
        static uint constexpr kHalf2_1024 = 0x64006400;                               
        static uint constexpr kHalf2_1056 = 0xE420E420;                                 

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int32_t q4 = q4w & kMask000F;
          int32_t q2 = (q2w & 0x3) | ((q2w & 0xC) << 14);
          q4w >>= 4;
          q2w >>= 4;
          if constexpr (std::is_same<T, half>::value) {
            int32_t q6 = q4 | (q2 << 4) | kHalf2_1024;
            asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(w[i]) : "v"(q6), "v"(kHalf2_1056));
          } else {
            int32_t int16_2 = q4 | (q2 << 4);
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);

            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

                              
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

                                 
      data[k] = w;
    }
  }
};

                                     
                                                                  
                    
template <typename T, int world_size>
struct CodecQ8 : public CodecBase {
  static constexpr int kWorldSize = world_size;

                                               
                                                        
                                                                 
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 2176;
  static constexpr int kRankTileScaleOffset = 2048;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0, "kRankTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride = kRankTileStride / sizeof(int32x4_t);

                                                      
  static constexpr int kTransmittedTileSize = kRankTransmittedTileSize * kWorldSize;

                            

                                    
  static constexpr int kScaleFactor = std::is_same<T, half>::value ? 0xA000A000 : 0xBC00BC00;

                          
  static constexpr int kScaleEpsilon = std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

                          
  static constexpr int kRangeMin = std::is_same<T, half>::value ? 0xD800D800 : 0xC300C300;
                          
  static constexpr int kRangeMax = std::is_same<T, half>::value ? 0x57F057F0 : 0x42FE42FE;

                            
  static constexpr int kRangeBias = 0x00800080;

  __quickreduce_device_inline__ CodecQ8(int thread, int rank) : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer, int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];
                                                                     
                                                                 
      int wblockmax = group_abs_max<T>(atom);

                      
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

                                             
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

                                           
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

                                   
      int32x2_t qw;
      qw[0] = q[0] | (q[1] << 8);
      qw[1] = q[2] | (q[3] << 8);

                                            
                                                     
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer, int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
                                                      
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) + (thread / 8);

      int32x2_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

                                
      int32x4_t w;
      {
        static uint constexpr kMask00FF = 0x00FF00FF;

                                     
        static uint constexpr kHalf2_1024 = 0x64006400;

                                       
        static uint constexpr kHalf2_1152 = 0xE480E480;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          if constexpr (std::is_same<T, half>::value) {
            int32_t q8 = ((qw[i / 2] >> ((i % 2) * 8)) & kMask00FF) | kHalf2_1024;
            w[i] = packed_add<half>(q8, kHalf2_1152);
          } else {
            int32_t int16_2 = (qw[i / 2] >> ((i % 2) * 8)) & kMask00FF;
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);
            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

                              
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

                     
template <typename T, class Codec, bool cast_bf2half>
struct AllReduceTwoshot {
  static_assert(sizeof(T) == 2);

  static constexpr int kWorldSize = Codec::kWorldSize;

  __device__ static void
  run(T const* __restrict__ input,
      T* __restrict__ output,
      uint32_t const N,                                         
      int const block,                                   
      int const rank,                                   
      uint8_t** __restrict__ buffer_list,                          
      uint32_t const data_offset,                                               
      uint32_t flag_color,
      int64_t data_size_per_phase) {
               
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    uint8_t* rank_buffer = buffer_list[rank];
    Codec codec(thread, rank);
    int block_id = blockIdx.x;
    int grid_size = gridDim.x;
                                                               
                                
    int32x4_t tA[kAtoms];

    BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
    uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
      if constexpr (cast_bf2half) {
        const nv_bfloat162* bf_buf = reinterpret_cast<const nv_bfloat162*>(&tA[i]);
        half2 half_buf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __bfloat1622float2(bf_buf[j]);
          half_buf[j] = __float22half2_rn(f);
        }
        tA[i] = *reinterpret_cast<const int32x4_t*>(half_buf);
      }
    }

                                                               
                                                                               
                                         
    uint32_t comm_data0_offset = data_offset + block_id * Codec::kTransmittedTileSize;
    uint32_t comm_data1_offset = data_size_per_phase + comm_data0_offset;

    uint32_t comm_flags0_offset = block_id * (kWorldSize * sizeof(uint32_t));
    uint32_t comm_flags1_offset = (data_offset / 2) + comm_flags0_offset;

    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, &tA[r * Codec::kRankAtoms]);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(buffer_list[r] + comm_flags0_offset + rank * sizeof(uint32_t));
      set_sync_flag(flag_ptr, flag_color);
    }
                                                               
                                                                        
    int32x4_t tR[Codec::kRankAtoms] = {};
    {
                                                     
      int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(rank_buffer + comm_data0_offset);
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(rank_buffer + comm_flags0_offset);

      for (int r = 0; r < kWorldSize; r++) {
                                        
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

                                                
        codec.recv(&recv_buffer, tA);

        for (int i = 0; i < Codec::kRankAtoms; i++) {
          packed_assign_add<T>(&tR[i], &tA[i]);
        }
      }
    }

                                                             
    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data1_offset + rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, tR);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(buffer_list[r] + comm_flags1_offset + rank * sizeof(uint32_t));
      set_sync_flag(flag_ptr, flag_color);
    }

                                                                              
    {
                                                     
      int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(rank_buffer + comm_data1_offset);
      uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(rank_buffer + comm_flags1_offset);

      for (int r = 0; r < kWorldSize; r++) {
                                        
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

                                                              
        codec.recv(&recv_buffer, &tA[r * Codec::kRankAtoms]);
      }
    }

                                                               
                                  
    BufferResource dst_buffer(output, N * sizeof(T));
    uint32_t dst_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      if constexpr (cast_bf2half) {
        const half2* half_buf = reinterpret_cast<const half2*>(&tA[i]);
        nv_bfloat162 bf16_buf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = __half22float2(half_buf[j]);
          bf16_buf[j] = __float22bfloat162_rn(f);
        }
        buffer_store_dwordx4(*reinterpret_cast<const int32x4_t*>(bf16_buf), dst_buffer.descriptor, dst_offset, 0, 0);
      } else {
        buffer_store_dwordx4(tA[i], dst_buffer.descriptor, dst_offset, 0, 0);
      }
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

}                          
