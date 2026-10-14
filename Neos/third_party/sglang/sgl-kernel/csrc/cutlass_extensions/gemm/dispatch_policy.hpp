                                                                                                               

#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

                                                                              

                                                               
                                                                               
                                                             
                                       
template <int ScaleGranularityM = 0>
struct KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum : KernelTmaWarpSpecializedCooperative {};

                                                                          
                                                                  
template <
    int Stages_,
    class ClusterShape_ = Shape<_1, _1, _1>,
    class KernelSchedule = KernelTmaWarpSpecialized,
    int ScaleGranularityM = 0                                                               
                                                                                             
                                                                                    
    >
struct MainloopSm90TmaGmmaWarpSpecializedBlockScalingSubGroupMFP8
    : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
      cute::
          is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<ScaleGranularityM>>,
      "KernelSchedule must be one of the warp specialized policies");
};

                                                                              

}                            
