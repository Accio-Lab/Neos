from enum import IntEnum, auto


class AttnForwardMethod(IntEnum):
    
    MHA = auto()

    
    MLA = auto()

    
    
    MHA_CHUNKED_KV = auto()

    
    
    MHA_ONE_SHOT = auto()

    
    MLA_FUSED_ROPE_ROCM = auto()

    
    MLA_FUSED_ROPE_CPU = auto()

    
    MHA_NPU = auto()

    
    MLA_NPU = auto()

    
    DSA_NPU = auto()
