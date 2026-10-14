# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def clz(x: Int32) -> Int32:
    
    
    
    
    
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res
