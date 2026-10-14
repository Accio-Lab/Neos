# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_combine_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
import operator
from typing import Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, const_expr

import sglang.jit_kernel.flash_attention.cute.utils as utils
from .seqlen_info import SeqlenInfo
from cutlass.cute import FastDivmodDivisor


class FlashAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        m_block_size: int = 8,
        k_block_size: int = 64,
        log_max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 4,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param m_block_size: m block size
        :param k_block_size: k block size
        :param log_max_splits: log2 of maximum splits
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.m_block_size = m_block_size
        self.k_block_size = k_block_size
        self.max_splits = 1 << log_max_splits
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16, Float32]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if m_block_size % 8 != 0:
            return False
        max_splits = 1 << log_max_splits
        if max_splits > 256:
            return False
        if (m_block_size * max_splits) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))  
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=async_copy_elems * self.dtype.width,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tOpartial_layout,
            vOpartial_layout,  
        )

        
        lse_copy_bits = Float32.width  
        m_block_smem = (
            128
            if self.m_block_size % 128 == 0
            else (
                64
                if self.m_block_size % 64 == 0
                else (
                    32
                    if self.m_block_size % 32 == 0
                    else (16 if self.m_block_size % 16 == 0 else 8)
                )
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        
        
        

        
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        
        
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle, 0, cute.make_ordered_layout((8, m_block_smem), order=(1, 0))
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.max_splits, self.m_block_size), (0, 1)
        )

        
        self.smem_layout_o = cute.make_ordered_layout(
            (self.m_block_size, self.k_block_size, self.stages), order=(1, 0, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")

        
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )

        
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mO_partial, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mO_partial, mO)
        ]
        
        
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        
        
        LSE_partial_layout_transpose = [2, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )

        
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.m_block_size], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        
        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.m_block_size),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            semaphore_to_reset,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        varlen: cutlass.Constexpr[bool],
    ):
        
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, batch_idx = cute.arch.block_idx()

        
        
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.m_block_size,))
        sO = storage.sO.get_tensor(smem_layout_o)

        
        if const_expr(semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and batch_idx == cute.arch.grid_dim()[2] - 1
            ):
                semaphore_to_reset[0] = 0

        
        num_splits = (
            num_splits_dynamic_ptr[batch_idx]
            if const_expr(num_splits_dynamic_ptr is not None)
            else mLSE_partial.shape[1]
        )
        
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        
        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head

        
        if (const_expr(num_splits_dynamic_ptr is None) or num_splits > 1) and (
            const_expr(not varlen) or m_block * self.m_block_size < max_idx
        ):
            
            
            

            if const_expr(cu_seqlens is None):
                
                mLSE_partial_cur = utils.coord_offset_i64(mLSE_partial, batch_idx, dim=3)
            else:
                
                mLSE_partial_cur = utils.domain_offset_i64((offset, 0, 0), mLSE_partial)
            mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))

            gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
            tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)

            
            cLSE = cute.make_identity_tensor((self.max_splits, self.m_block_size))
            tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)

            
            for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                mi = tLSEcLSE[0, 0, m][1]  
                idx = m_block * self.m_block_size + mi
                if idx < max_idx:
                    
                    if const_expr(not varlen):
                        head_idx, m_idx = divmod(idx, seqlen_divmod)
                    else:
                        head_idx = idx // seqlen
                        m_idx = idx - head_idx * seqlen
                    mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                    for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                        si = tLSEcLSE[0, s, 0][0]  
                        if si < num_splits:
                            cute.copy(
                                gmem_thr_copy_LSE,
                                mLSE_partial_cur_copy[None, si],
                                tLSEsLSE[None, s, m],
                            )
                        else:
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                
            cute.arch.cp_async_commit_group()

            
            
            

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.m_block_size, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            if const_expr(cu_seqlens is None):
                
                mO_partial_cur = utils.coord_offset_i64(mO_partial, batch_idx, dim=4)
            else:
                
                mO_partial_cur = utils.domain_offset_i64((offset, 0, 0, 0), mO_partial)

            
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_fragment(num_rows, cutlass.Int32)
            tOhidx = cute.make_fragment(num_rows, cutlass.Int32)
            tOrOptr = cute.make_fragment(num_rows, cutlass.Int64)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  
                idx = m_block * self.m_block_size + mi
                if const_expr(not varlen):
                    tOhidx[m], tOmidx[m] = divmod(idx, seqlen_divmod)
                else:
                    tOhidx[m] = idx // seqlen
                    tOmidx[m] = idx - tOhidx[m] * seqlen
                tOrOptr[m] = utils.elem_pointer_i64(
                    mO_partial_cur, (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m])
                ).toint()
                if idx >= max_idx:
                    tOhidx[m] = -1

            tOpO = cute.make_fragment(cute.size(tOcO, [2]), cutlass.Boolean)
            if const_expr(not self.is_even_k):
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
            

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

            
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            
            
            

            
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            
            
            
            
            

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_fragment_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)

            
            
            

            lse_sum = cute.make_fragment(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            
            max_valid_split = cute.make_fragment(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = utils.warp_reduce(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    op=cute.arch.fmax,
                    width=threads_per_col,
                )
                
                
                max_valid_idx = -1
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]  
                
                max_valid_split[m] = utils.warp_reduce(max_valid_idx, max, width=threads_per_col)
                
                lse_max_cur = (
                    0.0 if lse_max == -Float32.inf else lse_max
                )  
                LOG2_E = math.log2(math.e)
                lse_sum_cur = 0.0
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = utils.exp2f(ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E))
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale  
                lse_sum_cur = utils.warp_reduce(lse_sum_cur, operator.add, width=threads_per_col)
                lse_sum[m] = utils.logf(lse_sum_cur) + lse_max
                
                inv_sum = (
                    0.0 if (lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur) else 1.0 / lse_sum_cur
                )
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.m_block_size:
                        sMaxValidSplit[mi] = max_valid_split[m]

            
            
            

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    
                    mLSE_cur = utils.coord_offset_i64(mLSE, batch_idx, dim=2)
                else:
                    
                    mLSE_cur = utils.domain_offset_i64((offset, 0), mLSE)
                if k_block == 0:  
                    for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                        if ts2rcLSE[0, 0, m][0] == 0:  
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.m_block_size + mi
                            if idx < max_idx:
                                if const_expr(not varlen):
                                    head_idx, m_idx = divmod(idx, seqlen_divmod)
                                else:
                                    head_idx = idx // seqlen
                                    m_idx = idx - head_idx * seqlen
                                mLSE_cur[m_idx, head_idx] = lse_sum[m]

            
            
            

            cute.arch.sync_threads()

            
            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1])):
                thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])

            tOrO_partial = cute.make_fragment_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_fragment_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                
                scale = cute.make_fragment(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]  

                
                split_to_load = s + self.stages - 1
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                
                cute.arch.cp_async_wait_group(self.stages - 1)
                
                
                cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            
            
            

            rO = cute.make_fragment_like(tOrO, self.dtype)
            rO.store(tOrO.load().to(self.dtype))
            if const_expr(cu_seqlens is None):
                
                mO_cur = utils.coord_offset_i64(mO, batch_idx, dim=3)
            else:
                
                mO_cur = utils.domain_offset_i64((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
            
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            
            for m in cutlass.range(num_rows, unroll_full=True):
                if tOhidx[m] >= 0:
                    mO_cur_copy = cute.tiled_divide(
                        mO_cur[tOmidx[m], None, tOhidx[m]], (elems_per_store,)
                    )
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        k_idx = tOcO[0, 0, k][1] // elems_per_store
                        if const_expr(self.is_even_k) or tOpO[k]:
                            cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOpO: cute.Tensor,
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if const_expr(self.is_even_k) or tOpO[k]:
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            
                            utils.coord_offset_i64(mO_partial_cur_copy, split, dim=2)[None, k_idx],
                            tOsO_partial_cur[None, m, k],
                        )
