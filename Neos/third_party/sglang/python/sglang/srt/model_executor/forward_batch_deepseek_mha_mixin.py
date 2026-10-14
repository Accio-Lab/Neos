


from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton


class ForwardBatchDeepSeekMHAMixin:
    
    
    attn_attend_prefix_cache: Optional[bool] = None
    
    num_prefix_chunks: Optional[int] = None
    
    prefix_chunk_idx: Optional[int] = None
    
    prefix_chunk_len: Optional[int] = None
    
    prefix_chunk_starts: Optional[torch.Tensor] = None
    
    prefix_chunk_seq_lens: Optional[torch.Tensor] = None
    
    prefix_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    
    prefix_chunk_max_seq_lens: Optional[List[int]] = None
    
    prefix_chunk_num_tokens: Optional[List[int]] = None
    
    prefix_chunk_kv_indices: Optional[List[torch.Tensor]] = None
    
    
    mha_return_lse: Optional[bool] = None
    
    mha_one_shot: Optional[bool] = None
    
    mha_one_shot_kv_indices: Optional[torch.Tensor] = None

    def get_max_chunk_capacity(self):
        
        
        return 128 * 1024

    def set_prefix_chunk_idx(self, idx: int):
        self.prefix_chunk_idx = idx

    def set_attn_attend_prefix_cache(self, attn_attend_prefix_cache: bool):
        self.attn_attend_prefix_cache = attn_attend_prefix_cache

    def prepare_chunked_kv_indices(self, device: torch.device):
        self.prefix_chunk_kv_indices = []
        for idx in range(self.num_prefix_chunks):
            chunk_starts = self.prefix_chunk_starts[idx]
            chunk_seq_lens = self.prefix_chunk_seq_lens[idx]
            chunk_cu_seq_lens = self.prefix_chunk_cu_seq_lens[idx]
            num_chunk_tokens = self.prefix_chunk_num_tokens[idx]

            chunk_kv_indices = torch.empty(
                num_chunk_tokens, dtype=torch.int32, device=device
            )

            create_chunked_prefix_cache_kv_indices[(self.batch_size,)](
                self.req_to_token_pool.req_to_token,
                self.req_pool_indices,
                chunk_starts,
                chunk_seq_lens,
                chunk_cu_seq_lens,
                chunk_kv_indices,
                self.req_to_token_pool.req_to_token.shape[1],
            )
            self.prefix_chunk_kv_indices.append(chunk_kv_indices)

    
    
    
    
    
    
    
    def get_prefix_chunk_seq_lens(
        self, prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
    ):
        device = prefix_lens.device
        prefix_chunk_starts = (
            torch.arange(num_prefix_chunks, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, self.batch_size)
            * prefix_chunk_len
        )
        prefix_chunk_ends = torch.min(
            prefix_lens.unsqueeze(0),
            prefix_chunk_starts + prefix_chunk_len,
        ).to(torch.int32)

        prefix_chunk_seq_lens = (
            (prefix_chunk_ends - prefix_chunk_starts).clamp(min=0).to(torch.int32)
        )

        return prefix_chunk_starts, prefix_chunk_seq_lens

    
    
    def prepare_chunked_prefix_cache_info(self, device: torch.device):

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        assert isinstance(
            self.token_to_kv_pool, MLATokenToKVPool
        ), "Currently chunked prefix cache can only be used by Deepseek models"

        if not any(self.extend_prefix_lens_cpu):
            self.num_prefix_chunks = 0
            return

        if self.prefix_chunk_len is not None:
            
            return

        self.prefix_chunk_idx = -1

        
        chunk_capacity = self.get_max_chunk_capacity()
        self.prefix_chunk_len = chunk_capacity // self.batch_size

        self.num_prefix_chunks = (
            max(self.extend_prefix_lens_cpu) + self.prefix_chunk_len - 1
        ) // self.prefix_chunk_len

        
        prefix_chunk_starts_cuda, prefix_chunk_seq_lens_cuda = (
            self.get_prefix_chunk_seq_lens(
                self.extend_prefix_lens,
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        _, prefix_chunk_seq_lens_cpu = self.get_prefix_chunk_seq_lens(
            torch.tensor(self.extend_prefix_lens_cpu),
            self.num_prefix_chunks,
            self.prefix_chunk_len,
        )
        self.prefix_chunk_starts = prefix_chunk_starts_cuda
        self.prefix_chunk_seq_lens = prefix_chunk_seq_lens_cuda

        
        self.prefix_chunk_cu_seq_lens = torch.zeros(
            self.num_prefix_chunks,
            self.batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        self.prefix_chunk_cu_seq_lens[:, 1:] = prefix_chunk_seq_lens_cuda.cumsum(
            dim=1
        ).to(torch.int32)
        self.prefix_chunk_max_seq_lens = prefix_chunk_seq_lens_cpu.max(
            dim=1
        ).values.tolist()

        self.prefix_chunk_num_tokens = prefix_chunk_seq_lens_cpu.sum(dim=1).tolist()
        assert max(self.prefix_chunk_num_tokens) <= self.get_max_chunk_capacity()

        
        self.prepare_chunked_kv_indices(device)

    def fetch_mha_one_shot_kv_indices(self):
        if self.mha_one_shot_kv_indices is not None:
            return self.mha_one_shot_kv_indices
        batch_size = self.batch_size
        paged_kernel_lens_sum = sum(self.seq_lens_cpu)
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token_pool.req_to_token.shape[1],
        )
        self.mha_one_shot_kv_indices = kv_indices
        return kv_indices


@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  
    req_pool_indices_ptr,  
    chunk_start_idx_ptr,  
    chunk_seq_lens_ptr,  
    chunk_cu_seq_lens_ptr,  
    chunk_kv_indices_ptr,  
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )
