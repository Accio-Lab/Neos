import unittest

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    draft_decode_set_expand_metadata,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    def __init__(
        self,
        page_size=1,
        num_heads=2,
        head_dim=8,
    ):
        self.device = "cuda"
        self.dtype = torch.float16
        self.kv_cache_dtype = torch.float16
        self.is_hybrid_swa = False
        self.attention_chunk_size = None
        attention_arch = AttentionArch.MHA
        
        max_batch_size = 160
        
        max_context_len = 2048
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "is_encoder_decoder": False,
                "is_local_attention_model": False,
            },
        )()
        self.sliding_window_size = None
        
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                
                "size": max_batch_size,
                
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()
        self.page_size = page_size
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.dtype,
            head_num=num_heads,
            head_dim=head_dim,
            layer_num=1,  
            device=self.device,
            enable_memory_saver=False,
        )
        
        self.server_args = ServerArgs(model_path="dummy")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttentionBackend(CustomTestCase):
    def setUp(self):
        
        self.batch_size = 2
        self.seq_len = 256
        self.num_heads = 2
        self.head_dim = 8
        self.device = "cuda"
        self.dtype = torch.float16

    def _init_model_runner(self, page_size=1):
        self.model_runner = MockModelRunner(
            page_size=page_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.backend = FlashAttentionBackend(self.model_runner)
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)
        self.model_runner.model_config.num_attention_heads = self.num_heads

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len, page_size):
        
        
        self.req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
            + page_size
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        return (
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
        )

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
        else:  
            output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape, output_ref=None):
        """Verify output tensor shape, dtype, and values."""
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )
        self.assertEqual(output.dtype, self.dtype)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

        if output_ref is not None:
            if not torch.allclose(output, output_ref, atol=1e-1, rtol=0.0):
                
                diff_mask = ~torch.isclose(output, output_ref, atol=1e-1, rtol=0.0)

                
                if diff_mask.any():
                    first_mismatch_idx = diff_mask.nonzero()[0]
                    print(
                        "First mismatch at index:", tuple(first_mismatch_idx.tolist())
                    )
                    print("output:", output[tuple(first_mismatch_idx.tolist())])
                    print("output_ref:", output_ref[tuple(first_mismatch_idx.tolist())])
                raise AssertionError(
                    "Attention output is not close to the torch native backend output"
                )

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0, page_size=1):
        """Create a forward batch for testing based on mode and lengths."""
        self._init_model_runner(page_size=page_size)

        
        q_len = q_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * self.batch_size
            out_cache_end = total_len * self.batch_size

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [prefix_len] * self.batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [prefix_len] * self.batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor(
                    [q_len] * self.batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor(
                    [q_len] * self.batch_size, device="cpu"
                ),
                attn_backend=self.backend,
            )
        else:  
            decode_len = q_len  
            total_len = self.seq_len + decode_len
            if mode == ForwardMode.DECODE and page_size > 1:
                
                out_cache_start = (
                    self.batch_size * self.seq_len // page_size + 1
                ) * page_size
                
                out_cache_end = out_cache_start + decode_len * page_size
            else:
                out_cache_start = self.batch_size * self.seq_len
                out_cache_end = self.batch_size * total_len

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.tensor(
                    [out_cache_start, out_cache_end], device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                attn_backend=self.backend,
            )

        
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

        
        self._mock_write_to_req_to_token_pool(self.batch_size, total_len, page_size)
        
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        
        cache_k = torch.ones(
            self.batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * cache_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _run_attention_test(self, mode, q_len, prefix_len=0, page_size=1):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
            page_size: Page size for the KV cache
        """
        layer = self._create_attention_layer()

        
        forward_batch = self._create_forward_batch(mode, q_len, prefix_len, page_size)

        
        q, k, v = self._create_qkv_tensors(self.batch_size * q_len)

        
        
        
        if mode == ForwardMode.EXTEND:
            if prefix_len > 0:
                self._setup_kv_cache(forward_batch, layer, prefix_len)
        else:
            self._setup_kv_cache(forward_batch, layer, self.seq_len)

        self.backend.init_forward_metadata(forward_batch)

        if mode == ForwardMode.EXTEND:
            expected_shape = (
                self.batch_size * q_len,
                self.num_heads * self.head_dim,
            )
            output = self.backend.forward_extend(q, k, v, layer, forward_batch)
        else:
            expected_shape = (self.batch_size, self.num_heads * self.head_dim)
            output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        output_ref = self._run_reference_forward(
            mode, q, k, v, layer, forward_batch, expected_shape
        )

        self._verify_output(output, expected_shape, output_ref)

        return output

    def test_forward_extend(self):
        """Test the standard extend operation."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens."""
        prefix_len = self.seq_len // 2
        extend_len = self.seq_len - prefix_len
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )

    def test_forward_extend_with_page_size_greater_than_1(self):
        """Test extending from cached prefix tokens with page size greater than 1."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len, page_size=64)

    def test_forward_decode_with_page_size_greater_than_1(self):
        """Test decode operation with page size greater than 1."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1, page_size=64)


class TestUpdateDraftDecodeSetExpandMetadata(CustomTestCase):
    """
    All the test cases examples have 1 additional cache location than the decode length.
    This is to align with the current allocation logic. It does not affect the correctness.
    """

    def test_draft_decode_set_expand_metadata(self):
        bs, topk, page_size = 1, 2, 4

        cases = [
            (
                torch.tensor(
                    [
                        [23, 24],
                        [31, 32],
                    ],
                    dtype=torch.int32,
                ),
                torch.tensor(
                    [
                        [5, 6],
                        [7, 8],
                    ],
                    dtype=torch.int32,
                ),
                1,
            ),
            
            
            
            
            (
                torch.tensor(
                    [
                        [27, 28, 29, 30, 31, 32],
                        [35, 36, 37, 38, 39, 40],
                    ],
                    dtype=torch.int32,
                ),
                torch.tensor(
                    [
                        [6, 7, 8, 0, 0, 0],
                        [8, 9, 10, 0, 0, 0],
                    ],
                    dtype=torch.int32,
                ),
                5,
            ),
        ]

        last_page_lens = torch.tensor([3], dtype=torch.int32)
        for cache_loc, expected_page_table, decode_length in cases:
            cache_seqlens_int32 = torch.zeros(bs * topk, dtype=torch.int32)
            page_table = torch.zeros_like(cache_loc, dtype=torch.int32)
            draft_decode_set_expand_metadata(
                cache_seqlens_int32=cache_seqlens_int32,
                page_table=page_table,
                last_page_lens=last_page_lens,
                decode_length=decode_length,
                cache_loc=cache_loc,
                topk=topk,
                page_size=page_size,
            )

            expected_cache_seqlens = torch.tensor(
                [decode_length + 3, decode_length + 3], dtype=torch.int32
            )
            self.assertTrue(torch.equal(cache_seqlens_int32, expected_cache_seqlens))
            self.assertTrue(torch.equal(page_table, expected_page_table))

    def test_update_draft_decode_set_expand_metadata_multi_batch(self):
        """
        Ensure expand metadata works when batch size > 1 and last pages differ.
        """
        bs, topk, decode_length, page_size = 3, 2, 3, 4
        cache_loc = torch.tensor(
            [
                
                [1, 2, 3, 4],
                [6, 7, 8, 9],
                
                [3, 8, 9, 10],
                [14, 15, 16, 17],
                
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            dtype=torch.int32,
        )
        cache_seqlens_int32 = torch.zeros(bs * topk, dtype=torch.int32)
        last_page_lens = torch.tensor([1, 3, 0], dtype=torch.int32)
        page_table = torch.zeros_like(cache_loc, dtype=torch.int32)
        draft_decode_set_expand_metadata(
            cache_seqlens_int32=cache_seqlens_int32,
            page_table=page_table,
            last_page_lens=last_page_lens,
            decode_length=decode_length,
            cache_loc=cache_loc,
            topk=topk,
            page_size=page_size,
        )

        expected_cache_seqlens = torch.tensor([4, 4, 6, 6, 3, 3], dtype=torch.int32)
        expected_page_table = torch.tensor(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [0, 2, 0, 0],
                [3, 4, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(cache_seqlens_int32, expected_cache_seqlens))
        self.assertTrue(torch.equal(page_table, expected_page_table))


if __name__ == "__main__":
    unittest.main()
