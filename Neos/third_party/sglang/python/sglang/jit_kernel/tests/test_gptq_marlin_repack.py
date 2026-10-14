import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.utils import (
    gptq_quantize_weights,
    pack_rows,
    sort_weights,
)
from sglang.test.test_marlin_utils import get_weight_perm, marlin_weights

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]


@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_repack(
    k_chunk, n_chunk, quant_type, group_size, act_order, mnk_factors
):
    m_factor, n_factor, k_factor = mnk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if size_k % group_size != 0:
        pytest.skip("size_k must be divisible by group_size")

    
    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order
    )

    q_w_gptq = pack_rows(q_w, quant_type.size_bits, size_k, size_n)

    
    
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    marlin_layout_perm = get_weight_perm(quant_type.size_bits)
    q_w_marlin_ref = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, marlin_layout_perm
    )

    
    jit_output = gptq_marlin_repack(
        q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits
    )

    torch.cuda.synchronize()

    
    torch.testing.assert_close(jit_output, q_w_marlin_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
