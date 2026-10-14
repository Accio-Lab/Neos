# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch


def rewind_kv_cache(
    accepted_counts,
    prefill_status,
    last_kv_block_offset,
    kv_length_offsets,
    kv_block_counts,
    last_kv_block_id,
    kv_block_ids,
    num_speculative_tokens,
    block_size_tokens,
    num_active_requests=None,
):
    """Update the KV cache bookkeeping for speculative decoding.

    After forward pass with speculative tokens, some tokens may be rejected.
    This function "rewinds" the KV cache bookkeeping to reflect only the accepted tokens.

    When speculative tokens are rejected, we need to:
    1. Update kv_length_offsets (total sequence length)
    2. Update last_kv_block_offset (position within last block)
    3. If rewinding crosses a block boundary:
       - Reduce kv_block_counts
       - Update last_kv_block_id to point to the previous block
       - Clear the entry in kv_block_ids for the released block

    Mutates the input tensors in-place.

    Returns (blocks_to_release, remove_mask).
    """
    N = accepted_counts.shape[0]
    if num_active_requests is None:
        num_active_requests = N

    
    
    accepted_list = accepted_counts.tolist()
    prefill_list = prefill_status.tolist()
    offset_list = last_kv_block_offset.tolist()
    length_list = kv_length_offsets.tolist()
    block_count_list = kv_block_counts.tolist()
    last_block_list = last_kv_block_id.tolist()
    kv_block_ids_list = kv_block_ids.tolist()
    max_blocks = kv_block_ids.shape[1]

    blocks_to_release = torch.empty_like(last_kv_block_id)
    remove_mask = torch.empty(N, device=accepted_counts.device, dtype=torch.bool)

    for i in range(N):
        if i >= num_active_requests:
            blocks_to_release[i] = 0
            remove_mask[i] = False
            continue

        accepted = accepted_list[i]
        prefill = prefill_list[i]
        last_offset = offset_list[i]
        kv_length = length_list[i]
        block_count = block_count_list[i]
        last_block = last_block_list[i]

        
        
        
        num_to_rewind = 0 if prefill == 1 else num_speculative_tokens - accepted

        
        
        diff = last_offset - num_to_rewind
        remove = diff < 0

        
        new_offset = diff % block_size_tokens
        last_kv_block_offset[i] = new_offset
        kv_length_offsets[i] = kv_length - num_to_rewind

        
        
        
        
        
        
        blocks_to_release[i] = last_block

        
        new_block_count = block_count - 1 if remove else block_count
        kv_block_counts[i] = new_block_count

        
        prev_idx = max(new_block_count - 1, 0)
        prev_block_id = kv_block_ids_list[i][prev_idx]
        last_kv_block_id[i] = prev_block_id if remove else last_block

        
        scatter_idx = min(new_block_count, max_blocks - 1)
        if remove:
            kv_block_ids[i, scatter_idx] = -1

        remove_mask[i] = remove

    return blocks_to_release, remove_mask



def verify_speculative_tokens(
    input_tokens, output_tokens, num_decode_requests, num_prefill_requests, num_speculative_tokens
):
    """Verify speculative tokens against input tokens and compute acceptance.

    Creates an accepted tokens mask where:
    - For prefill requests, the token is always accepted.
    - For decode requests, the first token (base token) is always accepted, then we compare
      sampled tokens with input tokens and accept consecutive matches.
    Then finds the index of the last accepted token per request.

    Example (assume 1, 2, and 0 spec tokens are accepted in the first 3 decode requests):
        input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]  # Size 11
        Output tokens                       [ a6o a7o  a8o |  b40   b5o  b6o   |  c7o  c8o   c9o   |     d3o     |         e5o        ]
        Output tokens right shift           [ d3o a6o  a7o |  a8o   b40  b5o   |  b6o  c7o   c8o   |     c9o     |         d3o        ]
        Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
        Last one indices                    [      1       |         5         |        6          |      9      |         10         ]

    Returns:
        tuple: (last_one_indices, accepted_tokens_mask, input_tokens) where
            last_one_indices contains the index of the last accepted token per request.
    """
    if input_tokens.ndim == 2:
        input_tokens = input_tokens.squeeze(0)

    stride = num_speculative_tokens + 1
    active_request_count = num_decode_requests + num_prefill_requests
    decode_len = num_decode_requests * stride

    
    accepted_tokens_mask = torch.zeros_like(input_tokens, dtype=torch.bool)

    
    decode_mask_2d = None
    if num_decode_requests > 0:
        decode_inputs = input_tokens[:decode_len].reshape(num_decode_requests, stride)
        decode_outputs = output_tokens[:decode_len].reshape(num_decode_requests, stride)

        
        decode_outputs_shifted = decode_outputs.roll(1, dims=1)
        decode_mask_2d = decode_inputs == decode_outputs_shifted
        
        decode_mask_2d[:, 0] = True
        
        decode_mask_2d = decode_mask_2d.cummin(dim=1).values
        accepted_tokens_mask[:decode_len] = decode_mask_2d.flatten()

    
    if num_prefill_requests > 0:
        accepted_tokens_mask[decode_len:] = True

    last_one_indices = torch.full(
        (active_request_count,), -1, device=input_tokens.device, dtype=torch.long
    )

    if num_decode_requests > 0:
        
        local_last_indices = decode_mask_2d.sum(dim=1) - 1
        row_offsets = torch.arange(num_decode_requests, device=input_tokens.device) * stride
        last_one_indices[:num_decode_requests] = row_offsets + local_last_indices

    if num_prefill_requests > 0:
        prefill_valid = torch.nonzero(accepted_tokens_mask[decode_len:]).squeeze(-1) + decode_len
        last_one_indices[num_decode_requests:] = prefill_valid

    return last_one_indices, accepted_tokens_mask, input_tokens



def prepare_next_forward_pass(
    num_decode_requests,
    output_tokens,
    required_logit_indices,
    last_one_indices,
    accepted_tokens_mask,
    input_tokens,
    sampled_tokens_buf,
    last_accepted_seq_buf,
    accepted_tokens_per_request,
    accepted_token_counts,
    num_speculative_tokens,
):
    """Prepare data for the next forward pass after speculative token verification.

    For each active request:
    - Store the final sampled tokens for the next forward pass.
    - Store the last accepted positions in the packed sequence for serial
      MTP computation after verification.

    For decode requests, extract accepted tokens and counts:
        input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]
        Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
        Accepted tokens                     [   [a6s  -1]  |     [b4s  b5s]    |     [-1  -1]      ]  # Only decode requests (prefill defaults to -1)
        Accepted token counts               [      1       |         2         |         0         ]  # Prefill defaults to 0

    Writes results into the pre-allocated buffers provided by the caller.
    """
    active_request_count = last_one_indices.shape[0]
    stride = num_speculative_tokens + 1

    for pid in range(active_request_count):
        idx = last_one_indices[pid].item()

        
        sampled_tokens_buf[pid] = output_tokens[idx]

        
        
        last_accepted_seq_buf[pid] = required_logit_indices[idx]

        
        
        
        if pid < num_decode_requests:
            base = pid * stride
            
            for s in range(num_speculative_tokens):
                pos = base + 1 + s
                if accepted_tokens_mask[pos]:
                    accepted_tokens_per_request[pid, s] = input_tokens[pos]
                else:
                    accepted_tokens_per_request[pid, s] = -1

            count = 0
            for s in range(num_speculative_tokens):
                if accepted_tokens_per_request[pid, s].item() != -1:
                    count += 1
            accepted_token_counts[pid] = count


def mamba_state_selective_copy(
    intermediate_states, current_states, prefill_status, state_idx, accepted_counts, num_layers
):
    """Mamba speculative rewind state update.

    For each decode request, copies
    `intermediate[layer, slot, accepted_count, ...]` →
    `current[layer, slot, ...]` for every Mamba layer.
    """
    N = prefill_status.shape[0]
    for i in range(N):
        if prefill_status[i].item() == 1:
            continue
        slot = state_idx[i].item()
        accepted = accepted_counts[i].item()
        for layer in range(num_layers):
            current_states[layer, slot] = intermediate_states[layer, slot, accepted]
