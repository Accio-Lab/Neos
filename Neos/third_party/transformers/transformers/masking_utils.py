# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable

import torch
import torch.nn.functional as F

from .cache_utils import Cache
from .configuration_utils import PreTrainedConfig
from .utils import is_torch_xpu_available, logging
from .utils.generic import GeneralInterface, is_flash_attention_requested
from .utils.import_utils import is_torch_flex_attn_available, is_torch_greater_or_equal, is_tracing


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    
    BlockMask = torch.Tensor

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


logger = logging.get_logger(__name__)


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a full bidirectional mask.

    NOTE: It is important to keep an index-based version for non-vmap expansion.
    """
    return q_idx >= 0


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size

    return inner_mask


def blockwise_overlay(block_sequence_ids: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a blockwise masking pattern. Instead of a single
    token, each block consists of arbitrary length tokens. In causal setup, each block
    can attend to prev block causally and can't attend to future blocks. Within one block
    the attention is always bidirectional.
    Mostly used in MLLMs when non-text data attends bidirectionally to itself.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        
        q_group = block_sequence_ids[batch_idx, q_idx]
        kv_group = block_sequence_ids[batch_idx, kv_idx]
        return (q_group == kv_group) & (q_group >= 0)

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def sliding_window_bidirectional_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a bidirectional sliding window pattern.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (inclusive) sliding window size (distance <= sliding_window)."""
        return abs(q_idx - kv_idx) <= sliding_window

    return inner_mask


def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a bidirectional sliding window mask.
    """
    return and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)


def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        
        
        
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask


def prepare_padding_mask(attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int) -> torch.Tensor | None:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask


def maybe_pad_block_sequence_ids(
    block_sequence_ids: torch.Tensor, attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int
) -> torch.Tensor:
    """
    Pads the `block_sequence_ids` in case the total length is less than `kv_length`.
    Usually that happens with `StaticCache` generation or generating without cache.
    Pads to the right with `-1`.
    """
    if (padding_length := kv_length + kv_offset - block_sequence_ids.shape[-1]) > 0:
        block_sequence_ids = F.pad(block_sequence_ids, pad=(0, padding_length), value=-1)
    return block_sequence_ids


def _can_skip_causal_mask_xpu(
    padding_mask: torch.Tensor | None,
    query_length: int,
    kv_length: int,
    local_attention_size: int | None,
) -> bool:
    """
    XPU-specific logic for determining if we can skip causal mask creation.

    For XPU devices, we have special handling:
    - Single query tokens (query_length == 1) use the same logic as CUDA
    - Multi-query tokens can skip if padding_mask is provided and correctly structured
      The mask must have all True values in the query window and all False after
    """

    if is_tracing(padding_mask):
        return False

    
    if local_attention_size is not None and kv_length >= local_attention_size:
        return False

    if padding_mask is None:
        
        return query_length == 1 or kv_length == query_length

    
    if query_length == 1:
        
        return padding_mask.all()

    
    
    return padding_mask[:, :query_length].all() and not padding_mask[:, query_length:].any()


def _ignore_causal_mask_sdpa(
    padding_mask: torch.Tensor | None,
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: int | None = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    if _is_torch_xpu_available:
        
        
        
        
        return _can_skip_causal_mask_xpu(padding_mask, query_length, kv_length, local_attention_size)
    
    
    
    
    if (
        not is_tracing(padding_mask)
        
        and (query_length == 1 or kv_length == query_length)
        
        and (local_attention_size is None or kv_length < local_attention_size)
        
        and (padding_mask is None or padding_mask.all())
    ):
        return True

    return False


def _can_skip_bidirectional_mask_xpu(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None,
) -> bool:
    """
    XPU-specific logic for determining if we can skip bidirectional mask creation.

    For XPU devices, we have special handling:
    - Skip if no padding and no local attention constraint
    """

    if is_tracing(padding_mask):
        return False

    
    if local_attention_size is not None and kv_length >= local_attention_size:
        return False

    if padding_mask is None:
        
        return True

    
    return padding_mask.all()


def _ignore_bidirectional_mask_sdpa(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None = None,
) -> bool:
    """
    Detects whether the bidirectional mask can be ignored in case PyTorch's SDPA is used.

    In case no token is masked in the 2D `padding_mask` argument and no local attention constraint applies
    (i.e. `local_attention_size` is None or `kv_length < local_attention_size`), we skip mask creation,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if _is_torch_xpu_available:
        
        
        return _can_skip_bidirectional_mask_xpu(padding_mask, kv_length, local_attention_size)

    
    
    if (
        not is_tracing(padding_mask)
        and (padding_mask is None or padding_mask.all())
        
        and (local_attention_size is None or kv_length < local_attention_size)
    ):
        return True

    return False


def _vmap_expansion_sdpa(mask_function: Callable) -> Callable:
    """
    Used to vmap our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).
    """
    
    dimensions = [(None, None, None, 0), (None, None, 0, None), (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    """
    Used to broadcast our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Allows the usage of any index-based mask function without relying on vmap.

    NOTE: This is limited to index based functions only and is not guaranteed to work otherwise.

    Reference:
        - https://github.com/huggingface/optimum-onnx/blob/c123e8f4fab61b54a8e0e31ce74462bcacca576e/optimum/exporters/onnx/model_patcher.py#L362-L365
    """
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices


def sdpa_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor | None:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
        use_vmap (`bool`, optional):
            Whether to use `vmap` during the mask construction or not. Allows powerful custom patterns that may not be
            index-based (for the cost of speed performance). By default `False`.
        device (`torch.device` or `str`, optional):
            An optional device to create the mask on.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, q_length=5, kv_length=5)
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [ True,  True,  True,  True, False],
                  [ True,  True,  True,  True,  True]]]])
    ```

    ## Creating a sliding window mask:

    To create the following sliding window mask (`sliding_window=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, q_length=5, kv_length=5, mask_function=sliding_window_causal_mask_function(3))
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [False,  True,  True,  True, False],
                  [False, False,  True,  True,  True]]]])
    ```

    ## Creating a chunked attention mask

    To create the following chunked attention mask (`chunk_size=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ⬚ ⬚ ■ ⬚
        4 ⬚ ⬚ ⬚ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, q_length=5, kv_length=5, mask_function=chunked_causal_mask_function(3, torch.zeros(1, dtype=int)))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    
    
    
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask, kv_length, local_size):
        return None

    
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset

    
    
    if not use_vmap:
        
        attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
        
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)

    
    elif _is_torch_greater_or_equal_than_2_6:
        
        
        
        with TransformGetItemToIndex():
            attention_mask = _vmap_expansion_sdpa(mask_function)(batch_arange, head_arange, q_arange, kv_arange)

    
    else:
        raise ValueError(
            "The vmap functionality for mask creation is only supported from torch>=2.6. "
            "Please update your torch version or use `use_vmap=False` with index-based masks."
        )

    
    
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        attention_mask = attention_mask | torch.all(~attention_mask, dim=-1, keepdim=True)

    return attention_mask


def eager_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    allow_is_bidirectional_skip: bool = False,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
        use_vmap (`bool`, optional):
            Whether to use `vmap` during the mask construction or not. Allows powerful custom patterns that may not be
            index-based (for the cost of speed performance). By default `False`.
        device (`torch.device` or `str`, optional):
            An optional device to create the mask on.
    """
    
    _ = kwargs.pop("allow_is_causal_skip", None)
    _ = kwargs.pop("allow_torch_fix", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        allow_torch_fix=False,
        use_vmap=use_vmap,
        device=device,
        **kwargs,
    )
    
    if mask is not None:
        min_dtype = torch.finfo(dtype).min
        
        mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask


def flash_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    """
    Create the attention mask necessary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    We just slice it in case of sliding window.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    if attention_mask is not None:
        
        attention_mask = attention_mask[:, -kv_length:]
        
        
        if attention_mask.all():
            attention_mask = None

    return attention_mask


def flex_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> BlockMask:
    """
    Create a 4D block mask which is a compressed representation of the full 4D block causal mask. BlockMask is essential
    for performant computation of flex attention. See: https://pytorch.org/blog/flexattention/

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        device (`torch.device` or `str`, optional):
            An optional device to create the mask on.
    """
    
    if attention_mask is not None:
        
        
        pad_len = ((attention_mask.shape[1] // flex_default_block_size) + 1) * flex_default_block_size
        pad_len = pad_len - attention_mask.shape[1]
        if not _is_torch_greater_or_equal_than_2_6 and pad_len > 0:
            attention_mask = torch.nn.functional.pad(attention_mask, value=0, pad=(0, pad_len))

        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    
    mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)

    
    block_mask = create_block_mask(
        mask_mod=mask_function,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=device,
        _compile=_is_torch_greater_or_equal_than_2_6,
    )
    return block_mask


class AttentionMaskInterface(GeneralInterface):
    
    
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flash_attention_4": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }



ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].

        If the there is only one sequence in each batch item (and we don't compile), then we return `None` indicating
        no packed sequences. This is the same as [[0, 0, 0, 0, 0, 0]] for the example above.
    """
    
    
    
    
    
    first_dummy_value = position_ids[:, :1] - 1  
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    
    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None

    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    layer_idx: int | None,
    encoder_hidden_states: torch.Tensor | None = None,
) -> tuple[bool, torch.Tensor | BlockMask | None, int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the kv length.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None

    
    
    
    
    
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None

    
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    q_length = inputs_embeds.shape[1]
    
    if past_key_values is not None:
        q_offset = past_key_values.get_seq_length()
        
        
        q_offset = q_offset.to(inputs_embeds.device) if isinstance(q_offset, torch.Tensor) else q_offset
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
    
    else:
        q_offset = 0
        
        if attention_mask is None:
            
            kv_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else q_length
            kv_offset = 0
        
        
        
        
        
        
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

    
    
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = inputs_embeds.shape[0]
        
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset


def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
    """
    
    
    if not getattr(config, "is_causal", True):
        return create_bidirectional_mask(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    
    
    
    use_vmap = False

    
    
    if _is_torch_xpu_available:
        
        allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and q_length == 1)
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    
    
    
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False
    if block_sequence_ids is not None:
        block_sequence_ids = maybe_pad_block_sequence_ids(block_sequence_ids, attention_mask, kv_length, kv_offset)
        mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
        allow_is_causal_skip = False

    
    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  
        dtype=dtype,  
        config=config,  
        use_vmap=use_vmap,  
        device=device,
    )
    return causal_mask


def create_bidirectional_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    **kwargs,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard bidirectional mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the batch size, kv length and dtype.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
    """
    
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    
    early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = _preprocess_mask_arguments(
        config, inputs_embeds, attention_mask, past_key_values, None, layer_idx, encoder_hidden_states
    )
    if early_exit:
        return attention_mask

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, dtype = embeds.shape[0], embeds.dtype
    
    
    
    device = inputs_embeds.device
    mask_factory_function = bidirectional_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    
    allow_is_bidirectional_skip = True
    
    
    
    use_vmap = False

    
    
    
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True

    
    attention_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        dtype=dtype,  
        config=config,  
        use_vmap=use_vmap,  
        device=device,
    )
    return attention_mask


def create_sliding_window_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
    """
    
    
    if not getattr(config, "is_causal", True):
        return create_bidirectional_sliding_window_mask(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    
    
    
    use_vmap = False
    
    
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    
    
    
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False
    if block_sequence_ids is not None:
        block_sequence_ids = maybe_pad_block_sequence_ids(block_sequence_ids, attention_mask, kv_length, kv_offset)
        mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
        allow_is_causal_skip = False

    
    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  
        local_size=sliding_window,  
        dtype=dtype,  
        config=config,  
        use_vmap=use_vmap,  
        device=device,
    )
    return causal_mask


def create_bidirectional_sliding_window_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    **kwargs,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard bidirectional sliding window mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the batch size, kv length and dtype.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
    """
    
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    
    early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = _preprocess_mask_arguments(
        config, inputs_embeds, attention_mask, past_key_values, None, layer_idx, encoder_hidden_states
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, dtype, device = embeds.shape[0], embeds.dtype, embeds.device
    mask_factory_function = sliding_window_bidirectional_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False
    allow_is_bidirectional_skip = True

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True

    attention_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        local_size=sliding_window,  
        dtype=dtype,  
        config=config,  
        use_vmap=use_vmap,  
        device=device,
    )
    return attention_mask


def create_chunked_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "chunked_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
    """
    
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    
    if is_flash_attention_requested(config) and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    
    
    if attention_mask is not None:
        
        left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
    else:
        left_padding_tokens = torch.zeros(batch_size, device=device, dtype=int)
    mask_factory_function = chunked_causal_mask_function(chunk_size, left_padding_tokens)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    
    
    
    use_vmap = False
    
    
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    
    
    
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    
    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  
        local_size=chunk_size,  
        dtype=dtype,  
        config=config,  
        use_vmap=use_vmap,  
        device=device,
    )
    return causal_mask


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
    "compressed_sparse_attention": create_sliding_window_causal_mask,
    "heavily_compressed_attention": create_sliding_window_causal_mask,
    "deepseek_sparse_attention": create_causal_mask,
}


def create_masks_for_generate(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    **kwargs,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in places like `generate`
    in order to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
    """
    
    effective_config = config.get_text_config()
    
    mask_kwargs = {
        "config": effective_config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
        "block_sequence_ids": block_sequence_ids,
    }

    
    
    if hasattr(effective_config, "layer_types"):
        layer_patterns = set(effective_config.layer_types)
        if len(layer_patterns) == 1:
            return LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[next(iter(layer_patterns))](**mask_kwargs)
        causal_masks = {}
        for layer_pattern in layer_patterns:
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    
    return create_causal_mask(**mask_kwargs)
