# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
# Copyright 2025 The vLLM project
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

# This implementation is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/v1/serial_utils.py


import pickle
import struct
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, TypeAlias

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack
from tensordict import TensorDictBase

from transfer_queue.utils.logging_utils import get_logger

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_TENSOR = 3  
CUSTOM_TYPE_NESTED_TENSOR = 4  
CUSTOM_TYPE_NUMPY = 5  


_PICKLE_FALLBACK_SENTINEL = b"\xc1\xfe\xed"

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame

logger = get_logger(__name__)



warnings.filterwarnings(action="ignore", message=r"The given buffer is not writable*", category=UserWarning)



_encoder_aux_buffers: ContextVar[list[bytestr] | None] = ContextVar("encoder_aux_buffers", default=None)
_decoder_aux_buffers: ContextVar[Sequence[bytestr] | None] = ContextVar("decoder_aux_buffers", default=None)


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global encoder instance to be safely used across multiple
    threads and async coroutines.

    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)

    @property
    def aux_buffers(self) -> list[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _encoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of encode() context"
        return buffers

    def encode(self, obj: Any) -> Sequence[bytestr]:
        """Encode a given object to a byte array."""

        bufs: list[bytestr] = [b""]
        token = _encoder_aux_buffers.set(bufs)
        try:
            bufs[0] = self.encoder.encode(obj)
            
            
            
            
            return bufs
        finally:
            _encoder_aux_buffers.reset(token)

    def enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling

        """
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        
        if isinstance(obj, TensorDictBase):
            return self._encode_tensordict(obj)

        
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind != "O" and not obj.dtype.hasobject:
                try:
                    return self._encode_numpy(obj)
                except (TypeError, RuntimeError, ValueError):
                    
                    pass
            
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        if callable(obj):
            
            
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        
        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_tensordict(self, obj: Any) -> dict:
        """Convert TensorDict to a dict structure for recursive msgpack processing.

        This allows msgpack to recursively call enc_hook for each tensor inside,
        enabling zero-copy serialization of nested tensors.
        """
        
        
        data_dict = dict(obj.items())

        
        return {
            "__tq_tensordict__": True,
            "batch_size": list(obj.batch_size),  
            "data": data_dict,
        }

    def _encode_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode tensor with zero-copy buffer extraction (handles GPU, non-contiguous, nested)."""
        assert len(self.aux_buffers) > 0

        
        if obj.is_nested:
            return self._encode_nested_tensor(obj)

        return self._encode_regular_tensor(obj)

    def _encode_nested_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode nested tensor by unbinding into sub-tensors for zero-copy."""
        
        sub_tensors = obj.unbind()

        
        encoded_sub_tensors = []
        for t in sub_tensors:
            
            meta = self._encode_regular_tensor_meta(t)
            encoded_sub_tensors.append(meta)

        
        layout = "jagged" if obj.layout == torch.jagged else "strided"
        nested_meta = {
            "layout": layout,
            "tensors": encoded_sub_tensors,
        }
        return msgpack.Ext(CUSTOM_TYPE_NESTED_TENSOR, pickle.dumps(nested_meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_regular_tensor_meta(self, obj: torch.Tensor) -> tuple:
        """Encode a regular tensor and return its metadata tuple."""
        

        if not obj.is_contiguous():
            obj = obj.contiguous()

        
        if obj.device.type != "cpu":
            obj = obj.cpu()

        
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        dtype = str(obj.dtype).removeprefix("torch.")
        return (dtype, tuple(obj.shape), idx)

    def _encode_regular_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode a regular (non-nested) tensor with zero-copy."""
        

        if not obj.is_contiguous():
            obj = obj.contiguous()

        
        if obj.device.type != "cpu":
            obj = obj.cpu()

        if obj.is_sparse:
            
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        
        dtype = str(obj.dtype).removeprefix("torch.")
        meta = (dtype, tuple(obj.shape), idx)
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_numpy(self, obj: np.ndarray) -> msgpack.Ext:
        """Encode numpy array with zero-copy buffer extraction."""
        
        if not obj.flags["C_CONTIGUOUS"]:
            obj = np.ascontiguousarray(obj)

        
        buf = memoryview(obj.view(np.uint8).ravel())
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        meta = (str(obj.dtype), tuple(obj.shape), idx)
        return msgpack.Ext(CUSTOM_TYPE_NUMPY, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global decoder instance to be safely used across multiple
    threads and async coroutines.
    """

    def __init__(self):
        self.decoder = msgpack.Decoder(ext_hook=self.ext_hook)

    @property
    def aux_buffers(self) -> Sequence[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _decoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of decode() context"
        return buffers

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        """Decode a list of bytes."""
        if isinstance(bufs, bytestr):
            result = self.decoder.decode(bufs)
        else:
            token = _decoder_aux_buffers.set(bufs)
            try:
                result = self.decoder.decode(bufs[0])  # type: ignore[index]
            finally:
                _decoder_aux_buffers.reset(token)

        
        return self._reconstruct_special_types(result)

    def _reconstruct_special_types(self, obj: Any) -> Any:
        """Recursively reconstruct special types (TensorDict) from their dict representation."""
        if isinstance(obj, dict):
            
            if obj.get("__tq_tensordict__"):
                return self._reconstruct_tensordict(obj)
            
            return {k: self._reconstruct_special_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._reconstruct_special_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._reconstruct_special_types(item) for item in obj)
        return obj

    def _reconstruct_tensordict(self, obj: dict) -> Any:
        """Reconstruct TensorDict from marked dict structure."""
        try:
            from tensordict import TensorDict

            batch_size = obj["batch_size"]
            data = obj["data"]
            
            processed_data = self._reconstruct_special_types(data)
            return TensorDict(processed_data, batch_size=batch_size)
        except ImportError:
            
            return obj

    def _decode_tensor(self, meta: tuple) -> torch.Tensor:
        """Decode tensor from (dtype, shape, buffer_idx) tuple."""
        dtype, shape, idx = meta
        buffer = self.aux_buffers[idx]
        torch_dtype = getattr(torch, dtype)

        if not buffer:  
            return torch.empty(shape, dtype=torch_dtype)

        
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        
        return arr.view(torch_dtype).view(shape)

    def _decode_nested_tensor(self, nested_meta: dict) -> torch.Tensor:
        """Decode nested tensor from serialized sub-tensors."""
        layout = nested_meta["layout"]
        tensor_metas = nested_meta["tensors"]

        
        sub_tensors = [self._decode_tensor(meta) for meta in tensor_metas]

        
        if layout == "jagged":
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.jagged)
        else:  
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.strided)

    def _decode_numpy(self, meta: tuple) -> np.ndarray:
        """Decode numpy array from (dtype_str, shape, buffer_idx) tuple."""
        dtype_str, shape, idx = meta
        buffer = self.aux_buffers[idx]
        np_dtype = np.dtype(dtype_str)

        if not buffer:  
            return np.empty(shape, dtype=np_dtype)

        
        arr = np.frombuffer(buffer, dtype=np.uint8)
        return arr.view(np_dtype).reshape(shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        """Custom decoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling
        """
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)
        if code == CUSTOM_TYPE_TENSOR:
            meta = pickle.loads(data)
            return self._decode_tensor(meta)
        if code == CUSTOM_TYPE_NESTED_TENSOR:
            nested_meta = pickle.loads(data)
            return self._decode_nested_tensor(nested_meta)
        if code == CUSTOM_TYPE_NUMPY:
            meta = pickle.loads(data)
            return self._decode_numpy(meta)

        raise NotImplementedError(f"Extension type code {code} is not supported")


_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder()


def encode(obj: Any) -> list[bytestr]:
    """Encode an object via msgpack zero-copy; falls back to pickle on failure.

    The pickle path is a normal degradation path (e.g. body contains torch.dtype
    objects). Use this as the single entry point for all ZMQ message serialization.
    """
    try:
        return list(_encoder.encode(obj))
    except (TypeError, ValueError) as e:
        logger.debug(
            "encode: msgpack failed (%s), falling back to pickle.",
            type(e).__name__,
        )
        return [_PICKLE_FALLBACK_SENTINEL, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)]


def decode(frames: list) -> Any:
    """Decode frames produced by encode.

    Transparently handles both the msgpack zero-copy path and the pickle
    fallback path based on the leading sentinel frame.
    """
    if len(frames) >= 2 and frames[0] == _PICKLE_FALLBACK_SENTINEL:
        return pickle.loads(frames[1])
    return _decoder.decode(frames)






_PACK_HEADER_FMT = "<I"
_PACK_HEADER_SIZE = struct.calcsize(_PACK_HEADER_FMT)
_PACK_ENTRY_FMT = "<II"
_PACK_ENTRY_SIZE = struct.calcsize(_PACK_ENTRY_FMT)


def calc_packed_size(items: Sequence[bytestr]) -> int:
    """Total bytes required to pack ``items`` into one buffer."""
    return _PACK_HEADER_SIZE + len(items) * _PACK_ENTRY_SIZE + sum(memoryview(item).nbytes for item in items)


def pack_into(target_buffer: bytestr, items: Sequence[bytestr]) -> None:
    """Concatenate ``items`` into ``target_buffer``, which must be at least ``calc_packed_size(items)`` bytes."""
    target_mv = memoryview(target_buffer)
    required = calc_packed_size(items)
    if target_mv.nbytes < required:
        raise ValueError(f"pack_into: target buffer has {target_mv.nbytes} bytes, requires {required}")
    struct.pack_into(_PACK_HEADER_FMT, target_mv, 0, len(items))

    entry_offset = _PACK_HEADER_SIZE
    payload_offset = _PACK_HEADER_SIZE + len(items) * _PACK_ENTRY_SIZE

    target_tensor = torch.frombuffer(target_mv, dtype=torch.uint8)

    for item in items:
        item_mv = memoryview(item)
        nbytes = item_mv.nbytes
        struct.pack_into(_PACK_ENTRY_FMT, target_mv, entry_offset, payload_offset, nbytes)
        src_tensor = torch.frombuffer(item_mv, dtype=torch.uint8)
        target_tensor[payload_offset : payload_offset + nbytes].copy_(src_tensor)
        entry_offset += _PACK_ENTRY_SIZE
        payload_offset += nbytes


def unpack_from(source_buffer: bytestr) -> list[memoryview]:
    """Split a packed buffer back into N memoryview slices over ``source_buffer``."""
    mv = memoryview(source_buffer)
    item_count = struct.unpack_from(_PACK_HEADER_FMT, mv, 0)[0]
    result: list[memoryview] = []
    for i in range(item_count):
        offset, length = struct.unpack_from(_PACK_ENTRY_FMT, mv, _PACK_HEADER_SIZE + i * _PACK_ENTRY_SIZE)
        result.append(mv[offset : offset + length])
    return result


def batch_encode_into(
    objs: list[Any],
    alloc_buff_func: Callable[[list[int]], list[Any]],
    *,
    num_workers: int = 1,
) -> tuple[list[np.ndarray | memoryview], list[int]]:
    """Encode multiple objects in-place into caller-allocated buffers.

    Each object is msgpack-encoded (with zero-copy tensor/ndarray extraction)
    and packed into a buffer slot supplied by ``alloc_buff_func``. Buffers are
    written in place; the function returns the same buffer list along with
    each slot's packed byte length.

    Args:
        objs: Objects to encode, one per output buffer slot.
        alloc_buff_func: Callable taking per-object packed sizes and returning
            the corresponding buffer list. ``buffers[i]`` must be an
            ``np.ndarray`` or ``memoryview`` holding at least ``sizes[i]``
            bytes.
        num_workers: Thread count for parallel packing. Default 1 (serial).

    Returns:
        tuple[list[np.ndarray | memoryview], list[int]]: The buffers returned by
            ``alloc_buff_func`` with packed bytes written, paired with each
            object's packed length (``<=`` buffer capacity).

    Note:
        Lifetime is caller-owned: this function holds no references to the
        buffers after return. Whatever backs the allocation must outlive all
        downstream consumers.

    Example:
        >>> # Pack two tensors into pre-allocated pinned uint8 tensor buffers
        >>> def alloc(sizes):
        ...     return [torch.empty(s, dtype=torch.uint8, pin_memory=True) for s in sizes]
        >>> objs = [torch.tensor([1, 2, 3]), torch.tensor([4.0, 5.0])]
        >>> bufs, lengths = batch_encode_into(objs, alloc)
        >>> print(f"packed sizes: {lengths}")
    """
    batch_items = [encode(obj) for obj in objs]
    batch_sizes = [calc_packed_size(items) for items in batch_items]
    buffers = alloc_buff_func(batch_sizes)

    def _pack_one(pair: tuple[Any, list[bytestr]]) -> None:
        buf, items = pair
        mv = buf.numpy().data if hasattr(buf, "numpy") else memoryview(buf)
        pack_into(mv, items)

    if num_workers <= 1:
        for pair in zip(buffers, batch_items, strict=True):
            _pack_one(pair)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_pack_one, zip(buffers, batch_items, strict=True)))

    return buffers, batch_sizes


def batch_decode_from(source_buffers: Sequence[np.ndarray | memoryview]) -> list[Any]:
    """Reverse of ``batch_encode_into``: unpack and decode each filled buffer.

    Args:
        source_buffers: Per-object receive buffers in order. Each must be an
            ``np.ndarray`` or ``memoryview``.

    Returns:
        list[Any]: Decoded objects, one per input buffer, in the same order.

    Note:
        Tensors and ndarrays in the result are zero-copy views over the
        source buffers. The Python reference chain (``torch.frombuffer`` ->
        ``Py_buffer`` -> memoryview slice -> parent memoryview -> numpy array
        -> original buffer) keeps the source alive as long as the decoded
        object is reachable; the caller does NOT need to retain the source
        buffer separately.

    Example:
        >>> # Round-trip: encode then decode
        >>> def alloc(sizes):
        ...     return [torch.empty(s, dtype=torch.uint8) for s in sizes]
        >>> objs = [torch.tensor([1, 2, 3]), torch.tensor([4.0, 5.0])]
        >>> bufs, _ = batch_encode_into(objs, alloc)
        >>> decoded = batch_decode_from(bufs)
    """
    return [
        decode(unpack_from(buf.numpy().data if hasattr(buf, "numpy") else memoryview(buf))) for buf in source_buffers
    ]
