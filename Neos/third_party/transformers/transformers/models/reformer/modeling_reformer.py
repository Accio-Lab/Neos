# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch REFORMER model."""

import sys
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_reformer import ReformerConfig


logger = logging.get_logger(__name__)



LSHSelfAttentionOutput = namedtuple("LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hidden_states", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput", ["hidden_states", "attention_probs", "buckets"])
ReformerOutput = namedtuple("ReformerOutput", ["hidden_states", "attn_output", "attention_probs", "buckets"])
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput", ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"]
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


class ReformerDynamicCache:
    """
    A dynamic cache that stores past buckets instead of key/values.
    """

    def __init__(self, _distributed_cache_data: Iterable | None = None) -> None:
        self._seen_tokens = 0  
        self.buckets_cache: list[torch.Tensor] = []
        self.states_cache: list[torch.Tensor] = []

        if _distributed_cache_data is not None:
            for buckets, states in _distributed_cache_data:
                self.buckets_cache.append(buckets)
                self.states_cache.append(states)

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.states_cache)

    def update(
        self,
        buckets: torch.Tensor,
        states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `ReformerDynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        
        if layer_idx == 0:
            self._seen_tokens += states.shape[-2]

        
        if states is not None:
            if len(self.states_cache) <= layer_idx:
                self.states_cache.append(states)
            else:
                self.states_cache[layer_idx] = torch.cat([self.states_cache[layer_idx], states], dim=1)

        if buckets is not None:
            if len(self.buckets_cache) <= layer_idx:
                self.buckets_cache.append(buckets)
            else:
                self.buckets_cache[layer_idx] = torch.cat([self.buckets_cache[layer_idx], buckets], dim=-1)
        else:
            
            self.buckets_cache.append(torch.tensor([], device=self.states_cache[layer_idx].device))

        return self.buckets_cache[layer_idx], self.states_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return None

    def get_start_idx(self) -> int:
        if len(self) == 0:
            return 0
        return self.states_cache[0].shape[1]

    def reorder_cache(self, beam_idx):
        for layer_idx in range(len(self)):
            if self.buckets_cache[layer_idx] is not None and self.buckets_cache[layer_idx].numel() > 0:
                device = self.buckets_cache[layer_idx].device
                self.buckets_cache[layer_idx] = self.buckets_cache[layer_idx].index_select(0, beam_idx.to(device))

            if self.states_cache[layer_idx] is not None and self.states_cache[layer_idx].numel() > 0:
                device = self.states_cache[layer_idx].device
                self.states_cache[layer_idx] = self.states_cache[layer_idx].index_select(0, beam_idx.to(device))


def _stable_argsort(vector, dim):
    
    
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _get_min_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


class AxialPositionEmbeddings(nn.Module):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()

        if sum(self.axial_pos_embds_dim) != config.hidden_size:
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.hidden_size: {config.hidden_size}"
            )

        
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        ]

        if self.training is True:
            if reduce(mul, self.axial_pos_shape) != sequence_length:
                raise ValueError(
                    f"If training, make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply to "
                    f"sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. "
                    f"You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} "
                    "or changing config.axial_pos_shape."
                )

            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                
                transposed_weights = weights.transpose(2, 1)
                
                dropped_transposed_weights = nn.functional.dropout2d(
                    transposed_weights, p=self.dropout, training=self.training
                )
                dropped_weights = dropped_transposed_weights.transpose(2, 1)

                position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))

            else:
                position_encodings = torch.cat(
                    [torch.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights],
                    dim=-1,
                )

        else:
            if reduce(mul, self.axial_pos_shape) < sequence_length:
                raise ValueError(
                    f"Make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply at least to "
                    f"max(sequence_length, least_common_mult_chunk_length): max({sequence_length}, "
                    f"{self.least_common_mult_chunk_length})."
                )

            
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) // self.axial_pos_shape[1])

            
            position_encodings = torch.cat(
                [weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights], dim=-1
            )
            position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))

            
            position_encodings = torch.cat(
                [
                    torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
                    for i in range(batch_size)
                ],
                dim=0,
            )

        return position_encodings


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """
        Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError(f"Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}")


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx

        self.dropout = config.lsh_attention_probs_dropout_prob

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3), persistent=False)
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5), persistent=False)
        self.register_buffer("mask_value_float16", torch.tensor(-1e4), persistent=False)
        self.register_buffer("mask_value_float32", torch.tensor(-1e9), persistent=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_hashes=None,
        buckets=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        
        exists_cache = past_buckets_states is not None and len(past_buckets_states) > self.layer_idx
        if exists_cache:
            assert sequence_length == 1, (
                "At the moment, auto-regressive language generation is only possible one word at a time. Make sure"
                f" that input sequence length {sequence_length} equals 1, when `past_buckets_states` is passed."
            )

            
            query_vectors = self.query_key(hidden_states)
            query_vectors = self._split_hidden_size_dim(
                query_vectors, self.num_attention_heads, self.attention_head_size
            )

            past_buckets = past_buckets_states.buckets_cache[self.layer_idx]
            past_states = past_buckets_states.states_cache[self.layer_idx]

            if past_buckets.numel() != 0:
                key_value_hidden_states, sorted_bucket_idx, buckets = self._get_relevant_hid_states_and_buckets(
                    query_vectors=query_vectors,
                    attention_mask=attention_mask,
                    num_hashes=num_hashes,
                    hidden_states=hidden_states,
                    past_states=past_states,
                    past_buckets=past_buckets,
                )

                query_key_vectors = self._query_per_attn_head(key_value_hidden_states)
                value_vectors = self._value_per_attn_head(key_value_hidden_states)

                
                
                query_key_vectors = self._split_seq_length_dim_to(
                    query_key_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                value_vectors = self._split_seq_length_dim_to(
                    value_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                
                query_vectors = query_vectors.unsqueeze(2).repeat(1, 1, num_hashes, 1, 1)
            else:
                key_value_hidden_states = torch.cat([past_states, hidden_states], dim=1)

                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)

        else:
            
            query_vectors = None
            query_key_vectors = self.query_key(hidden_states)
            value_vectors = self.value(hidden_states)

        
        if not exists_cache or past_buckets.numel() == 0:
            query_key_vectors = self._split_hidden_size_dim(
                query_key_vectors, self.num_attention_heads, self.attention_head_size
            )
            value_vectors = self._split_hidden_size_dim(
                value_vectors, self.num_attention_heads, self.attention_head_size
            )

        
        if exists_cache and key_value_hidden_states.shape[1] >= self.chunk_length:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        
        del hidden_states

        assert query_key_vectors.shape[-1] == self.attention_head_size, (
            f"last dim of query_key_vectors is {query_key_vectors.shape[-1]} but should be {self.attention_head_size}."
        )
        assert value_vectors.shape[-1] == self.attention_head_size, (
            f"last dim of value_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."
        )

        do_standard_self_attention = (sequence_length <= self.chunk_length) or (
            exists_cache and past_states is not None
        )
        
        if not do_standard_self_attention:
            
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            
            if buckets is None:
                
                buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
            else:
                
                buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes * sequence_length)

            assert int(buckets.shape[-1]) == num_hashes * sequence_length, (
                f"last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}"
            )

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            if self.chunk_length is None:
                assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                    "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                    " `config.num_chunks_before` are set to 0."
                )
        elif exists_cache and past_buckets.numel() != 0:
            
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            
            sorted_bucket_idx_per_hash = torch.arange(sequence_length, device=query_key_vectors.device).repeat(
                batch_size, self.num_attention_heads, 1
            )

        
        sqrt_num = np.sqrt(self.attention_head_size)
        key_vectors = self._len_and_dim_norm(query_key_vectors, sqrt_num)

        
        query_vectors = query_vectors if query_vectors is not None else query_key_vectors

        
        del query_key_vectors

        
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            do_standard_self_attention=do_standard_self_attention,
            use_cache=exists_cache,
        )

        
        del key_vectors, value_vectors

        
        if not do_standard_self_attention:
            
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention or (exists_cache and past_buckets.numel() != 0):
            
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                
                del probs_vectors

            
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), (
            "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length,"
            " config.attention_head_size]`."
        )

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        if buckets is not None:
            buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes, -1)

        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _query_per_attn_head(self, hidden_states):
        per_head_query_key = self.query_key.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        
        query_key_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        per_head_value = self.value.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        
        value_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_value)
        return value_vectors

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]

        
        
        
        if isinstance(self.num_buckets, int):
            assert self.num_buckets % 2 == 0, (
                f"There should be an even number of buckets, but `self.num_buckets`: {self.num_buckets}"
            )
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, (
                    f"The number of buckets should be even, but `num_bucket`: {bucket_factor}"
                )
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        
        vectors = vectors.detach()

        if self.hash_seed is not None:
            
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
            
            num_buckets = num_buckets + 1
            
            buckets_mask = attention_mask.to(torch.bool)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(
                buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device)
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        
        
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        
        with torch.no_grad():
            
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        
        num_buckets = 2**num_buckets_pow_2

        
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        logger.warning(f"config.num_buckets is not set. Setting config.num_buckets to {num_buckets}...")

        
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        do_standard_self_attention,
        use_cache,
    ):
        
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        
        
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        
        del query_vectors, key_vectors

        
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads
            )
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif use_cache and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = (
                key_value_bucket_idx.new_ones(key_value_bucket_idx.shape[:-1] + (1,)) * key_value_bucket_idx.max()
            )
        elif use_cache and query_key_dots.ndim <= 4:
            query_bucket_idx = (query_key_dots.shape[-1] - 1) * torch.ones_like(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx = torch.arange(
                query_key_dots.shape[-1], dtype=torch.long, device=query_key_dots.device
            )[None, None, :].expand(query_bucket_idx.shape[:2] + (-1,))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        if not use_cache:
            mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention,
            )

            if mask is not None:
                query_key_dots = torch.where(mask, query_key_dots, mask_value)

            
            del mask

        
        
        
        
        
        
        
        
        

        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        
        attention_probs = torch.exp(query_key_dots - logits)

        
        del query_key_dots

        
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        
        out_vectors = torch.matmul(attention_probs, value_vectors)

        
        del value_vectors

        
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):
        
        if attention_mask is not None:
            
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard_self_attention:
                
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets
    ):
        
        hidden_states = torch.cat([past_states, hidden_states], dim=1)

        
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        
        max_bucket = self.num_buckets if isinstance(self.num_buckets, int) else reduce(mul, self.num_buckets)

        
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1

        
        query_buckets = self._hash_vectors(
            query_vectors, num_hashes, attention_mask, increase_num_buckets=increase_num_buckets
        )

        
        concat_buckets = torch.cat([past_buckets, query_buckets.unsqueeze(-1)], dim=-1)

        
        bucket_idx = _stable_argsort(concat_buckets, dim=-1)

        
        assert bucket_idx.shape == (
            batch_size,
            self.num_attention_heads,
            num_hashes,
            sequence_length,
        ), (
            f"bucket_idx should have shape {(batch_size, self.num_attention_heads, num_hashes, sequence_length)}, but"
            f" has shape {bucket_idx.shape}."
        )

        
        relevant_bucket_idx = (bucket_idx == (bucket_idx.shape[-1] - 1)).nonzero()

        
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(relevant_bucket_idx, sequence_length)
        relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.transpose(0, 1))]

        
        offset = torch.arange(relevant_bucket_idx_chunk.shape[-1], device=hidden_states.device, dtype=torch.long)
        bucket_idx_batch_offset = sequence_length * (
            batch_size * torch.div(offset, relevant_bucket_idx_chunk.shape[-1], rounding_mode="floor")
        )

        
        relevant_bucket_idx_chunk_all_batch = relevant_bucket_idx_chunk + bucket_idx_batch_offset
        hidden_states = hidden_states.reshape((-1, self.hidden_size))

        
        relevant_hidden_states = hidden_states.index_select(0, relevant_bucket_idx_chunk_all_batch)

        
        relevant_hidden_states = relevant_hidden_states.reshape(
            batch_size, self.num_attention_heads, -1, self.hidden_size
        )
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(
            batch_size, self.num_attention_heads, num_hashes, -1
        )

        assert (
            relevant_hidden_states.shape[2]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`,"
            f" there are {relevant_hidden_states.shape[2]} `hidden_states`."
        )

        assert (
            relevant_bucket_idx_chunk.shape[-1]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are"
            f" {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`."
        )

        return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        
        start_indices_chunk = ((indices[:, -1] // self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = expanded_start_indices + torch.arange(
            total_chunk_size, device=indices.device, dtype=torch.long
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors, sqrt_num):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors / sqrt_num
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        variance = torch.mean(x**2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class ReverseSort(Function):
    """
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        
        return grad_out_vectors, grad_logits, None, None


class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.pad_token_id = config.pad_token_id

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.local_attention_probs_dropout_prob

        
        self.register_buffer("mask_value_float16", torch.tensor(-1e4), persistent=False)
        self.register_buffer("mask_value_float32", torch.tensor(-1e9), persistent=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        
        if past_buckets_states is not None and len(past_buckets_states) > self.layer_idx:
            past_buckets = past_buckets_states.buckets_cache[self.layer_idx]
            past_states = past_buckets_states.states_cache[self.layer_idx]

            assert past_buckets.numel() == 0, (
                "LocalSelfAttention should not make use of `buckets`. There seems to be an error when caching"
                " hidden_states_and_buckets."
            )
            key_value_hidden_states = self._retrieve_relevant_hidden_states(
                past_states, self.chunk_length, self.num_chunks_before
            )
            key_value_hidden_states = torch.cat([key_value_hidden_states, hidden_states], dim=1)

            
            query_vectors = self.query(hidden_states)
            
            key_vectors = self.key(key_value_hidden_states)
            value_vectors = self.value(key_value_hidden_states)

            
            del key_value_hidden_states
        else:
            
            query_vectors = self.query(hidden_states)
            key_vectors = self.key(hidden_states)
            value_vectors = self.value(hidden_states)

        
        query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert query_vectors.shape[-1] == self.attention_head_size, (
            f"last dim of query_key_vectors is {query_vectors.shape[-1]} but should be {self.attention_head_size}."
        )
        assert key_vectors.shape[-1] == self.attention_head_size, (
            f"last dim of query_key_vectors is {key_vectors.shape[-1]} but should be {self.attention_head_size}."
        )
        assert value_vectors.shape[-1] == self.attention_head_size, (
            f"last dim of query_key_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."
        )

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                " `config.num_chunks_before` are set to 0."
            )

        
        key_vectors = key_vectors / np.sqrt(self.attention_head_size)

        
        indices = torch.arange(sequence_length, device=query_vectors.device).repeat(
            batch_size, self.num_attention_heads, 1
        )

        
        do_standard_self_attention = sequence_length <= self.chunk_length

        
        if not do_standard_self_attention:
            
            
            query_vectors = self._split_seq_length_dim_to(
                query_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            key_vectors = self._split_seq_length_dim_to(
                key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            
            query_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)
            key_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)

            
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices = key_indices = indices

        
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(
            query_indices, key_indices, attention_mask, query_key_dots.shape, do_standard_self_attention
        )

        if mask is not None:
            
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16.half()
            else:
                mask_value = self.mask_value_float32

            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        
        del mask

        
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)

        
        del logits

        
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        
        out_vectors = torch.matmul(attention_probs, value_vectors)

        
        del value_vectors

        
        if not do_standard_self_attention:
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard_self_attention
    ):
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]

            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
                attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
            
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)

        
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length, num_chunks_before):
        start_position = ((previous_hidden_states.shape[1] // chunk_length) - num_chunks_before) * chunk_length
        return previous_hidden_states[:, start_position:]


class ReformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dropout = config.hidden_dropout_prob

        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config, layer_idx=layer_id)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config, layer_idx=layer_id)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == {"lsh", "local"}:
            
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(config, layer_idx=layer_id)
            else:
                self.self_attention = LocalSelfAttention(config, layer_idx=layer_id)
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        self.output = ReformerSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        buckets=None,
        **kwargs,
    ):
        hidden_states = self.layer_norm(hidden_states)

        
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            buckets=buckets,
        )

        
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        
        if use_cache and past_buckets_states is not None:
            
            states = (
                hidden_states[:, :orig_sequence_length]
                if len(past_buckets_states.states_cache) <= self.layer_id
                else hidden_states
            )
            buckets = (
                buckets[:, :, :, :orig_sequence_length]
                if (
                    len(past_buckets_states.buckets_cache) <= self.layer_id
                    and buckets is not None
                    and orig_sequence_length > 1
                )
                else buckets
            )
            buckets, hidden_states = past_buckets_states.update(
                buckets, states[:, :orig_sequence_length], self.layer_id
            )

        
        attention_output = self.output(self_attention_outputs.hidden_states)

        return AttentionOutput(
            hidden_states=attention_output,
            attention_probs=self_attention_outputs.attention_probs,
            buckets=buckets,
        )


class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)
        
        
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        
        
        if len(torch.cuda.default_generators) > 0:
            
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            
            self.attention_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        
        
        if len(torch.cuda.default_generators) > 0:
            
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.feed_forward_seed)

    def forward(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
    ):
        with torch.no_grad():
            
            
            
            if self.training:
                self._init_attention_seed()

            attn_outputs = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs.hidden_states

            
            
            attn_output = prev_attn_output + attn_output

            
            del prev_attn_output

            
            
            
            if self.training:
                self._init_feed_forward_seed()
            
            hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    def backward_pass(
        self,
        next_attn_output,
        hidden_states,
        grad_attn_output,
        grad_hidden_states,
        attention_mask=None,
        buckets=None,
    ):
        
        
        
        

        assert self.training, (
            "If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the"
            " model into training mode."
        )

        with torch.enable_grad():
            next_attn_output.requires_grad = True

            
            torch.manual_seed(self.feed_forward_seed)
            
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)

        with torch.no_grad():
            
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states

            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None

        with torch.enable_grad():
            hidden_states.requires_grad = True

            
            torch.manual_seed(self.attention_seed)
            
            
            output = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                buckets=buckets,
            ).hidden_states
            output.backward(grad_attn_output, retain_graph=True)

        with torch.no_grad():
            
            attn_output = next_attn_output - output
            del output, next_attn_output

            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()

        return ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )


class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )

            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.attention_mask = attention_mask

        
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)

        
        attn_output, hidden_states = ctx.saved_tensors

        
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )

        
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                attention_mask=attention_mask,
                buckets=buckets,
            )

        assert all_buckets == (), "buckets have to be empty after backpropagation"
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)

        
        
        return grad_hidden_states, None, None, None, None, None, None, None, None, None, None, None


class ReformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob

        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        
        
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        
        all_hidden_states = []
        all_attentions = []

        
        if use_cache and past_buckets_states is None:
            past_buckets_states = ReformerDynamicCache()

        
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        
        hidden_states = self.layer_norm(hidden_states)

        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        next_cache = past_buckets_states if use_cache else None

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=next_cache,
        )


class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)

    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


@auto_docstring
class ReformerPreTrainedModel(PreTrainedModel):
    config: ReformerConfig
    base_model_prefix = "reformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, AxialPositionEmbeddings):
            for weight in module.weights:
                init.normal_(weight, std=self.config.axial_norm_std)
        elif isinstance(module, LSHSelfAttention):
            init.constant_(module.self_mask_value_float16, -1e3)
            init.constant_(module.self_mask_value_float32, -1e5)
            init.constant_(module.mask_value_float16, -1e4)
            init.constant_(module.mask_value_float32, -1e9)
        elif isinstance(module, LocalSelfAttention):
            init.constant_(module.mask_value_float16, -1e4)
            init.constant_(module.mask_value_float32, -1e9)


@auto_docstring(
    custom_intro="""
    Output type of [`ReformerModel`].
    """
)
@dataclass
class ReformerModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
        Sequence of hidden-states at the last layer of the model.

        `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
        corresponds to `sequence_length`.
    past_buckets_states (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
        being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
        second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

        Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
        up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor
    past_buckets_states: list[tuple[torch.LongTensor, torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Output type of [`ReformerModelWithLMHead`].
    """
)
@dataclass
class ReformerModelWithLMHeadOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
        corresponds to `sequence_length`.
    past_buckets_states (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
        being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
        second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

        Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
        up sequential decoding.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_buckets_states: list[tuple[torch.LongTensor, torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        assert self.config.num_hidden_layers > 0, (
            "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"
        )

        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)

        
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        num_hashes: int | None = None,
        past_buckets_states: ReformerDynamicCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | ReformerModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
            a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
            are automatically padded to be a multiple of the chunk length.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        num_hashes (`int`, *optional*):
            The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
            the default defined in `config.num_hashes`.

            For more information, see `num_hashes` in [`ReformerConfig`].
        past_buckets_states (`ReformerDynamicCache`, *optional*):
            List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
            up sequential decoding.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert len(input_shape) == 2, (
            f"`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {input_shape}"
        )

        if past_buckets_states is not None:
            assert not self.training, "`past_buckets_states` can only be used for inference, not for training`."

        
        orig_sequence_length = input_shape[-1]

        
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        min_chunk_length = _get_min_chunk_len(self.config)

        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0
            and input_shape[-1] > min_chunk_length
            and past_buckets_states is None
        )

        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length

            if self.training is True:
                raise ValueError(
                    f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                    f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                    f"of {input_shape[-1] + padding_length}."
                )

            
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )

        
        start_idx_pos_encodings = past_buckets_states.get_start_idx() if past_buckets_states is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states

        
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        past_buckets_states = encoder_outputs.past_buckets_states if use_cache else None
        hidden_states = encoder_outputs.all_hidden_states if output_hidden_states else None
        attentions = encoder_outputs.all_attentions if output_attentions else None

        if not return_dict:
            return tuple(v for v in [sequence_output, past_buckets_states, hidden_states, attentions] if v is not None)
        return ReformerModelOutput(
            last_hidden_state=sequence_output,
            past_buckets_states=past_buckets_states,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def _pad_to_mult_of_chunk_length(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        input_shape=None,
        padding_length=None,
        padded_seq_length=None,
        device=None,
    ):
        logger.warning_once(
            f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
            f"multiple of `config.chunk_length`: {padded_seq_length}"
        )

        padded_input_ids = torch.full(
            (input_shape[0], padding_length),
            self.config.pad_token_id,
            device=device,
            dtype=torch.long,
        )

        
        if attention_mask is not None:
            pad_attention_mask = torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype)

            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.bool),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.bool),
                ],
                dim=-1,
            )

        
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()

            
            if position_ids is not None:
                padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)

        
        if inputs_embeds is not None:
            padded_inputs_embeds = self.get_input_embeddings()(padded_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape


@auto_docstring(
    custom_intro="""
    Reformer Model with a `language modeling` head on top.
    """
)
class ReformerModelWithLMHead(ReformerPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        assert config.is_decoder, "If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`."
        assert "local" not in self.config.attn_layers or config.local_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.local_num_chunks_after` is set to 0 and not"
            f" {config.local_num_chunks_after}."
        )
        assert "lsh" not in self.config.attn_layers or config.lsh_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.lsh_num_chunks_after` is set to 1 and not"
            f" {config.lsh_num_chunks_after}."
        )

        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
        self.lm_head.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        num_hashes: int | None = None,
        past_buckets_states: list[tuple[torch.Tensor]] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
            a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
            are automatically padded to be a multiple of the chunk length.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        num_hashes (`int`, *optional*):
            The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
            the default defined in `config.num_hashes`.

            For more information, see `num_hashes` in [`ReformerConfig`].
        past_buckets_states (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*):
            List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
            up sequential decoding.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        hidden_states = reformer_outputs[0]
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ReformerModelWithLMHeadOutput(
            loss=loss,
            logits=logits,
            past_buckets_states=reformer_outputs.past_buckets_states,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        
        return None

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, use_cache=None, num_hashes=None, is_first_iteration=False, **kwargs
    ):
        

        
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_buckets_states": past_key_values,
            "use_cache": use_cache,
            "num_hashes": num_hashes,
        }

        
        kwargs.pop("attention_mask", None)
        
        for key, value in kwargs.items():
            if key not in model_inputs:
                print(f"Warning: {key} is not a recognized input.")
                model_inputs[key] = value

        return model_inputs


@auto_docstring
class ReformerForMaskedLM(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        assert not config.is_decoder, (
            "If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional"
            " self-attention."
        )
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
        self.lm_head.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        num_hashes: int | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | MaskedLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
            a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
            are automatically padded to be a multiple of the chunk length.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        num_hashes (`int`, *optional*):
            The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
            the default defined in `config.num_hashes`.

            For more information, see `num_hashes` in [`ReformerConfig`].
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels

            <Tip warning={true}>

            This example uses a false checkpoint since we don't have any available pretrained model for the masked language
            modeling task with the Reformer architecture.

            </Tip>

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ReformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
        >>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

        >>> # add mask_token
        >>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})  # doctest: +IGNORE_RESULT
        >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

        >>> # resize model's embedding matrix
        >>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)  # doctest: +IGNORE_RESULT

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> predicted_token = tokenizer.decode(predicted_token_id)
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        >>> # mask labels of non-[MASK] tokens
        >>> labels = torch.where(
        ...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
        ... )

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = round(outputs.loss.item(), 2)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            use_cache=False,  
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """
)
class ReformerForSequenceClassification(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.reformer = ReformerModel(config)
        self.classifier = ReformerClassificationHead(config)
        if config.is_decoder is True:
            logger.warning("You might want to disable causal masking for sequence classification")

        
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        num_hashes: int | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | SequenceClassifierOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
            a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
            are automatically padded to be a multiple of the chunk length.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        num_hashes (`int`, *optional*):
            The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
            the default defined in `config.num_hashes`.

            For more information, see `num_hashes` in [`ReformerConfig`].
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Example of single-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ReformerForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
        >>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> label = model.config.id2label[predicted_class_id]
        ```

        ```python
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = ReformerForSequenceClassification.from_pretrained(
        ...     "google/reformer-crime-and-punishment", num_labels=num_labels
        ... )

        >>> labels = torch.tensor(1)
        >>> loss = model(**inputs, labels=labels).loss
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ReformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@auto_docstring
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.reformer = ReformerModel(config)
        
        self.qa_outputs = nn.Linear(2 * config.hidden_size, config.num_labels)

        
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        num_hashes: int | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | QuestionAnsweringModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
            a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
            are automatically padded to be a multiple of the chunk length.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        num_hashes (`int`, *optional*):
            The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
            the default defined in `config.num_hashes`.

            For more information, see `num_hashes` in [`ReformerConfig`].
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            use_cache=False,  
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + reformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )


__all__ = [
    "ReformerAttention",
    "ReformerForMaskedLM",
    "ReformerForQuestionAnswering",
    "ReformerForSequenceClassification",
    "ReformerLayer",
    "ReformerModel",
    "ReformerModelWithLMHead",
    "ReformerPreTrainedModel",
]
