# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        
        attention_scores += index_mask.view(b, 1, sq, sk)
        
        index_scores += index_mask

    
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    
    
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    
    
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    
    
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    
    
    kl_div = kl_per_element.sum(dim=-1).mean()

    
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    
    
    
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    
    index_scores = torch.relu(index_scores)

    
    
    
    index_scores = index_scores * weights.unsqueeze(-1)

    
    
    index_scores = index_scores.sum(dim=2)

    
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    seqlen = q.size(0)
    
    
    
    
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    
    
    
    topk_k = min(index_topk, seqlen)
    
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss, pg_collection
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    """Naive implementation of backward pass for indexer loss."""
    index_scores = _compute_index_scores(q, weights, k)  

    sq, b, np, hn = query.size()
    sk = key.size(0)

    
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    
    del query_reshaped, key_reshaped

    
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    
    
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    
    index_scores = index_scores + causal_mask.unsqueeze(0)
    
    del causal_mask

    if sparse_loss:
        
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        
        index_scores = index_scores + index_mask

    
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    
    del index_scores

    
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    
    del attention_scores_sum

    
    
    grad_kl_div = grad_loss * loss_coeff  

    
    grad_kl_per_row = grad_kl_div / (b * sq)  

    
    
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    
    
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    
    del attention_scores_normalized

    
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    
    
    
    causal_valid_mask = torch.tril(
        torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    )  
    if sparse_loss:
        
        index_valid_mask = index_mask == 0  
        del index_mask  
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  
        del index_valid_mask
    else:
        del index_mask  
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  
    del grad_index_scores_logits

    
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  
    del grad_index_scores

    
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  
    
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    
    
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  

    
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  
    del grad_weighted_scores, scores_after_relu

    
    grad_scores = grad_scores_after_relu * relu_mask.float()  
    del grad_scores_after_relu, relu_mask

    
    
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  
    
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        )

        
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.pg_collection = pg_collection

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.pg_collection,
        )

        
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        
        
        x_nope, x_pe = torch.split(
            x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        
        x = torch.cat([x_nope, x_pe], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        
        
        
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        
        
        
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        
        
        
        seqlen, bsz, _ = x.size()

        
        
        
        
        q, _ = self.linear_wq_b(qr)
        
        
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        
        
        
        
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)
        
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        
        
        
        q = rotate_activation(q)
        k = rotate_activation(k)

        
        
        
        
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

        
        
        
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Unfused sparse attention implementation.
    """
    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    
    
    
    
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    
    
    
    
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    
    index_mask += causal_mask.view(1, sq, skv)
    
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    
    
    
    
    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    
    output = torch.bmm(attention_scores.to(value.dtype), value)
    
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    
    output = output.reshape(sq, b, np * hnv)
    return output


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        sq, b, np, hn = query.size()
        skv = key.size(0)
        hnv = value.size(3)

        
        x = x.detach()
        qr = qr.detach()

        
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            
            
            
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            
            mask = attention_mask.squeeze()
            
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        if self.training and torch.is_grad_enabled():
            
            
            
            q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
            indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

            
            
            
            
            topk_indices, indexer_loss = FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                self.indexer.index_topk,
                indexer_loss_coeff,
                float_mask,
                getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                self.indexer.pg_collection,
            )
            
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                )

            
            
            
            output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

            
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        else:
            
            
            
            _, topk_indices = self.indexer.forward_with_scores(
                x, qr, mask=float_mask, packed_seq_params=packed_seq_params
            )

            
            
            
            output = unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

        return output
