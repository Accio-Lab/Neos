from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    extra_groups_for_head_shards,
)
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated import Mixer2RMSNormGated
from sglang.srt.layers.attention.mamba.ops import (
    mamba_chunk_scan_combined,
    selective_state_update,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_loader.weight_utils import (
    composed_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.utils import is_cpu, is_cuda, is_npu, set_weight_attrs

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu as causal_conv1d_fn,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_update_npu as causal_conv1d_update,
    )

LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def mamba_v2_sharded_weight_loader(
    shard_spec: List[Tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures the the all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        
        boundary, loaded_boundary = 0, 0

        
        if is_cpu():
            full_dim_sum = 0
            full_dim_list = []
            weight_full_dim_list = []
            for full_dim, _, _ in shard_spec:
                full_dim_sum = full_dim_sum + full_dim
                full_dim_list.append(full_dim)
            for full_dim in full_dim_list:
                weight_full_dim_list.append(
                    int(full_dim / full_dim_sum * loaded_weight.size(0))
                )

        
        for full_dim, extra, duplicate_groups in shard_spec:
            
            
            
            
            
            

            
            shard_size = full_dim // tp_size

            
            
            
            
            
            rank = 0 if duplicate_groups else tp_rank

            
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            
            take = min(shard_size, full_dim - extra - loaded_skip)

            
            
            if is_cpu() and loaded_weight.size(0) % tp_size != 0:
                import copy

                loaded_weight_ = copy.deepcopy(loaded_weight)
                q, k, v = torch.split(
                    loaded_weight_,
                    weight_full_dim_list,
                    dim=0,
                )
                pad_qk = torch.zeros(
                    full_dim_list[0] - weight_full_dim_list[0],
                    loaded_weight.size(1),
                    loaded_weight.size(2),
                ).to(loaded_weight.dtype)
                pad_v = torch.zeros(
                    full_dim_list[2] - weight_full_dim_list[2],
                    loaded_weight.size(1),
                    loaded_weight.size(2),
                ).to(loaded_weight.dtype)
                q = torch.cat((q, pad_qk), dim=0)
                k = torch.cat((k, pad_qk), dim=0)
                v = torch.cat((v, pad_v), dim=0)
                loaded_weight_qk = torch.cat((q, k), dim=0)
                loaded_weight = torch.cat((loaded_weight_qk, v), dim=0)

            
            
            
            
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)  # type: ignore[misc]
            ]  # type: ignore[misc]

            
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


class MambaMixer2(torch.nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(
        self,
        cache_params: Mamba2CacheParams,
        hidden_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.num_heads = num_heads = cache_params.shape.num_heads
        self.head_dim = cache_params.shape.head_dim

        assert (
            num_heads % self.tp_size == 0
        ), "Tensor parallel world size must divide num heads."

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_groups, "
            "then num_groups must equal 1."
        )

        assert (
            (n_groups % self.tp_size == 0) or self.tp_size == 1 or quant_config is None
        ), (
            "Tensor parallel currently supported for quantized models only "
            "if tensor parallel world size divides num groups."
        )

        self.ssm_state_size = cache_params.shape.ssm_state_size
        self.activation = activation

        conv_kernel_size = cache_params.shape.conv_kernel
        self.intermediate_size = intermediate_size = (
            cache_params.shape.intermediate_size
        )
        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            
            
            
            groups = extra_groups_for_head_shards(n_groups, self.tp_size)
            self.n_groups = n_groups + groups
        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = cache_params.shape.conv_dim

        if n_groups % self.tp_size == 0:
            self.conv1d = MergedColumnParallelLinear(
                input_size=conv_kernel_size,
                output_sizes=[
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                ],
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            self.in_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[
                    intermediate_size,
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ],
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
        else:
            
            

            self.conv1d = ColumnParallelLinear(
                input_size=conv_kernel_size,
                output_size=self.conv_dim,
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            self.in_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=intermediate_size + self.conv_dim + self.num_heads,
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )

            
            
            
            
            
            
            group_shard_settings = (
                self.groups_ssm_state_size,  
                (self.n_groups - n_groups) * self.ssm_state_size,  
                n_groups == 1,  
            )
            intermediate_settings = (intermediate_size, 0, False)
            head_settings = (self.num_heads, 0, False)

            
            
            
            
            delattr(self.conv1d.bias, "weight_loader")
            set_weight_attrs(
                self.conv1d.bias,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        self.tp_rank,
                    )
                },
            )

            delattr(self.conv1d.weight, "weight_loader")
            set_weight_attrs(
                self.conv1d.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        self.tp_rank,
                    )
                },
            )

            if quant_config is None:
                
                delattr(self.in_proj.weight, "weight_loader")
                set_weight_attrs(
                    self.in_proj.weight,
                    {
                        "weight_loader": mamba_v2_sharded_weight_loader(
                            [
                                intermediate_settings,  
                                intermediate_settings,
                                group_shard_settings,
                                group_shard_settings,
                                head_settings,  
                            ],
                            self.tp_size,
                            self.tp_rank,
                        )
                    },
                )

        
        
        
        
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        
        
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float())
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        self.prefix = prefix

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        metadata: Mamba2Metadata,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        
        
        
        
        state_indices_tensor = metadata.mamba_cache_indices
        conv_state = layer_cache.conv[0]
        ssm_state = layer_cache.temporal

        query_start_loc = metadata.query_start_loc

        
        projected_states, _ = self.in_proj(hidden_states)

        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        
        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        num_prefills = metadata.num_prefills  
        num_decodes = metadata.num_decodes  
        num_decode_tokens = (
            num_decodes * metadata.draft_token_num
            if metadata.is_target_verify
            else num_decodes
        )
        num_prefill_tokens = metadata.num_prefill_tokens  
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens
        assert num_actual_tokens == projected_states.shape[0]

        
        
        
        hidden_states_B_C_p, hidden_states_B_C_d = torch.split(
            hidden_states_B_C,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )
        dt_p, dt_d = torch.split(
            dt,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )
        
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            state_indices_tensor,
            [num_prefills, num_decodes],
            dim=0,
        )
        query_start_loc_p = query_start_loc[: num_prefills + 1] if has_prefill else None

        
        

        preallocated_ssm_out = torch.empty(
            [
                projected_states.shape[0],
                (self.num_heads * self.head_dim) // self.tp_size,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )

        
        if has_prefill:
            mixed_metadata = metadata.mixed_metadata
            assert mixed_metadata is not None
            
            
            
            has_initial_states_p = mixed_metadata.has_initial_states
            prep_initial_states = mixed_metadata.prep_initial_states
            cache_indices = state_indices_tensor_p
            x = hidden_states_B_C_p.transpose(
                0, 1
            )  
            ccfn = (
                causal_conv1d_fn
                if not use_triton_causal_conv
                else causal_conv1d_fn_triton
            )
            hidden_states_B_C_p = ccfn(
                x,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc_p,
                seq_lens_cpu=mixed_metadata.extend_seq_lens_cpu,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(hidden_states_B_C_p)

            
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[state_indices_tensor_p],
                    0,
                )

            
            varlen_state = mamba_chunk_scan_combined(
                hidden_states_p.view(
                    1, num_prefill_tokens, self.num_heads // self.tp_size, self.head_dim
                ),
                dt_p.unsqueeze(0),
                self.A,
                B_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                C_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                chunk_size=mixed_metadata.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=mixed_metadata.seq_idx,
                chunk_indices=mixed_metadata.chunk_indices,
                chunk_offsets=mixed_metadata.chunk_offsets,
                cu_seqlens=query_start_loc_p,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(
                    1, num_prefill_tokens, -1, self.head_dim
                ),
                state_dtype=ssm_state.dtype,
            )

            
            
            ssm_state[state_indices_tensor_p] = varlen_state

        
        if has_decode:
            is_target_verify = metadata.is_target_verify

            
            if is_target_verify:
                assert (
                    use_triton_causal_conv
                ), "Speculative decoding requires use_triton_causal_conv=True for intermediate state support"
                assert isinstance(
                    layer_cache, MambaPool.SpeculativeState
                ), "layer_cache must be SpeculativeState for speculative decoding"
                draft_token_num = metadata.draft_token_num
                self.intermediate_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=state_indices_tensor_d.device
                )

                
                hidden_states_B_C_d_reshaped = hidden_states_B_C_d.view(
                    num_decodes, draft_token_num, -1
                ).transpose(1, 2)

                hidden_states_B_C_d_processed = causal_conv1d_update_triton(
                    hidden_states_B_C_d_reshaped,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=state_indices_tensor_d[:num_decodes],
                    intermediate_conv_window=layer_cache.intermediate_conv_window[0],
                    intermediate_state_indices=self.intermediate_state_indices,
                    retrieve_next_token=metadata.retrieve_next_token,
                    retrieve_next_sibling=metadata.retrieve_next_sibling,
                    retrieve_parent_token=metadata.retrieve_parent_token,
                )
                hidden_states_B_C_d = hidden_states_B_C_d_processed.transpose(
                    1, 2
                ).view(num_decode_tokens, -1)
            else:
                ccu = (
                    causal_conv1d_update
                    if not use_triton_causal_conv
                    else causal_conv1d_update_triton
                )
                hidden_states_B_C_d = ccu(
                    hidden_states_B_C_d,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=state_indices_tensor_d,
                )

            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(hidden_states_B_C_d)

            
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            if is_target_verify:
                selective_state_update(
                    ssm_state,
                    hidden_states_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    dt_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    A_d,
                    B_d.view(num_decodes, draft_token_num, n_groups, -1),
                    C_d.view(num_decodes, draft_token_num, n_groups, -1),
                    D_d,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_tensor_d[:num_decodes],
                    out=preallocated_ssm_out_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    disable_state_update=True,
                    intermediate_states_buffer=layer_cache.intermediate_ssm,
                    cache_steps=draft_token_num,
                    retrieve_parent_token=metadata.retrieve_parent_token,
                    intermediate_state_indices=self.intermediate_state_indices,
                )
            else:
                selective_state_update(
                    ssm_state,
                    hidden_states_d,
                    dt_d,
                    A_d,
                    B_d,
                    C_d,
                    D_d,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_tensor_d,
                    out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
                )

        
        
        
        
        hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])

        
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    @property
    def mamba_type(self) -> str:
        return "mamba2"
