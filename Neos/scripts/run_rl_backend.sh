#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

MODE=full
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEOS_ROOT="$(dirname "${SCRIPT_DIR}")"
source "${NEOS_QWEN35_ACTIVATE:-${NEOS_ROOT}/environment/activate.sh}"
cd "${NEOS_ROOT}"

: "${MODEL_PATH:?Set the initial Policy checkpoint}"
: "${DATA_DIR:?Set the RL parquet directory}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-neos-coevolution}"
RUN_ROOT="${RUN_ROOT:-${NEOS_OUTPUT_ROOT:?Configure NEOS_OUTPUT_ROOT}/rl/${RUN_NAME}/policy}"
LOG_ROOT="${LOG_ROOT:-${NEOS_OUTPUT_ROOT:?Configure NEOS_OUTPUT_ROOT}/rl/${RUN_NAME}/logs}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAINER_NNODES="${TRAINER_NNODES:-1}"
TOTAL_TRAINER_GPUS="$((GPUS_PER_NODE * TRAINER_NNODES))"
MEGATRON_TP_SIZE="${MEGATRON_TP_SIZE:-4}"
MEGATRON_PP_SIZE="${MEGATRON_PP_SIZE:-1}"
MEGATRON_CP_SIZE="${MEGATRON_CP_SIZE:-2}"
MEGATRON_EP_SIZE="${MEGATRON_EP_SIZE:-1}"
MEGATRON_ETP_SIZE="${MEGATRON_ETP_SIZE:-1}"
for integer_name in GPUS_PER_NODE TRAINER_NNODES MEGATRON_TP_SIZE \
  MEGATRON_PP_SIZE MEGATRON_CP_SIZE MEGATRON_EP_SIZE MEGATRON_ETP_SIZE; do
  integer_value="${!integer_name}"
  if ! [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_name} must be a positive integer; got ${integer_value}" >&2
    exit 2
  fi
done
MEGATRON_MODEL_PARALLEL_SIZE="$((MEGATRON_TP_SIZE * MEGATRON_PP_SIZE * MEGATRON_CP_SIZE))"
if ((TOTAL_TRAINER_GPUS % MEGATRON_MODEL_PARALLEL_SIZE != 0)); then
  echo "TP*PP*CP=${MEGATRON_MODEL_PARALLEL_SIZE} must divide total GPUs=${TOTAL_TRAINER_GPUS}" >&2
  exit 2
fi
MEGATRON_DP_SIZE="$((TOTAL_TRAINER_GPUS / MEGATRON_MODEL_PARALLEL_SIZE))"
if ((GPUS_PER_NODE % MEGATRON_TP_SIZE != 0)); then
  echo "MEGATRON_TP_SIZE=${MEGATRON_TP_SIZE} must divide GPUs per node=${GPUS_PER_NODE} so TP stays intra-node" >&2
  exit 2
fi

                                                                           
                                                                               
MBRIDGE_ROOT="${MBRIDGE_ROOT:-${NEOS_ROOT}/third_party/mbridge}"
export PYTHONPATH="${NEOS_ROOT}/verl:${NEOS_ROOT}:${MBRIDGE_ROOT}:${NEOS_ROOT}/verl:${NEOS_ROOT}/third_party/sglang/python:${NEOS_ROOT}/third_party/ms-swift:${PYTHONPATH:-}"
source "${NEOS_ROOT}/environment/tools.sh"
export NEOS_MEI_MODEL_SPEC="${NEOS_MEI_MODEL_SPEC:-local/neos-mei}"
export NEOS_MEI_FALLBACK_MODELS="${NEOS_MEI_FALLBACK_MODELS-}"
export NEOS_JUDGE_MODEL="${NEOS_JUDGE_MODEL:-qwen3.7-plus}"
export NEOS_MEI_TRACE_ROOT="${NEOS_MEI_TRACE_ROOT:-${NEOS_OUTPUT_ROOT:?Configure NEOS_OUTPUT_ROOT}/mei-traces}"
export NEOS_MEI_TRACE_RUN_ID="${NEOS_MEI_TRACE_RUN_ID:-${RUN_NAME}}"
export NEOS_MEI_MAX_CONCURRENCY="${NEOS_MEI_MAX_CONCURRENCY:-256}"
export NEOS_WEB_MEI_MAX_CONCURRENCY="${NEOS_WEB_MEI_MAX_CONCURRENCY:-128}"
export NEOS_IMAGE_MEI_MAX_CONCURRENCY="${NEOS_IMAGE_MEI_MAX_CONCURRENCY:-64}"
export NEOS_REVERSE_MEI_MAX_CONCURRENCY="${NEOS_REVERSE_MEI_MAX_CONCURRENCY:-32}"
export NEOS_TEXT_SEARCH_QUERY_WORKERS="${NEOS_TEXT_SEARCH_QUERY_WORKERS:-384}"
export HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/neos-ray-${UID}}"
export HF_DATASETS_CACHE="${RL_HF_DATASETS_CACHE:-${TMPDIR:-/tmp}/neos-hf-datasets}"
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/rollouts" "${LOG_ROOT}" "${RAY_TMPDIR}" "${HF_DATASETS_CACHE}"

MODE_ARGS=()
TRAIN_BSZ="${TRAIN_BSZ:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-122880}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
                                                                          
                                                                              
MINI_BSZ="${MINI_BSZ:-${TRAIN_BSZ}}"
MEGATRON_PARAM_OFFLOAD="${MEGATRON_PARAM_OFFLOAD:-True}"
MEGATRON_GRAD_OFFLOAD="${MEGATRON_GRAD_OFFLOAD:-True}"
MEGATRON_OPTIMIZER_OFFLOAD="${MEGATRON_OPTIMIZER_OFFLOAD:-True}"
MEGATRON_USE_REMOVE_PADDING="${MEGATRON_USE_REMOVE_PADDING:-False}"
MEGATRON_USE_FUSED_KERNELS="${MEGATRON_USE_FUSED_KERNELS:-True}"
MEGATRON_FREEZE_MOE_ROUTER="${MEGATRON_FREEZE_MOE_ROUTER:-False}"
MEGATRON_PAD_BSHD_TO_MINIBATCH_MAX=False
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-256}"
REWARD_WORKERS="${REWARD_WORKERS:-64}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-32}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1000000}"
                                                                           
                                                                          
                                                                            
                                                                         
                                                                   
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
USE_DYNAMIC_BSZ=False
MAX_ASSISTANT_TURNS_CONFIG="${MAX_ASSISTANT_TURNS_CONFIG:-50}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"
  MODE_ARGS+=(trainer.save_freq="${SAVE_FREQ:-10}" trainer.test_freq="${TEST_FREQ:-5}" trainer.resume_mode="${RESUME_MODE:-auto}")
  if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
    [[ "${RESUME_MODE:-}" == resume_path && -d "${RESUME_FROM_PATH}/actor" ]] || {
      echo "RESUME_FROM_PATH requires resume_path mode and an existing actor checkpoint" >&2
      exit 2
    }
    MODE_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
  fi
  if [[ "${NEOS_MODEL_ONLY_CHECKPOINT:-0}" =~ ^(1|true|yes|on)$ ]]; then
    MODE_ARGS+=(
      'actor_rollout_ref.actor.checkpoint.save_contents=["model"]'
      'actor_rollout_ref.actor.checkpoint.load_contents=["model"]'
    )
  fi
  if [[ -n "${MAX_ACTOR_CKPT_TO_KEEP:-}" ]]; then
    if ! [[ "${MAX_ACTOR_CKPT_TO_KEEP}" =~ ^[1-9][0-9]*$ ]]; then
      echo "MAX_ACTOR_CKPT_TO_KEEP must be a positive integer" >&2
      exit 2
    fi
    MODE_ARGS+=(trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}")
  fi
  if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    MODE_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
  fi
if [[ "${NEOS_NO_VALIDATION:-0}" == 1 ]]; then
                                                                              
                                                                             
  MODE_ARGS+=('data.val_files=[]' data.val_batch_size=1 data.validation_shuffle=False trainer.test_freq=-1 trainer.val_before_train=False)
fi
for integer_name in TRAIN_BSZ ROLLOUT_N MINI_BSZ; do
  integer_value="${!integer_name}"
  if ! [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_name} must be a positive integer; got ${integer_value}" >&2
    exit 2
  fi
done
if (( MINI_BSZ > TRAIN_BSZ || TRAIN_BSZ % MINI_BSZ != 0 )); then
  echo "MINI_BSZ is measured in prompts and must divide TRAIN_BSZ: TRAIN_BSZ=${TRAIN_BSZ} MINI_BSZ=${MINI_BSZ} rollout_n=${ROLLOUT_N}" >&2
  exit 2
fi
                                                                         
                                                                        
                                              
if [[ "${VERL_ADAPTIVE_MIXED_GROUP_WINDOW:-0}" =~ ^(1|true|yes|on)$ ]]; then
  export VERL_ADAPTIVE_BASE_SESSIONS="${ROLLOUT_N}"
  export VERL_ADAPTIVE_MAX_SESSIONS="$((ROLLOUT_N + ROLLOUT_N / 2))"
  export VERL_ADAPTIVE_GROUP_WINDOW="$((2 * TRAIN_BSZ))"
fi
EFFECTIVE_TRAJECTORY_MINI_BSZ="$((MINI_BSZ * ROLLOUT_N))"
if ((EFFECTIVE_TRAJECTORY_MINI_BSZ % MEGATRON_DP_SIZE != 0)); then
  echo "Effective trajectory mini-batch=${EFFECTIVE_TRAJECTORY_MINI_BSZ} must divide Megatron DP=${MEGATRON_DP_SIZE}" >&2
  exit 2
fi
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_MODEL_LEN + MEGATRON_CP_SIZE - 1) / MEGATRON_CP_SIZE))}"
if ! [[ "${PPO_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU must be a positive integer; got ${PPO_MAX_TOKEN_LEN_PER_GPU}" >&2
  exit 2
fi
EFFECTIVE_CP_TOKEN_BUDGET="$((PPO_MAX_TOKEN_LEN_PER_GPU * MEGATRON_CP_SIZE))"
if (( EFFECTIVE_CP_TOKEN_BUDGET < MAX_MODEL_LEN )); then
  echo "PPO token budget is too small for the configured context: per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} cp=${MEGATRON_CP_SIZE} effective=${EFFECTIVE_CP_TOKEN_BUDGET} context=${MAX_MODEL_LEN}" >&2
  exit 2
fi
export NEOS_RL_EXPECTED_CONTEXT_TOKENS="${MAX_MODEL_LEN}"
export NEOS_RL_EXPECTED_MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS_CONFIG}"

COMMON_ARGS=(
  model_engine=megatron
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.rollout_correction.bypass_mode=True
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${TRAIN_BSZ}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=False
  data.dataloader_num_workers=0
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding="${MEGATRON_USE_REMOVE_PADDING}"
  actor_rollout_ref.model.enable_gradient_checkpointing=False
  actor_rollout_ref.model.use_fused_kernels="${MEGATRON_USE_FUSED_KERNELS}"
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BSZ}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.actor.freeze_vision_tower=True
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${MEGATRON_TP_SIZE}"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${MEGATRON_PP_SIZE}"
  actor_rollout_ref.actor.megatron.context_parallel_size="${MEGATRON_CP_SIZE}"
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="${MEGATRON_EP_SIZE}"
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${MEGATRON_ETP_SIZE}"
  actor_rollout_ref.actor.megatron.sequence_parallel=True
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True
  actor_rollout_ref.actor.megatron.use_remove_padding="${MEGATRON_USE_REMOVE_PADDING}"
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max="${MEGATRON_PAD_BSHD_TO_MINIBATCH_MAX}"
  actor_rollout_ref.actor.megatron.param_offload="${MEGATRON_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.megatron.grad_offload="${MEGATRON_GRAD_OFFLOAD}"
  actor_rollout_ref.actor.megatron.optimizer_offload="${MEGATRON_OPTIMIZER_OFFLOAD}"
  ++actor_rollout_ref.actor.megatron.override_mcore_model_config.moe_config.freeze_moe_router="${MEGATRON_FREEZE_MOE_ROUTER}"
  ++actor_rollout_ref.actor.megatron.override_mcore_model_config.moe_config.router_freeze_scope=expert_router_only
  actor_rollout_ref.actor.megatron.dtype=bfloat16
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.checkpoint.mbridge_config.distributed_filesystem=True
  +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=True
  actor_rollout_ref.ref.strategy=megatron
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${MEGATRON_TP_SIZE}"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${MEGATRON_PP_SIZE}"
  actor_rollout_ref.ref.megatron.context_parallel_size="${MEGATRON_CP_SIZE}"
  actor_rollout_ref.ref.megatron.expert_model_parallel_size="${MEGATRON_EP_SIZE}"
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${MEGATRON_ETP_SIZE}"
  actor_rollout_ref.ref.megatron.sequence_parallel=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True
  actor_rollout_ref.ref.megatron.use_remove_padding="${MEGATRON_USE_REMOVE_PADDING}"
  actor_rollout_ref.ref.megatron.pad_bshd_to_minibatch_max="${MEGATRON_PAD_BSHD_TO_MINIBATCH_MAX}"
  actor_rollout_ref.ref.megatron.param_offload="${MEGATRON_PARAM_OFFLOAD}"
  actor_rollout_ref.ref.megatron.dtype=bfloat16
  actor_rollout_ref.rollout.name=sglang
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP:-1}"
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS_CONFIG}"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${NEOS_ROOT}/configs/neos/tool_config.yaml"
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=20480
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
  +actor_rollout_ref.rollout.multi_turn.max_duplicate_tool_calls=0
  actor_rollout_ref.rollout.agent.default_agent_loop=neos_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${NEOS_ROOT}/configs/neos/agent_loop.yaml"
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}"
  reward.custom_reward_function.path="${NEOS_ROOT}/neos/workflow/rl_reward.py"
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=rate_limited
  reward.num_workers="${REWARD_WORKERS}"
  +reward.max_concurrent="${REWARD_MAX_CONCURRENT}"
  +reward.timeout="${REWARD_TIMEOUT:-300}"
  trainer.critic_warmup=0
  trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}"
  trainer.v1.sampler.max_off_policy_strategy=drop
  trainer.logger='["console"]'
  trainer.project_name=neos
  trainer.experiment_name="${RUN_NAME}"
  trainer.n_gpus_per_node="${GPUS_PER_NODE}"
  trainer.nnodes="${TRAINER_NNODES}"
  trainer.balance_batch=False
  trainer.val_before_train=False
  trainer.total_epochs="${TOTAL_EPOCHS:-1}"
  trainer.rollout_data_dir="${RUN_ROOT}/rollouts"
  trainer.default_local_dir="${RUN_ROOT}/checkpoints"
)

                                                                          
                                                                              
if [[ -n "${ROLLOUT_CHUNKED_PREFILL_SIZE:-}" ]]; then
  [[ "${ROLLOUT_CHUNKED_PREFILL_SIZE}" =~ ^[1-9][0-9]*$ ]] || exit 2
  COMMON_ARGS+=(++actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size="${ROLLOUT_CHUNKED_PREFILL_SIZE}")
fi
if [[ -n "${ROLLOUT_MAX_PREFILL_TOKENS:-}" ]]; then
  [[ "${ROLLOUT_MAX_PREFILL_TOKENS}" =~ ^[1-9][0-9]*$ ]] || exit 2
  COMMON_ARGS+=(++actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens="${ROLLOUT_MAX_PREFILL_TOKENS}")
fi

echo "rl_contract=model:${MODEL_PATH} data:${TRAIN_FILE} mei:${NEOS_MEI_MODEL_SPEC} judge:${NEOS_JUDGE_MODEL} train_backend:megatron async_agent_workers:${AGENT_LOOP_WORKERS} reward_workers:${REWARD_WORKERS} context:${MAX_MODEL_LEN} prompt:${MAX_PROMPT_LENGTH} response:${MAX_RESPONSE_LENGTH} max_turns:${MAX_ASSISTANT_TURNS_CONFIG} train_prompts:${TRAIN_BSZ} mini_prompts:${MINI_BSZ} rollout_n:${ROLLOUT_N} effective_trajectory_mini_batch:${EFFECTIVE_TRAJECTORY_MINI_BSZ} megatron_tp:${MEGATRON_TP_SIZE} megatron_pp:${MEGATRON_PP_SIZE} megatron_cp:${MEGATRON_CP_SIZE} megatron_dp:${MEGATRON_DP_SIZE} megatron_ep:${MEGATRON_EP_SIZE} ppo_tokens_per_gpu:${PPO_MAX_TOKEN_LEN_PER_GPU} effective_cp_token_budget:${EFFECTIVE_CP_TOKEN_BUDGET} fused_loss:${MEGATRON_USE_FUSED_KERNELS} remove_padding:${MEGATRON_USE_REMOVE_PADDING} dynamic_bsz:${USE_DYNAMIC_BSZ} rollout_eager:${ROLLOUT_ENFORCE_EAGER} rollout_max_num_seqs:${ROLLOUT_MAX_NUM_SEQS} nodes:${TRAINER_NNODES} gpus_per_node:${GPUS_PER_NODE}"

set +e
python -m neos.workflow.train_main "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
if ((status != 0)); then
  echo "Neos RL ${MODE} failed; log=${LOG_FILE}" >&2
  exit "${status}"
fi
echo "neos_rl_complete=true mode=${MODE} log=${LOG_FILE}"
