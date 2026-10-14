#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

                                                         
                                                                               
                                               
                                                              

MODE=full
[[ "${1:-}" == full ]] && shift
: "${RL_JOB_ID:?Set the same unique RL_JOB_ID on all nodes}"
RANK="${RANK:-${NODE_RANK:-}}"
: "${RANK:?Set RANK=0 on the head and a unique RANK on each worker}"
WORLD_SIZE="${WORLD_SIZE:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
if [[ ! "${WORLD_SIZE}" =~ ^(2|4)$ || ! "${RANK}" =~ ^[0-3]$ || "${GPUS_PER_NODE}" != "8" ]] || (( RANK >= WORLD_SIZE )); then
  echo "This launcher requires WORLD_SIZE=2 or 4, 0 <= RANK < WORLD_SIZE, and GPUS_PER_NODE=8" >&2
  exit 2
fi
if [[ ! "${RL_JOB_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RL_JOB_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

NEOS_ROOT="${NEOS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${NEOS_ROOT}/environment/config.sh"
MBRIDGE_ROOT="${MBRIDGE_ROOT:-${NEOS_ROOT}/third_party/mbridge}"
INNER_SCRIPT="${INNER_SCRIPT:-${NEOS_ROOT}/scripts/run_rl_backend.sh}"
RUN_NAME="${RUN_NAME:-${RL_JOB_ID}}"
RUN_ROOT="${RUN_ROOT:-${NEOS_OUTPUT_ROOT:?Configure NEOS_OUTPUT_ROOT}/rl/${RUN_NAME}/policy}"
LOG_ROOT="${LOG_ROOT:-${NEOS_OUTPUT_ROOT:?Configure NEOS_OUTPUT_ROOT}/rl/${RUN_NAME}/logs}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-${LOG_ROOT}/bootstrap}"
CLUSTER_FILE="${BOOTSTRAP_ROOT}/${RL_JOB_ID}.cluster"
STATUS_FILE="${BOOTSTRAP_ROOT}/${RL_JOB_ID}.status"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
JOIN_TIMEOUT_SECONDS="${JOIN_TIMEOUT_SECONDS:-900}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-$(mktemp -d /tmp/neos-rl-XXXXXXXX)}"
RAY_SOCKET_PATH_TEMPLATE="${RAY_TEMP_DIR%/}/session_YYYY-MM-DD_HH-MM-SS_ffffff_1234567890/sockets/plasma_store"
if (( ${#RAY_SOCKET_PATH_TEMPLATE} > 107 )); then
  echo "RAY_TEMP_DIR is too long for Ray AF_UNIX sockets: template_length=${#RAY_SOCKET_PATH_TEMPLATE} limit=107 path=${RAY_TEMP_DIR}" >&2
  exit 2
fi

[[ -f "${MBRIDGE_ROOT}/mbridge/__init__.py" ]] || {
  echo "Missing Qwen3.5-capable mbridge source: ${MBRIDGE_ROOT}" >&2
  exit 2
}
export MBRIDGE_ROOT
source "${NEOS_QWEN35_ACTIVATE:-${NEOS_ROOT}/environment/activate.sh}"
                                                                             
                                                                             
                                                                              
export PYTHONPATH="${NEOS_ROOT}/verl:${NEOS_ROOT}:${MBRIDGE_ROOT}:${NEOS_ROOT}/verl:${NEOS_ROOT}/third_party/sglang/python:${NEOS_ROOT}/third_party/ms-swift:${PYTHONPATH:-}"
RAY_PYTHON="${RAY_PYTHON:-$(command -v python)}"
if [[ ! -x "${RAY_PYTHON}" ]]; then
  echo "Ray Python is not executable: ${RAY_PYTHON}" >&2
  exit 2
fi
if ! "${RAY_PYTHON}" - <<'PY'
import os
import ray
import transfer_queue
import verl
from mbridge import AutoBridge
print(
    "ray_python_preflight=ok "
    f"python={__import__('sys').executable} "
    f"ray={ray.__file__} transfer_queue={transfer_queue.__file__} "
    f"mbridge={AutoBridge.__module__} verl={verl.__file__}"
)
if not str(verl.__file__).startswith(
    os.environ['NEOS_ROOT']+'/verl/'
):
    raise SystemExit(f"stale VERL source selected: {verl.__file__}")
PY
then
  echo "The Qwen3.5 Ray Python cannot import ray and transfer_queue: ${RAY_PYTHON}" >&2
  exit 2
fi
RAY_CLI=("${RAY_PYTHON}" -m ray.scripts.scripts)
mkdir -p "${BOOTSTRAP_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}" "${RAY_TEMP_DIR}"
exec > >(tee -a "${LOG_ROOT}/rank${RANK}.log") 2>&1

NEOS_NOFILE_LIMIT="${NEOS_NOFILE_LIMIT:-65536}"
if ! [[ "${NEOS_NOFILE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NEOS_NOFILE_LIMIT must be a positive integer; got ${NEOS_NOFILE_LIMIT}" >&2
  exit 2
fi
nofile_hard="$(ulimit -Hn)"
nofile_target="${NEOS_NOFILE_LIMIT}"
if [[ "${nofile_hard}" != "unlimited" ]] && (( nofile_target > nofile_hard )); then
  nofile_target="${nofile_hard}"
fi
ulimit -Sn "${nofile_target}"
nofile_soft="$(ulimit -Sn)"
if [[ "${nofile_soft}" != "unlimited" ]] && (( nofile_soft < 8192 )); then
  echo "Ray requires a soft open-file limit of at least 8192; got soft=${nofile_soft} hard=${nofile_hard}" >&2
  exit 2
fi
export NEOS_NOFILE_LIMIT
echo "nofile_preflight=ok requested=${NEOS_NOFILE_LIMIT} soft=${nofile_soft} hard=${nofile_hard}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "${NEOS_ROOT}/environment/tools.sh"
export NEOS_MEI_MODEL_SPEC="${NEOS_MEI_MODEL_SPEC:-local/neos-mei}"
export NEOS_MEI_FALLBACK_MODELS="${NEOS_MEI_FALLBACK_MODELS-}"
export NEOS_JUDGE_MODEL="${NEOS_JUDGE_MODEL:-qwen3.7-plus}"
export NEOS_MEI_MAX_CONCURRENCY="${NEOS_MEI_MAX_CONCURRENCY:-256}"
export NEOS_WEB_MEI_MAX_CONCURRENCY="${NEOS_WEB_MEI_MAX_CONCURRENCY:-128}"
export NEOS_IMAGE_MEI_MAX_CONCURRENCY="${NEOS_IMAGE_MEI_MAX_CONCURRENCY:-64}"
export NEOS_REVERSE_MEI_MAX_CONCURRENCY="${NEOS_REVERSE_MEI_MAX_CONCURRENCY:-32}"
export NEOS_TEXT_SEARCH_QUERY_WORKERS="${NEOS_TEXT_SEARCH_QUERY_WORKERS:-384}"
export HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${RL_HF_DATASETS_CACHE:-/tmp/neos-hf-datasets-${RL_JOB_ID}}"
mkdir -p "${HF_DATASETS_CACHE}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

CURRENT_NODE_IP="${CURRENT_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "${CURRENT_NODE_IP}" ]]; then
  echo "Unable to determine CURRENT_NODE_IP" >&2
  exit 1
fi

RUN_STATUS_PUBLISHED=0
publish_status() {
  local status="$1"
  if [[ "${RANK}" == "0" && "${RUN_STATUS_PUBLISHED}" == "0" ]]; then
    printf '%s\n' "${status}" >"${STATUS_FILE}.tmp.$$"
    mv "${STATUS_FILE}.tmp.$$" "${STATUS_FILE}"
    RUN_STATUS_PUBLISHED=1
  fi
}
cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  publish_status "${status}"
  "${RAY_CLI[@]}" stop --force >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if pgrep -x raylet >/dev/null 2>&1; then
  echo "An existing Ray runtime is present; refusing to stop it." >&2
  trap - EXIT INT TERM
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '/^[0-9]+$/ {found=1} END {exit !found}'; then
  echo "An existing GPU workload is present; nothing was stopped." >&2
  trap - EXIT INT TERM
  exit 2
fi
if [[ -e "${STATUS_FILE}" ]]; then
  echo "Run ID already completed; choose a new ID." >&2
  trap - EXIT INT TERM
  exit 2
fi
if [[ "${RANK}" == "0" ]]; then
  rm -f "${STATUS_FILE}"
  {
    printf 'MASTER_ADDR=%s\n' "${CURRENT_NODE_IP}"
    printf 'RAY_PORT=%s\n' "${RAY_PORT}"
  } >"${CLUSTER_FILE}.tmp.$$"
  mv "${CLUSTER_FILE}.tmp.$$" "${CLUSTER_FILE}"
  MASTER_ADDR="${CURRENT_NODE_IP}"
else
  ready=0
  for _ in $(seq 1 "${JOIN_TIMEOUT_SECONDS}"); do
    if [[ -s "${CLUSTER_FILE}" ]]; then
      MASTER_ADDR="$(sed -n 's/^MASTER_ADDR=//p' "${CLUSTER_FILE}" | head -n 1)"
      published_port="$(sed -n 's/^RAY_PORT=//p' "${CLUSTER_FILE}" | head -n 1)"
      if [[ -n "${MASTER_ADDR}" && "${published_port}" =~ ^[0-9]+$ ]]; then
        RAY_PORT="${published_port}"
        ready=1
        break
      fi
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "Timed out waiting for ${CLUSTER_FILE}" >&2
    exit 1
  fi
fi

export MASTER_ADDR RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
echo "cluster_contract=job:${RL_JOB_ID} rank:${RANK}/${WORLD_SIZE} node_ip:${CURRENT_NODE_IP} master:${RAY_ADDRESS} transport:NCCL mei:${NEOS_MEI_MODEL_SPEC} judge:${NEOS_JUDGE_MODEL} ray_temp:${RAY_TEMP_DIR}"

if [[ "${RANK}" == "0" ]]; then
  "${RAY_CLI[@]}" start --head \
    --node-ip-address="${CURRENT_NODE_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --temp-dir="${RAY_TEMP_DIR}" \
    --disable-usage-stats
else
  joined=0
  for _ in $(seq 1 "${JOIN_TIMEOUT_SECONDS}"); do
    if "${RAY_CLI[@]}" start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="${CURRENT_NODE_IP}" \
      --num-gpus="${GPUS_PER_NODE}" \
      --temp-dir="${RAY_TEMP_DIR}" \
      --disable-usage-stats; then
      joined=1
      break
    fi
    if [[ -s "${STATUS_FILE}" ]]; then
      exit "$(<"${STATUS_FILE}")"
    fi
    sleep 5
  done
  if [[ "${joined}" != "1" ]]; then
    echo "Timed out joining Ray head ${RAY_ADDRESS}" >&2
    exit 1
  fi
  echo "ray_worker_joined=true; waiting for rank-0 status"
  while [[ ! -s "${STATUS_FILE}" ]]; do sleep 5; done
  status="$(<"${STATUS_FILE}")"
  if [[ ! "${status}" =~ ^[0-9]+$ ]]; then
    echo "Invalid rank-0 status: ${status}" >&2
    exit 1
  fi
  exit "${status}"
fi

cluster_ready=0
for _ in $(seq 1 "${JOIN_TIMEOUT_SECONDS}"); do
  alive_nodes="$("${RAY_PYTHON}" - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
print(sum(bool(node.get("Alive")) for node in ray.nodes()))
ray.shutdown()
PY
)"
  if [[ "${alive_nodes}" == "${WORLD_SIZE}" ]]; then
    cluster_ready=1
    break
  fi
  sleep 2
done
if [[ "${cluster_ready}" != "1" ]]; then
  echo "Ray cluster did not reach ${WORLD_SIZE} alive nodes" >&2
  exit 1
fi
echo "ray_cluster_ready=true nodes=${WORLD_SIZE} total_gpus=$((WORLD_SIZE * GPUS_PER_NODE))"

export RUN_NAME RUN_ROOT LOG_ROOT
export TRAINER_NNODES="${WORLD_SIZE}" GPUS_PER_NODE
export RL_TRAINER_MODE="${RL_TRAINER_MODE:-colocate_async}"

bash "${INNER_SCRIPT}" \
  trainer.v1.trainer_mode="${RL_TRAINER_MODE}" \
  +ray_kwargs.ray_init.address=auto \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
  "$@"
