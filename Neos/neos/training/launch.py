
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from neos.paths import PROJECT_ROOT
REPO = PROJECT_ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['head', 'worker', 'mei'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--rank', type=int, help='Policy node rank; worker defaults to rank 1')
    p.add_argument('--dry-run', action='store_true', help='Print the launch contract without releasing GPUs')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    epochs = float(config['mei_epochs'])
    if not epochs.is_integer() or epochs < 1:
        raise ValueError('mei_epochs must be a positive whole number (for example 1)')
    config['mei_epochs'] = int(epochs)
    forced_correct_reward = float(config.get('forced_final_correct_reward', 1.0))
    if not 0.5 < forced_correct_reward <= 1.0:
        raise ValueError('forced_final_correct_reward must be in (0.5, 1.0]')
    ips = subprocess.check_output(['hostname', '-I'], text=True).split()
    hosts = config['policy_hosts']
    if len(hosts) not in {2, 4} or len(set(hosts)) != len(hosts) or config['mei_host'] in hosts:
        raise ValueError('Require 2 or 4 distinct policy hosts and one separate MEI host')
    rank = 0 if a.role == 'head' else a.rank if a.rank is not None else 1
    if a.role == 'head' and a.rank not in {None, 0}:
        raise ValueError('Head must be rank 0')
    if a.role == 'worker' and (not 1 <= rank < len(hosts)):
        raise ValueError('Worker rank must index a configured worker host')
    role_host = config['mei_host'] if a.role == 'mei' else hosts[rank]
    if socket.gethostbyname(role_host) not in ips and role_host not in ips:
        raise SystemExit(f'Run {a.role} on {role_host}')
    if a.role == 'mei':
        if a.dry_run:
            print(json.dumps({'role': 'mei', 'host': role_host, 'epochs': config['mei_epochs'], 'mei_k': config['mei_update_k']}))
            return
        os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('mei_node.py')), '--config', str(a.config)])
    root = config['root']
    job = config['job_id']
    batch = int(config.get('policy_batch_size', 16))
    rollout_n = int(config.get('rollout_n', 8))
    training_rows = 0
    training_file = Path(config['rl_data_dir']) / 'train.parquet'
    if training_file.is_file():
        import pyarrow.parquet as parquet
        training_rows = int(parquet.ParquetFile(training_file).metadata.num_rows)
    resume_path = config.get('resume_checkpoint', '')
    policy_model = config['initial_policy']
    if resume_path:
        checkpoint = Path(resume_path)
        if not (checkpoint / 'actor/ckpt_contents.json').is_file():
            raise ValueError(f'No saved actor checkpoint at {checkpoint}')
        completed = int(checkpoint.name.removeprefix('global_step_'))
        if not 0 < completed < int(config['policy_steps']):
            raise ValueError('Resume step must precede the requested final step')
        policy_model = str(checkpoint / 'actor/model/huggingface')
        if not (Path(policy_model) / 'model.safetensors.index.json').is_file():
            raise ValueError('Weight-only co-evolution resume requires saved HF weights')
    if batch < 1 or rollout_n < 2:
        raise ValueError('policy_batch_size must be positive and rollout_n must be at least 2')
    parallel = config.get('policy_parallelism', {})
    tp = int(parallel.get('tp', 4))
    cp = int(parallel.get('cp', 2))
    pp = int(parallel.get('pp', 1))
    ep = int(parallel.get('ep', 1))
    etp = int(parallel.get('etp', 1))
    total_gpus = 8 * len(hosts)
    if min(tp, cp, pp, ep, etp) < 1 or total_gpus % (tp * cp * pp) or total_gpus % (ep * etp * pp) or 8 % tp:
        raise ValueError('Policy attention/expert parallelism must divide the allocated GPUs; TP stays intra-node')
    gen_tp = int(config.get('rollout_tensor_parallel_size', 1))
    if gen_tp < 1 or 8 % gen_tp:
        raise ValueError('rollout_tensor_parallel_size must divide eight GPUs per node')
    policy_epochs = int(config.get('policy_epochs', 2))
    if policy_epochs < 1:
        raise ValueError('policy_epochs must be positive')
    env = {
        'NEOS_COEV_ROOT': root,
        'NEOS_COEV_K': str(config['mei_update_k']),
        'NEOS_COEV_MEI_LR': str(config['mei_learning_rate']),
        'NEOS_COEV_MEI_EPOCHS': str(config['mei_epochs']),
        'NEOS_COEV_WAIT_TIMEOUT': str(config.get('mei_wait_timeout', 7200)),
        'RL_JOB_ID': job,
        'RUN_NAME': job,
        'RUN_ROOT': root + '/policy',
        'LOG_ROOT': config['policy_log_root'],
        'MODEL_PATH': policy_model,
        'DATA_DIR': config['rl_data_dir'],
        'WORLD_SIZE': str(len(hosts)),
        'GPUS_PER_NODE': '8',
        'NODE_RANK': str(rank),
        'CURRENT_NODE_IP': role_host,
        'MODE': 'full',
        'TRAIN_BSZ': str(batch),
        'MINI_BSZ': str(batch),
        'ROLLOUT_N': str(rollout_n),
        'TOTAL_TRAINING_STEPS': str(config['policy_steps']),
        'TOTAL_EPOCHS': str(policy_epochs),
        'SAVE_FREQ': str(config.get('policy_save_freq', 10)),
        'TEST_FREQ': '-1',
        'RESUME_MODE': 'resume_path' if resume_path else 'disable',
        'RESUME_FROM_PATH': resume_path,
        'NEOS_MODEL_ONLY_CHECKPOINT': '1',
        'MAX_ACTOR_CKPT_TO_KEEP': str(config.get('policy_checkpoints_to_keep', 2)),
        'MEGATRON_TP_SIZE': str(tp),
        'MEGATRON_CP_SIZE': str(cp),
        'MEGATRON_PP_SIZE': str(pp),
        'MEGATRON_EP_SIZE': str(ep),
        'MEGATRON_ETP_SIZE': str(etp),
        'MEGATRON_FREEZE_MOE_ROUTER': 'True' if config.get('policy_freeze_router', False) else 'False',
        'MEGATRON_PARAM_OFFLOAD': 'True',
        'MEGATRON_GRAD_OFFLOAD': 'True',
        'MEGATRON_OPTIMIZER_OFFLOAD': 'True',
        'MAX_MODEL_LEN': '131072',
        'MAX_PROMPT_LENGTH': '8192',
        'MAX_RESPONSE_LENGTH': '122880',
        'MAX_ASSISTANT_TURNS_CONFIG': '50',
        'ACTOR_LR': str(config.get('policy_learning_rate', 1e-06)),
        'GEN_TP': str(gen_tp),
        'ROLLOUT_GPU_MEMORY_UTILIZATION': str(config.get('rollout_gpu_memory_utilization', 0.55)),
        'ROLLOUT_ENFORCE_EAGER': 'True',
        'ROLLOUT_MAX_NUM_SEQS': str(config.get('rollout_max_num_seqs', 16)),
        'VERL_SGLANG_NCCL_PORT_BASE': '24000',
        'VERL_ROLLOUT_STICKY_ROUTING': '0',
        'VERL_ROLLING_TRAJECTORY_WINDOW': '1',
        'VERL_ADAPTIVE_MIXED_GROUP_WINDOW': '1',
        'VERL_ROLLING_ADMISSION_RPC_CONCURRENCY': str(max(65536, 2 * training_rows * (rollout_n + rollout_n // 2) + 1024)),
        'VERL_ROLLING_ADMISSION_CONTROL_TIMEOUT_S': '60',
        'AGENT_LOOP_WORKERS': '128',
        'REWARD_WORKERS': '64',
        'REWARD_MAX_CONCURRENT': '32',
        'NEOS_TURN_CREDIT_MODEL': config.get('credit_judge_model', 'qwen3.7-plus'),
        'NEOS_TURN_CREDIT_REASONING_EFFORT': config.get('credit_reasoning_effort', 'xhigh'),
        'NEOS_TURN_CREDIT_CAUSAL_COMPONENTS': '1' if config.get('causal_component_credit', False) else '0',
        'NEOS_TURN_CREDIT_MAX_CONCURRENCY': '64',
        'NEOS_TURN_CREDIT_PREFETCH': '1',
        'NEOS_TURN_CREDIT_PREFETCH_MAX_PENDING': '128',
        'NEOS_TURN_CREDIT_REQUEST_TIMEOUT': str(config.get('credit_request_timeout', 600)),
        'NEOS_TURN_CREDIT_TOTAL_TIMEOUT': str(config.get('credit_total_timeout', 1200)),
        'NEOS_TURN_CREDIT_BATCH_WAIT_TIMEOUT': str(config.get('credit_batch_wait_timeout', 1800)),
        'NEOS_MEI_MODEL_SPEC': 'local/' + config['mei_name_prefix'] + f"-h{int(config.get('initial_mei_version', 0)):03d}",
        'NEOS_MEI_FALLBACK_MODELS': '',
        'NEOS_LOCAL_MEI_ENDPOINT': f"http://{config['mei_host']}:{config['service_port']}/v1",
        'NEOS_LOCAL_MEI_API_KEY': '',
        'NEOS_LOCAL_MEI_ENABLE_THINKING': '0',
        'NEOS_MEI_MAX_COMPLETION_TOKENS': '4096',
        'NEOS_REUSE_LOADED_IMAGES': '1' if config.get('reuse_loaded_images', False) else '0',
        'NEOS_MEI_TRACE_ROOT': root + '/mei_traces',
        'NEOS_MEI_TRACE_RUN_ID': job,
        'NEOS_MEI_TRACE_BATCH_ROWS': '1',
        'NEOS_MEI_TRACE_LAYOUT': 'writer_shards',
        'NEOS_JUDGE_MODEL': config.get('answer_judge_model', 'qwen3.7-plus'),
        'NEOS_NO_VALIDATION': '1' if config.get('no_validation', False) else '0',
        'NEOS_FORCED_FINAL_CORRECT_REWARD': str(forced_correct_reward),
    }
    for config_key, env_key in (('rollout_chunked_prefill_size', 'ROLLOUT_CHUNKED_PREFILL_SIZE'), ('rollout_max_prefill_tokens', 'ROLLOUT_MAX_PREFILL_TOKENS')):
        if config_key in config:
            value = int(config[config_key])
            if value <= 0:
                raise ValueError(f'{config_key} must be positive')
            env[env_key] = str(value)
    os.environ.update(env)
    print(json.dumps({'role': a.role, 'host': role_host, 'job': job, 'policy_batch': batch, 'rollout_n': rollout_n, 'max_assistant_turns': 50, 'policy_nodes': len(hosts), 'node_rank': rank, 'initial_policy': config['initial_policy'], 'engine_initial_weights': policy_model, 'policy_steps': config['policy_steps'], 'mei_k': config['mei_update_k'], 'policy_epochs': policy_epochs, 'policy_parallelism': {'tp': tp, 'cp': cp, 'pp': pp, 'dp': total_gpus // (tp * cp * pp), 'ep': ep, 'etp': etp}, 'rollout_tp': gen_tp, 'rollout_servers': total_gpus // gen_tp, 'mei_learning_rate': config['mei_learning_rate'], 'policy_freeze_router': config.get('policy_freeze_router', False), 'router_freeze_scope': 'expert_router_only' if config.get('policy_freeze_router', False) else None, 'forced_final_correct_reward': forced_correct_reward, 'resume_checkpoint': resume_path, 'checkpoint_load_contents': ['model'], 'credit_judge': env['NEOS_TURN_CREDIT_MODEL'], 'answer_judge': env['NEOS_JUDGE_MODEL'], 'mei_endpoint': env['NEOS_LOCAL_MEI_ENDPOINT']}, ensure_ascii=False), flush=True)
    if a.dry_run:
        return
    os.execv('/bin/bash', ['bash', str(REPO / 'scripts/run_rl_cluster.sh'), 'full'])
if __name__ == '__main__':
    main()
