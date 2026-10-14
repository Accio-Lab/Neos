
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from neos.paths import PROJECT_ROOT, project_path
from neos.configuration import resolve_recipe_paths


def build_command(config, *, python=sys.executable):
    cfg = resolve_recipe_paths(config)
    world = int(cfg.get('nodes', 1)) * int(cfg.get('gpus_per_node', 8))
    tp, cp, pp = [int(cfg.get(k, 1)) for k in ('tp', 'cp', 'pp')]
    ep, etp = int(cfg.get('ep', 1)), int(cfg.get('etp', 1))
    if min(tp, cp, pp, ep, etp) < 1 or world % (tp * cp * pp) or world % (ep * etp * pp):
        raise ValueError('Model/expert parallelism must divide the allocated GPUs')
    dp = world // (tp * cp * pp)
    micro, batch = int(cfg['micro_batch']), int(cfg['global_batch'])
    if micro < 1 or batch % (micro * dp):
        raise ValueError('Global batch must be divisible by micro-batch * DP')
    if float(cfg['epochs']) != int(cfg['epochs']) or int(cfg['epochs']) < 1:
        raise ValueError('Epochs must be a positive integer')
    if cfg.get('padding_free', True) and cfg.get('group_by_length', False):
        raise ValueError('Length grouping requires padded batches')
    model, dataset, output = (project_path(cfg[k]) for k in ('model', 'dataset', 'output'))
    plugin = project_path(cfg.get('plugin', 'plugins/qwen35_neos_native_tools.py'))
    args = [python, '-m', 'torch.distributed.run',
        f'--nnodes={cfg.get("nodes", 1)}', f'--nproc-per-node={cfg.get("gpus_per_node", 8)}',
        f'--node-rank={os.getenv("NODE_RANK", "0")}',
        f'--master-addr={os.getenv("MASTER_ADDR", "127.0.0.1")}',
        f'--master-port={os.getenv("MASTER_PORT", "29676")}', '--max-restarts=0',
        '--module', 'swift.cli._megatron.sft']
    options = {
        'model': model, 'model_type': cfg['model_type'], 'external_plugins': plugin,
        'dataset': dataset, 'output_dir': output, 'template': 'qwen3_5',
        'agent_template': 'qwen3_5', 'tuner_type': 'full', 'torch_dtype': 'bfloat16',
        'bf16': True, 'freeze_llm': False, 'freeze_vit': True, 'freeze_aligner': True,
        'max_length': cfg['max_length'], 'max_pixels': 786432,
        'split_dataset_ratio': 0, 'strict': True, 'truncation_strategy': 'delete',
        'padding_free': cfg.get('padding_free', True), 'packing': False,
        'group_by_length': cfg.get('group_by_length', False), 'loss_scale': 'default',
        'preserve_thinking': cfg.get('preserve_thinking', True),
        'add_non_thinking_prefix': True,
        'tensor_model_parallel_size': tp, 'context_parallel_size': cp,
        'pipeline_model_parallel_size': pp, 'expert_model_parallel_size': ep,
        'expert_tensor_parallel_size': etp, 'sequence_parallel': tp > 1,
        'micro_batch_size': micro, 'global_batch_size': batch,
        'num_train_epochs': int(cfg['epochs']), 'lr': cfg['learning_rate'],
        'min_lr': cfg.get('min_lr', 1e-6), 'lr_decay_style': 'cosine',
        'lr_warmup_fraction': 0.05, 'weight_decay': 0.1,
        'adam_beta1': 0.9, 'adam_beta2': 0.95, 'adam_eps': 1e-8, 'clip_grad': 1.0,
        'attention_backend': 'flash', 'cross_entropy_loss_fusion': True,
        'gradient_accumulation_fusion': False, 'use_distributed_optimizer': True,
        'recompute_granularity': 'full', 'recompute_method': 'uniform',
        'recompute_num_layers': 1, 'finetune': True, 'nccl_comm_warmup': True,
        'manual_gc': True, 'manual_gc_steps': 1,
        'dataset_num_proc': 8, 'dataloader_num_workers': cfg.get('data_workers', 4),
        'train_dataloader_shuffle': True, 'logging_steps': 1, 'report_to': 'tensorboard',
        'seed': 42, 'data_seed': 42, 'no_save_optim': True, 'no_save_rng': True,
        'save_safetensors': True, 'save_strategy': 'epoch', 'save_total_limit': 2,
    }
    if cfg.get('optimizer_cpu_offload', False):
        options.update(use_precision_aware_optimizer=True, optimizer_cpu_offload=True,
                       optimizer_offload_fraction=1.0)
    else:
        options['optimizer_cpu_offload'] = False
    if cfg.get('freeze_router', False):
        options['freeze_parameters_regex'] = r'(?:^|\.)mlp\.router\.'
    if cfg.get('enable_thinking') is not None:
        options['enable_thinking'] = cfg['enable_thinking']
    if cfg.get('validation'):
        options.update(val_dataset=project_path(cfg['validation']), val_dataset_shuffle=False,
                       eval_iters=2, eval_steps=cfg.get('eval_steps', 240))
    for key, value in options.items():
        args += ['--' + key, str(value).lower() if isinstance(value, bool) else str(value)]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--model'); p.add_argument('--dataset'); p.add_argument('--output')
    p.add_argument('--validation'); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    for key in ('model', 'dataset', 'output', 'validation'):
        if getattr(a, key) is not None: config[key] = getattr(a, key)
    config = resolve_recipe_paths(config)
    command = build_command(config)
    if a.dry_run:
        print(json.dumps({'recipe': config['name'], 'command': command}, indent=2)); return
    for key in ('dataset', 'validation'):
        if config.get(key) and not project_path(config[key]).is_file():
            raise FileNotFoundError(project_path(config[key]))
    model_cfg = json.loads((project_path(config['model']) / 'config.json').read_text())
    if model_cfg['model_type'] != config['model_type']:
        raise ValueError('Checkpoint architecture does not match this recipe')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    if any(line.strip().isdigit() for line in active.splitlines()):
        raise RuntimeError('The training GPUs already have a workload; nothing was stopped')
    output = project_path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training-command.json').exists():
        raise FileExistsError('Choose a fresh output directory; existing results are not overwritten')
    (output / 'training-command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = dict(os.environ, USE_MCORE_GDN='1', CUDA_DEVICE_MAX_CONNECTIONS='1',
               IMAGE_MAX_TOKEN_NUM='1728', TOKENIZERS_PARALLELISM='false',
               SFT_ROUTER_AUDIT_DIR=str(output / 'router-audit'))
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('NCCL_IB_DISABLE', '1' if config.get('nodes', 1) == 1 else '0')
    env.setdefault('NCCL_SOCKET_IFNAME', 'lo' if config.get('nodes', 1) == 1 else 'eth0')
    env.setdefault('GLOO_SOCKET_IFNAME', env['NCCL_SOCKET_IFNAME'])
    child = None
    def stop(sig, frame):
        raise SystemExit(128 + sig)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    code = 1
    try:
        child = subprocess.Popen(command, env=env, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (output / 'train.log').open('ab') as log:
            for line in iter(child.stdout.readline, b''):
                log.write(line); log.flush()
                sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
        code = child.wait()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        (output / 'exit.json').write_text(json.dumps({'code': code,
            'at_utc': datetime.now(timezone.utc).isoformat()}) + '\n')
    raise SystemExit(code)


if __name__ == '__main__': main()
