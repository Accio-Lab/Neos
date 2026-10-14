
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
from neos.paths import PROJECT_ROOT, project_path
from neos.configuration import resolve_recipe_paths, output_path
from neos.workflow.shared_state_io import atomic_json

def resolve_config(recipe, *, policy_hosts, mei_host, run_id=None, run_dir=None, model=None, mei_model=None, data_dir=None, resume_checkpoint=None, resume_source_root=None, dry_run=False):
    run_id = run_id or os.getenv('NEOS_RUN_ID') or 'neos-coevolution'
    c = dict(recipe)
    for field, value in [('initial_policy', model), ('initial_mei', mei_model), ('rl_data_dir', data_dir)]:
        if value is not None:
            c[field] = value
    c = resolve_recipe_paths(c)
    c['answer_judge_model'] = os.getenv('NEOS_JUDGE_MODEL') or c['answer_judge_model']
    c['credit_judge_model'] = os.getenv('NEOS_TURN_CREDIT_MODEL') or c['credit_judge_model']
    if not re.fullmatch('[A-Za-z0-9._-]+', run_id):
        raise ValueError('Invalid run ID')
    if not policy_hosts:
        if not dry_run:
            raise ValueError('Provide --policy-hosts or NEOS_POLICY_HOSTS')
        policy_hosts = [f'policy{i}' for i in range(4)]
    if not mei_host:
        if not dry_run:
            raise ValueError('Provide --mei-host or NEOS_MEI_HOST')
        mei_host = 'mei'
    if len(policy_hosts) not in {2, 4} or len(set(policy_hosts)) != len(policy_hosts) or mei_host in policy_hosts:
        raise ValueError('Use two/four distinct Policy hosts and a separate MEI host')
    if not dry_run:
        policy_hosts = [socket.gethostbyname(host) for host in policy_hosts]
        mei_host = socket.gethostbyname(mei_host)
        if len(set(policy_hosts)) != len(policy_hosts) or mei_host in policy_hosts:
            raise ValueError('Hostnames must resolve to distinct instances')
    root = project_path(run_dir) if run_dir else output_path(f'rl/{run_id}')
    c.update(job_id=run_id, root=str(root), policy_log_root=str(root / 'logs'), policy_terminal_status=str(root / 'logs/rank0.status'), policy_hosts=policy_hosts, mei_host=mei_host, mei_name_prefix=run_id + '-mei')
    for key, value in [('initial_policy', model or c['initial_policy']), ('initial_mei', mei_model or c['initial_mei']), ('rl_data_dir', data_dir or c['rl_data_dir'])]:
        c[key] = str(project_path(value))
    if resume_checkpoint:
        checkpoint = project_path(resume_checkpoint)
        c['resume_checkpoint'] = str(checkpoint)
        c['resume_source_root'] = str(project_path(resume_source_root) if resume_source_root else checkpoint.parents[2])
    return c

def inherit_resume(c):
    if not c.get('resume_checkpoint'):
        return
    old = Path(c['resume_source_root'])
    root = Path(c['root'])
    completed = int(Path(c['resume_checkpoint']).name.removeprefix('global_step_'))
    if not (Path(c['resume_checkpoint']) / 'actor/ckpt_contents.json').is_file():
        raise FileNotFoundError('Missing model-only Policy checkpoint manifest')
    if old.resolve() == root.resolve():
        raise ValueError('Resume into a fresh run directory')
    if not old.is_dir():
        raise FileNotFoundError(old)
    for path in sorted((old / 'job_results').glob('after-step-*.json')):
        step = int(path.stem.rsplit('-', 1)[-1])
        if step > completed:
            continue
        result = json.loads(path.read_text())
        if result.get('status') not in {'trained', 'skipped_insufficient_rows'}:
            continue
        atomic_json(root / 'job_results' / path.name, result)
        data = old / 'mei_training' / f'after-step-{step}' / 'data'
        target = root / 'mei_training' / f'after-step-{step}' / 'data'
        if data.exists() and (not target.exists()):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(data.resolve(), target_is_directory=True)
        if result.get('status') == 'trained':
            c.update(initial_mei=result['checkpoint'], initial_mei_version=int(result['version']), initial_mei_after_step=step)
    if completed >= c['mei_update_k'] and (not c.get('initial_mei_after_step')):
        raise ValueError('Resume requires the matching committed MEI history')
    for path in sorted((old / 'jobs').glob('after-step-*.json')):
        step = int(path.stem.rsplit('-', 1)[-1])
        if not c.get('initial_mei_after_step', 0) < step <= completed:
            continue
        job = json.loads(path.read_text())
        atomic_json(root / 'jobs' / path.name, job)
        target = root / 'mei_training' / f'after-step-{step}' / 'data'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((old / 'mei_training' / f'after-step-{step}' / 'data').resolve(), target_is_directory=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['0', '1', '2', '3', 'mei'])
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--policy-hosts', default=os.getenv('NEOS_POLICY_HOSTS', ''))
    p.add_argument('--mei-host', default=os.getenv('NEOS_MEI_HOST', ''))
    p.add_argument('--run-id', default=os.getenv('NEOS_RUN_ID') or None, help='Optional shared run name; defaults to neos-coevolution on all hosts.')
    p.add_argument('--run-dir')
    p.add_argument('--model')
    p.add_argument('--mei-model')
    p.add_argument('--data-dir')
    p.add_argument('--resume-checkpoint')
    p.add_argument('--resume-source-root')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    c = resolve_config(json.loads(a.config.read_text()), policy_hosts=[s.strip() for s in a.policy_hosts.split(',') if s.strip()], mei_host=a.mei_host, run_id=a.run_id, run_dir=a.run_dir, model=a.model, mei_model=a.mei_model, data_dir=a.data_dir, resume_checkpoint=a.resume_checkpoint, resume_source_root=a.resume_source_root, dry_run=a.dry_run)
    print(f"RL run: {c['job_id']}; output: {c['root']}", file=sys.stderr, flush=True)
    if a.role != 'mei' and int(a.role) >= len(c['policy_hosts']):
        raise ValueError('Role outside configured Policy hosts')
    if a.dry_run:
        print(json.dumps({'role': a.role, 'config': c}, indent=2))
        return
    root = Path(c['root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'configuration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inherit_resume(c)
        effective = root / 'configuration.json'
        if effective.exists() and json.loads(effective.read_text()) != c:
            raise ValueError('All roles must use identical configuration; choose a new run ID for changes')
        atomic_json(effective, c)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    role_lock = (root / f'role-{a.role}.lock').open('a')
    fcntl.flock(role_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (logs / 'rank0.status').exists():
        raise RuntimeError('Completed/failed run: resume into a new run ID')
    role = 'mei' if a.role == 'mei' else 'head' if a.role == '0' else 'worker'
    command = [sys.executable, '-u', '-m', 'neos.training.launch', role, '--config', str(effective)]
    if role != 'mei':
        command += ['--rank', a.role]
    env = dict(os.environ)
    env.update(VERL_MEGATRON_PREWARM_GRAD_STATS='1', VERL_SGLANG_SCOPE_LOCAL_TP_SERVER_CUDA='1', VERL_QWEN35_FIX_REPLICATED_KV_OUTPUT_GATE='1', VERL_ADAPTIVE_PROGRESS_CHECK_SECONDS='10')
    child = None

    def stop(sig, frame):
        raise SystemExit(128 + sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        with (logs / f'role-{a.role}.log').open('ab') as log:
            for line in iter(child.stdout.readline, b''):
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        raise SystemExit(child.wait())
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
if __name__ == '__main__':
    main()
