
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import time
import uuid

ROUTE_SCHEMA = 'neos-credit-model-route'


def resolve_request_model(kwargs, environ=None):
    env = os.environ if environ is None else environ
    result = dict(kwargs)
    configured = str(kwargs.get('model') or '')
    meta = {'configured_model': configured, 'request_model': configured,
            'route_applied': False, 'route_version': None}
    root_value = env.get('NEOS_COEV_ROOT')
    job_id = env.get('RL_JOB_ID')
    if not root_value or not job_id:
        return result, meta
    root = Path(root_value)
    route_file = root / 'credit_model_route.json'
    if not route_file.is_file():
        return result, meta
    route = json.loads(route_file.read_text())
    if route.get('job_id') != job_id or root.name != job_id:
        raise ValueError('Credit route job identity mismatch')
    if route.get('schema_version') != ROUTE_SCHEMA or route.get('status') != 'active':
        raise ValueError('Invalid/inactive credit route file')
    first = int(route['effective_from_policy_step'])
    if first < 1 or int(route['activation_boundary']['completed_policy_step']) != first - 1:
        raise ValueError('Credit route must start at a completed-policy boundary')
    if route['activation_boundary'].get('credit_http_workers') != 0:
        raise ValueError('Credit route was not activated with drained HTTP workers')
    state = json.loads((root / 'policy.json').read_text())
    step = int(state['step'])
    target = str(route['target_model']).strip()
    if not target or any(c.isspace() for c in target):
        raise ValueError('Invalid credit target model')
    if configured not in {route['source_model'], target}:
        raise ValueError('Credit route does not match the requested source model')
    meta.update(job_id=job_id, route_version=route['route_version'],
                effective_from_policy_step=first, observed_policy_step=step,
                observed_policy_phase=state.get('phase'),
                audit_root=str(root), request_uuid=uuid.uuid4().hex,
                request_scope='diagnostic' if env.get('NEOS_CREDIT_ROUTE_PROBE') == '1' else 'training')
    if step >= first:
        result['model'] = target
        meta.update(request_model=target, route_applied=True)
    return result, meta


def append_route_audit(meta, event, **fields):
    pass
    if not meta.get('audit_root'):
        return None
    record = {'schema_version': ROUTE_SCHEMA, 'event': event, 'at_unix': time.time(),
              'hostname': socket.gethostname(), 'worker_pid': os.getpid(),
              'parent_pid': os.getppid(),
              **{k: v for k, v in meta.items() if k != 'audit_root'}, **fields}
    payload = (json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    descriptor = None
    try:
        descriptor = os.open(Path(meta['audit_root'])/'credit_model_route_audit.jsonl',
                             os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(descriptor, payload)
    except OSError as exc:
        
        return f'{type(exc).__name__}: {exc}'
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return None
