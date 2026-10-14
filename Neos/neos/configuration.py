import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = frozenset({
    'NEOS_VENV', 'CUDA_HOME', 'NEOS_BASE_MODEL_9B', 'NEOS_BASE_MODEL_35B',
    'NEOS_POLICY_9B_MODEL', 'NEOS_POLICY_35B_MODEL', 'NEOS_MEI_MODEL',
    'NEOS_SFT_DATA', 'NEOS_MEI_TRAIN_DATA', 'NEOS_MEI_VALIDATION_DATA',
    'NEOS_RL_INPUT_JSONL', 'NEOS_RL_DATA_DIR', 'NEOS_EVAL_DATA_ROOT',
    'NEOS_TEXT_SEARCH_DATA_ROOT', 'NEOS_OUTPUT_ROOT', 'NEOS_SFT_CKPT_9B',
    'NEOS_SFT_CKPT_35B', 'NEOS_SFT_CKPT_MEI', 'NEOS_SERVE_POLICY_MODEL',
})


def config_file():
    selected = os.environ.get('NEOS_CONFIG', '').strip()
    if selected:
        path = Path(selected).expanduser()
        return path if path.is_absolute() else ROOT / path
    local = ROOT / 'config.local.json'
    return local if local.is_file() else ROOT / 'config.example.json'


def settings():
    path = config_file()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    allowed = set(json.loads((ROOT / 'config.example.json').read_text()))
    unknown = set(data) - allowed
    if unknown:
        raise ValueError('Unknown configuration fields: ' + ', '.join(sorted(unknown)))
    for key, value in data.items():
        if not re.fullmatch('[A-Z][A-Z0-9_]*', key):
            raise ValueError('Invalid configuration field name')
        if key.endswith('_HEADERS'):
            if not isinstance(value, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in value.items()):
                raise ValueError(f'{key} must be a JSON object of strings')
        elif key == 'NEOS_POLICY_HOSTS':
            if not isinstance(value, list) or not all(isinstance(v,str) for v in value):
                raise ValueError(f'{key} must be an array of hostnames or IPs')
        elif not isinstance(value, str):
            raise ValueError(f'{key} must be a string')
    return data


def configured_environment():
    raw = settings()
    def expand(value, trail=()):
        def substitute(match):
            key = match[1] or match[2]
            if key in trail:
                raise ValueError('Cyclic configuration reference: ' + ' -> '.join((*trail,key)))
            referenced = os.environ.get(key, raw.get(key))
            if not isinstance(referenced,str) or not referenced.strip():
                raise ValueError(f'Configuration reference {key} is empty or undefined')
            return expand(referenced, (*trail,key))
        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)', substitute, value)
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = json.dumps(value) if value else ''
        elif isinstance(value, list):
            value = ','.join(value)
        value = os.environ.get(key, value)
        if not str(value).strip():
            continue
        if key in PATH_KEYS:
            path = Path(expand(str(value), (key,))).expanduser()
            value = str(path if path.is_absolute() else ROOT / path)
        result[key] = str(value)
    output = result.get('NEOS_OUTPUT_ROOT') or os.getenv('NEOS_OUTPUT_ROOT')
    if output:
        for key, suffix in [('NEOS_EVAL_ARTIFACT_ROOT','artifacts'),
                            ('NEOS_MEI_TRACE_ROOT','mei-traces')]:
            result[key] = os.environ.get(key) or str(Path(output) / suffix)
    return result


def apply_environment():
    os.environ.update(configured_environment())


def required_setting(key):
    value = os.getenv(key, '').strip()
    if not value:
        raise ValueError(f'Fill {key} in Neos/config.local.json (or set that environment variable)')
    return value


def output_path(suffix):
    return Path(required_setting('NEOS_OUTPUT_ROOT')) / suffix


def evaluation_output_path(*, benchmark, repeat_count, run_id=None, out=None):
    if out:
        return Path(out).expanduser().resolve()
    root = output_path('evaluations').expanduser().resolve()
    if run_id:
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
            raise ValueError('Run ID may contain letters, digits, dot, underscore or dash')
        return root / run_id
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return Path(tempfile.mkdtemp(prefix=f'{benchmark}-repeat{repeat_count}-{stamp}-', dir=root))


def resolve_recipe_paths(recipe):
    result = dict(recipe)
    for field, key in result.get('path_settings', {}).items():
        if not result.get(field):
            result[field] = required_setting(key)
    if not result.get('output') and result.get('output_subdir'):
        result['output'] = str(output_path(result['output_subdir']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shell', action='store_true')
    args = parser.parse_args()
    if args.shell:
        for key, value in configured_environment().items():
            print(f'export {key}={shlex.quote(value)}')
    else:
        data = configured_environment()
        print(json.dumps({'config_file': str(config_file()),
                          'configured_fields': sorted(data)}, indent=2))
