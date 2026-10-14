
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('NEOS_ROOT', Path(__file__).resolve().parents[1])).resolve()


def project_path(value):
    if value is None or not str(value).strip():
        raise ValueError('A required path is empty; configure Neos/config.local.json')
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():return path
    return PROJECT_ROOT / path
