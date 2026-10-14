import json
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4


def new_id():
    return uuid4().hex


def path_namespace(path):
    return Path(*Path(path).expanduser().resolve().parts[1:])


def file_key(value):
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)
    encoded = quote(text, safe='')
    return '/'.join('item-' + encoded[i:i+100] for i in range(0, len(encoded), 100))
