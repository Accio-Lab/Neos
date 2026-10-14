from __future__ import annotations
from pathlib import Path
NEOS_CHAT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'qwen_vl_tool_chat_template.jinja'

def load_neos_chat_template() -> str:
    return NEOS_CHAT_TEMPLATE_PATH.read_text(encoding='utf-8')
