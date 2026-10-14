from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any
MAX_PARALLEL_TOOL_CALLS = 2
MAX_DUPLICATE_TOOL_CALLS = 3
MAX_TOOL_RESPONSE_LENGTH = 14 * 1024
VISIT_PAGE_TEXT_CHARS = 20000
MAX_CONTEXT_TOKENS = 65536
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 1.0
MAX_ROLLOUT_RESPONSE_TOKENS = 20480
DUPLICATE_TOOL_CALL_WARNING = '[Warning] You have already called this tool with the exact same arguments multiple times and received the same result. Repeating it yields no new information. Do NOT call this tool with these arguments again — rephrase your query, try a different tool, or answer directly with what you have.'
DUPLICATE_TOOL_CALL_FINAL = '[Error] Duplicate tool call limit reached: this exact call was repeated too many times. The interaction is terminated.'
_LEADING_THINK_RE = re.compile('\\A\\s*<think>(.*?)</think>(.*?)\\s*\\Z', re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile('\\s*<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL | re.IGNORECASE)
_ANSWER_ACTION_RE = re.compile('\\s*<answer>(.*?)</answer>\\s*\\Z', re.DOTALL | re.IGNORECASE)
_PROTOCOL_TAG_RE = re.compile('</?(?:think|tool_call|tool_response|answer)>', re.IGNORECASE)

@dataclass(frozen=True)
class SerializedAssistantTurn:
    pass
    valid: bool
    reasoning: str = ''
    action: str = ''
    action_kind: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    answer: str | None = None
    error: str | None = None

def _invalid_assistant_turn(error: str) -> SerializedAssistantTurn:
    return SerializedAssistantTurn(valid=False, error=error)

def parse_serialized_assistant_turn(output: str | None) -> SerializedAssistantTurn:
    pass
    text = str(output or '').strip()
    reasoning = ''
    action = text
    if text.lower().startswith('<think>'):
        match = _LEADING_THINK_RE.fullmatch(text)
        if match is None:
            return _invalid_assistant_turn('leading reasoning block is unclosed or malformed')
        reasoning = match.group(1).strip()
        action = match.group(2).strip()
        if not reasoning:
            return _invalid_assistant_turn('reasoning block is empty')
        if _PROTOCOL_TAG_RE.search(reasoning):
            return _invalid_assistant_turn('reasoning contains a protocol tag')
    if not action:
        return _invalid_assistant_turn('missing action')
    answer_match = _ANSWER_ACTION_RE.fullmatch(action)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        if not answer:
            return _invalid_assistant_turn('answer block is empty')
        if _PROTOCOL_TAG_RE.search(answer):
            return _invalid_assistant_turn('answer contains a nested protocol tag')
        return SerializedAssistantTurn(valid=True, reasoning=reasoning, action=action, action_kind='answer', answer=answer)
    matches = list(_TOOL_CALL_BLOCK_RE.finditer(action))
    if not matches:
        return _invalid_assistant_turn('action is neither tool calls nor an answer')
    residual = _TOOL_CALL_BLOCK_RE.sub('', action)
    if residual.strip():
        return _invalid_assistant_turn('text exists outside tool-call blocks')
    if not 1 <= len(matches) <= MAX_PARALLEL_TOOL_CALLS:
        return _invalid_assistant_turn(f'tool-call count must be between 1 and {MAX_PARALLEL_TOOL_CALLS}')
    calls: list[dict[str, Any]] = []
    for tool_match in matches:
        raw = tool_match.group(1)
        if _PROTOCOL_TAG_RE.search(raw):
            return _invalid_assistant_turn('tool call contains a nested protocol tag')
        try:
            call = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return _invalid_assistant_turn('tool call is not valid JSON')
        if not isinstance(call, dict):
            return _invalid_assistant_turn('tool call JSON is not an object')
        name = call.get('name')
        arguments = call.get('arguments')
        if not isinstance(name, str) or not name.strip():
            return _invalid_assistant_turn('tool-call name is not a non-empty string')
        if not isinstance(arguments, dict):
            return _invalid_assistant_turn('tool-call arguments is not an object')
        calls.append({'name': name, 'arguments': arguments})
    return SerializedAssistantTurn(valid=True, reasoning=reasoning, action=action, action_kind='tool_calls', tool_calls=tuple(calls))

def tool_call_overflow_message(emitted: int, limit: int) -> str:
    rejected = max(0, emitted - limit)
    return f'[Error] Per-turn tool-call limit exceeded: received {emitted} calls; only the first {limit} were executed. The remaining {rejected} call(s) were NOT executed. Issue them in a later assistant turn, with at most {limit} calls per turn.'

def tool_call_key(name: str, arguments: Any) -> str:
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass
    return f'{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}'

def truncate_tool_response(text: str, *, max_length: int=MAX_TOOL_RESPONSE_LENGTH, side: str='middle') -> str:
    text = str(text or '')
    if len(text) <= max_length:
        return text
    if side == 'left':
        marker = '(truncated)...'
        if max_length <= len(marker):
            return marker[:max_length]
        return marker + text[-(max_length - len(marker)):]
    if side == 'right':
        marker = '...(truncated)'
        if max_length <= len(marker):
            return marker[:max_length]
        return text[:max_length - len(marker)] + marker
    marker = '...(truncated)...'
    if max_length <= len(marker):
        return marker[:max_length]
    payload_length = max_length - len(marker)
    left_length = payload_length // 2
    right_length = payload_length - left_length
    return text[:left_length] + marker + text[-right_length:]
