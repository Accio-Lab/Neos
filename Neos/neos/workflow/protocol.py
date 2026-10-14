from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any
from neos.workflow.toolsets import NEOS_TOOLSET_PROFILE, adapt_system_prompt
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_TOOL_NAMES, ArgumentNormalizationError, normalize_tool_arguments
NEOS_PROTOCOL_VERSION = 'neos-native-tool-calling'
NATIVE_ASSISTANT_FORMAT_PROFILE = 'openai_native'
QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE = 'qwen35_native_compat'
QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION = 'neos-qwen35-native-tool-normalization'
ASSISTANT_FORMAT_PROFILES = (NATIVE_ASSISTANT_FORMAT_PROFILE, QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE)
MAX_ASSISTANT_TURNS = 50
MAX_TOOL_CALLS_PER_TURN = 1
MAX_CONTEXT_TOKENS = 131072
MAX_TOOL_RESPONSE_LENGTH = 20 * 1024
MAX_ROLLOUT_RESPONSE_TOKENS = 16000
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 1.0
_RESERVED_SERIALIZATION_TAGS = ('<think>', '</think>', '<thinking>', '</thinking>', '<tool_call>', '</tool_call>', '<answer>', '</answer>')

def assistant_format_adapter_contract_version(profile: str) -> str | None:
    if profile == NATIVE_ASSISTANT_FORMAT_PROFILE:
        return None
    if profile == QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE:
        return QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION
    raise ValueError(f'unknown Neos assistant format profile: {profile}')
NEOS_SYSTEM_PROMPT = f"You are a long-horizon multimodal web-search agent. Solve the user's question by interleaving careful reasoning with live-web retrieval and selective visual observation. Work for as many steps as needed, but return the best-supported final answer within {MAX_ASSISTANT_TURNS} assistant turns.\n\nEVIDENCE DISCIPLINE\n1. Track the question's constraints and unresolved evidence gaps. Use each action to reduce a specific uncertainty, revise hypotheses when evidence conflicts, and verify decisive facts before answering.\n2. Distinguish visual evidence from metadata. Titles, snippets, alt text, captions, URLs, and MEI summaries are discovery signals, not proof of what an image's pixels contain.\n3. Search results deliberately contain only lightweight references. A candidate image becomes visible evidence only after FetchImage loads it into the visual register. CropImage creates another reusable visual observation from pixels already in that register.\n4. Inspect fine-grained visual details when relevant: text, logos, faces, object parts, colors, counts, spatial relations, and background clues. Do not claim a pixel-level fact about an image that has not been loaded.\n5. Prefer focused, complementary searches and source pages. Cross-check ambiguous identities and final claims; do not repeatedly issue an identical call expecting new information.\n\nTOOL WORKFLOW\n- TextSearch discovers webpages as title-URL-snippet records.\n- WebVisit extracts goal-relevant evidence and up to fifteen lightweight page-image references.\n- ImageSearch discovers and goal-ranks image candidates without exposing their pixels to you.\n- ReverseImageSearch searches from up to three loaded image tokens or image URLs and returns textual match evidence plus lightweight candidate references; it does not load candidate pixels.\n- FetchImage loads up to five selected URLs in parallel and appends successful images to the visual register in request order.\n- CropImage crops one or more loaded img_N observations with normalized 0..1000 [left, top, right, bottom] boxes. It processes requested crops in parallel and appends them in deterministic request order.\n\nNATIVE ACTION CONTRACT\n- Tool definitions are supplied through the API tools field. Use native function calls only. Never print XML tags, JSON tool-call wrappers, or a textual imitation of a function call.\n- On each assistant turn, take exactly one action: either issue exactly one native function call or return the final answer as ordinary text. Batch independent queries, URLs, images, or boxes inside that one function call when its schema permits.\n- Think carefully before choosing the action. Do not duplicate a narration of the intended function call in answer text.\n- Do not fabricate tool results. The environment returns a role=tool message whose tool_call_id matches the assistant tool call and reports any newly loaded img_N references.\n- Tool-returned pixels are registered as img_N only when the environment explicitly reports them. Only use existing visual-register tokens.\n- When the complete answer is supported, stop calling tools and return only the concise exact answer requested by the user as ordinary text. Do not wrap it in XML, Markdown, an answer label, or an explanation unless the question explicitly asks for one.\n\nBefore answering, synthesize the visual and web evidence and check that the visual identification and all answer constraints are supported."
if NEOS_TOOLSET_PROFILE != 'full':
    NEOS_SYSTEM_PROMPT = adapt_system_prompt(NEOS_SYSTEM_PROMPT)
    _toolset_suffix = '-' + NEOS_TOOLSET_PROFILE.replace('_', '-')
    NEOS_PROTOCOL_VERSION += _toolset_suffix

def system_prompt_for_turn_limit(turn_limit: int) -> str:
    pass
    value = int(turn_limit)
    if value == MAX_ASSISTANT_TURNS:
        return NEOS_SYSTEM_PROMPT
    raise ValueError(f'unsupported Neos turn limit: {value}')

def protocol_version_for_turn_limit(turn_limit: int) -> str:
    value = int(turn_limit)
    if value == MAX_ASSISTANT_TURNS:
        return NEOS_PROTOCOL_VERSION
    raise ValueError(f'unsupported Neos turn limit: {value}')

@dataclass(frozen=True)
class NeosAssistantTurn:
    valid: bool
    reasoning: str = ''
    action_kind: str | None = None
    tool_call: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    message: dict[str, Any] | None = None
    adaptation: dict[str, Any] | None = None
    argument_normalizations: list[dict[str, str]] | None = None

def _invalid(error: str) -> NeosAssistantTurn:
    return NeosAssistantTurn(valid=False, error=error)

def _text_content(value: Any) -> str | None:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict) or item.get('type') != 'text':
                return None
            parts.append(str(item.get('text') or ''))
        return '\n'.join(parts)
    return None

def parse_neos_assistant_message(message: dict[str, Any] | None, *, tool_schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> NeosAssistantTurn:
    pass
    if not isinstance(message, dict):
        return _invalid('assistant response is not a message object')
    if message.get('role', 'assistant') != 'assistant':
        return _invalid('response message role is not assistant')
    reasoning_value = message.get('reasoning_content')
    if reasoning_value is None:
        reasoning_value = message.get('reasoning')
    if reasoning_value is not None and (not isinstance(reasoning_value, str)):
        return _invalid('assistant reasoning_content is not a string')
    reasoning = str(reasoning_value or '').strip()
    content = _text_content(message.get('content'))
    if content is None:
        return _invalid('assistant content is neither text nor text blocks')
    for field_name, field_value in (('reasoning_content', reasoning), ('content', content)):
        lowered = field_value.lower()
        leaked = next((tag for tag in _RESERVED_SERIALIZATION_TAGS if tag in lowered), None)
        if leaked is not None:
            return _invalid(f'serving parser leaked reserved serialization token {leaked!r} into assistant {field_name}; the model output is not a valid native message')
    calls = message.get('tool_calls') or []
    if not isinstance(calls, list):
        return _invalid('assistant tool_calls is not a list')
    if calls:
        if len(calls) != MAX_TOOL_CALLS_PER_TURN:
            return _invalid(f'Neos requires exactly one native tool call; got {len(calls)}')
        if content.strip():
            return _invalid('native tool-call turn also contains visible answer content')
        raw_call = calls[0]
        if not isinstance(raw_call, dict):
            return _invalid('native tool call is not an object')
        call_id = str(raw_call.get('id') or '').strip()
        if not call_id:
            return _invalid('native tool call has no id')
        if raw_call.get('type', 'function') != 'function':
            return _invalid('native tool call type is not function')
        function = raw_call.get('function')
        if not isinstance(function, dict):
            return _invalid('native tool call has no function object')
        name = str(function.get('name') or '').strip()
        if name not in NEOS_TOOL_NAMES:
            return _invalid(f'unknown Neos tool {name!r}; available tools: ' + ', '.join(sorted(NEOS_TOOL_NAMES)))
        raw_arguments = function.get('arguments')
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                return _invalid(f'native function.arguments is invalid JSON: {exc}')
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            return _invalid('native function.arguments must decode to an object')
        try:
            arguments, argument_normalizations = normalize_tool_arguments(name, arguments, schema_version=tool_schema_version)
        except ArgumentNormalizationError as exc:
            return _invalid(str(exc))
        wire_arguments = json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))
        normalized_call = {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': wire_arguments}}
        normalized_message = {'role': 'assistant', 'content': '', 'tool_calls': [normalized_call]}
        if reasoning:
            normalized_message['reasoning_content'] = reasoning
        return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='tool_call', tool_call={'id': call_id, 'name': name, 'arguments': arguments}, message=normalized_message, argument_normalizations=argument_normalizations)
    answer = content.strip()
    if not answer:
        return _invalid('assistant returned neither a native tool call nor an answer')
    normalized_message = {'role': 'assistant', 'content': answer}
    if reasoning:
        normalized_message['reasoning_content'] = reasoning
    return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='answer', answer=answer, message=normalized_message)

def image_reference_convention(n_question_images: int) -> str:
    if n_question_images <= 0:
        return 'The visual register is initially empty. FetchImage outputs begin at img_1; CropImage outputs continue the same sequence.'
    if n_question_images == 1:
        return 'The question image is already loaded as img_1. FetchImage and CropImage outputs are appended as img_2, img_3, ... in the order reported by the environment.'
    return f'The {n_question_images} question images are already loaded as img_1 through img_{n_question_images}. FetchImage and CropImage outputs append from img_{n_question_images + 1} onward in the order reported by the environment.'

def build_user_prompt(question: str, n_question_images: int) -> str:
    return f'Question: {str(question).strip()}\n\nUse the available live-web and visual-observation tools to investigate the question. Return only the concise exact answer as ordinary text once the evidence chain is supported.\n\n' + image_reference_convention(n_question_images)

def build_final_turn_instruction(turn_limit: int=MAX_ASSISTANT_TURNS) -> str:
    return f'FINAL TURN ({turn_limit}/{turn_limit}): tools are disabled. Using only the evidence already collected, return the best-supported concise exact answer now as ordinary text, with no answer label or surrounding markup.'
