from __future__ import annotations
import base64
import io
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image
from neos.eval.trace_artifacts import EvaluationArtifactStore, describe_dataset_image, describe_unpersisted_image
from neos.workflow.image_codec import codec_config, context_image_data_uri, prepare_context_images
from neos.workflow.models import NeosModelRequestError
from neos.workflow.native_adapter import adapt_qwen35_native_assistant_message, build_format_retry_instruction
from neos.workflow.protocol import ASSISTANT_FORMAT_PROFILES, MAX_ASSISTANT_TURNS, MAX_ROLLOUT_RESPONSE_TOKENS, MAX_TOOL_RESPONSE_LENGTH, NEOS_PROTOCOL_VERSION, NEOS_SYSTEM_PROMPT, NATIVE_ASSISTANT_FORMAT_PROFILE, QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION, ROLLOUT_TEMPERATURE, ROLLOUT_TOP_P, assistant_format_adapter_contract_version, build_final_turn_instruction, build_user_prompt, parse_neos_assistant_message
from neos.workflow.runtime import NeosRuntime
from neos.workflow.toolsets import NEOS_TOOLSET_PROFILE, redact_disabled_tool_mentions
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_TOOL_NAMES, get_neos_tool_schemas
from neos.workflow.visual_results import VISUAL_RESULT_SCHEMA_VERSION, verify_execution_output_refs
TRACE_SCHEMA_VERSION = 'neos-agent-trace'
TIMING_SCHEMA_VERSION = 'neos-agent-timing'
logger = logging.getLogger(__name__)
MAX_FORMAT_RECOVERY_RETRIES = 2
MAX_TURN_MODEL_REQUEST_RETRIES = 8
MAX_FORMAT_RETRY_REASONING_ECHO_CHARS = 8000
MAX_FORMAT_RETRY_CONTENT_ECHO_CHARS = 4000
_REJECTED_SERIALIZATION_TOKEN_RE = re.compile('</?(?:think|thinking|tool_call|tool_response|answer)>|<\\|(?:im_start|im_end|assistant|endoftext)[^>]*\\|>', re.IGNORECASE)
AGENT_TURN_CHECKPOINT_SCHEMA_VERSION = 'neos-agent-turn-checkpoint'

@dataclass(frozen=True)
class AgentToolResult:
    pass
    execution: Any
    error: str | None
    latency_ms: float

@dataclass(frozen=True)
class AgentToolRequest:
    pass
    turn: int
    name: str
    arguments: dict[str, Any]
    runtime: NeosRuntime = field(repr=False, compare=False)
    registry: tuple[Image.Image, ...] = field(repr=False, compare=False)
    source_urls: tuple[str | None, ...] = field(repr=False, compare=False)
    schema_version: str = 'neos-agent-tool-event'
    event: str = 'tool_execution_requested'

    def execute(self) -> AgentToolResult:
        started = time.perf_counter()
        try:
            execution = self.runtime.execute(self.name, self.arguments, registry=list(self.registry), source_urls=list(self.source_urls))
            execution_error = None
        except Exception as exc:
            from neos.tools.runtime import ToolExecution
            execution = ToolExecution(f'Error executing {self.name}: {type(exc).__name__}: {exc}')
            execution_error = f'{type(exc).__name__}: {exc}'
        return AgentToolResult(execution=execution, error=execution_error, latency_ms=round((time.perf_counter() - started) * 1000, 3))

@dataclass(frozen=True)
class AgentTurnComplete:
    pass
    turn: int
    next_turn: int
    tool_name: str
    model_calls: int
    tool_calls_executed: int
    schema_version: str = 'neos-agent-turn-event'
    event: str = 'tool_turn_complete'

def _checkpoint_registry(messages: list[dict]) -> list[Image.Image]:
    pass
    images: list[Image.Image] = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                continue
            image_url = part.get('image_url') or {}
            url = image_url.get('url') if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith('data:image/'):
                continue
            _, encoded = url.split(',', 1)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
                images.append(opened.convert('RGB'))
    return images

def _read_turn_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != AGENT_TURN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f'unsupported Agent turn checkpoint: {path}')
    return payload

def _write_turn_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 384)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

def _pil_content(image: Image.Image, codec_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    return {'type': 'image_url', 'image_url': {'url': context_image_data_uri(image, codec_overrides)}}

def _truncate(text: str, limit: int=MAX_TOOL_RESPONSE_LENGTH) -> str:
    value = str(text or '')
    if len(value) <= limit:
        return value
    head = limit * 3 // 5
    tail = limit - head
    return value[:head] + f'\n...[observation middle-truncated: {len(value) - limit} chars]...\n' + value[-tail:]

def _load_inputs(image_inputs: list[str | Path | Image.Image] | None) -> tuple[list[Image.Image], list[Path | None]]:
    images = []
    paths: list[Path | None] = []
    for item in image_inputs or []:
        if isinstance(item, Image.Image):
            images.append(item.convert('RGB'))
            paths.append(None)
        else:
            path = Path(item).expanduser().resolve()
            with Image.open(path) as opened:
                images.append(opened.convert('RGB'))
            paths.append(path)
    return (images, paths)

def _usage_from_provider(metadata: dict | None, aggregate: Counter[str]) -> None:
    if not metadata:
        return
    usage = metadata.get('response', {}).get('usage') or {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = usage.get(key)
        if isinstance(value, int) and (not isinstance(value, bool)):
            aggregate[key] += value
    details = usage.get('completion_tokens_details') or {}
    if isinstance(details.get('reasoning_tokens'), int):
        aggregate['reasoning_tokens'] += details['reasoning_tokens']
    prompt_details = usage.get('prompt_tokens_details') or {}
    if isinstance(prompt_details.get('cached_tokens'), int):
        aggregate['cached_tokens'] += prompt_details['cached_tokens']

def _rejected_message_for_retry(output: Any) -> dict[str, Any]:
    pass
    if not isinstance(output, dict):
        return {'role': 'assistant', 'content': redact_disabled_tool_mentions(str(output or ''))}
    reasoning = output.get('reasoning_content')
    if reasoning is None:
        reasoning = output.get('reasoning')
    if not isinstance(reasoning, str):
        reasoning = ''
    reasoning = _REJECTED_SERIALIZATION_TOKEN_RE.sub('', reasoning)
    reasoning = reasoning[-MAX_FORMAT_RETRY_REASONING_ECHO_CHARS:]
    content = output.get('content')
    if isinstance(content, list):
        content = '\n'.join((str(item.get('text') or '') for item in content if isinstance(item, dict) and item.get('type') == 'text'))
    elif not isinstance(content, str):
        content = ''
    content = _REJECTED_SERIALIZATION_TOKEN_RE.sub('', content)
    content = content[-MAX_FORMAT_RETRY_CONTENT_ECHO_CHARS:]
    calls = output.get('tool_calls') or []
    if calls:
        rendered = json.dumps(calls, ensure_ascii=False, default=str)
        rendered = rendered[-MAX_FORMAT_RETRY_CONTENT_ECHO_CHARS:]
        content = f'{content}\n\n[Rejected function-call payload]\n{rendered}'.strip()
    if not content and (not reasoning):
        content = '[The rejected response contained no usable action.]'
    result: dict[str, Any] = {'role': 'assistant', 'content': redact_disabled_tool_mentions(content)}
    if reasoning:
        result['reasoning_content'] = redact_disabled_tool_mentions(reasoning)
    return result

def iter_agent_turns(*, question: str, image_inputs: list[str | Path | Image.Image] | None, model, runtime: NeosRuntime, image_source_urls: list[str | None] | None=None, max_turns: int=MAX_ASSISTANT_TURNS, artifact_store: EvaluationArtifactStore | None=None, max_generation_tokens: int=MAX_ROLLOUT_RESPONSE_TOKENS, temperature: float=ROLLOUT_TEMPERATURE, top_p: float=ROLLOUT_TOP_P, assistant_format_profile: str=NATIVE_ASSISTANT_FORMAT_PROFILE, tool_schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION, turn_checkpoint_path: str | Path | None=None) -> Generator[AgentToolRequest | AgentTurnComplete, AgentToolResult | None, dict[str, Any]]:
    pass
    agent_started = time.perf_counter()
    timing_totals: Counter[str] = Counter()
    if max_turns <= 0:
        raise ValueError('max_turns must be positive')
    if assistant_format_profile not in ASSISTANT_FORMAT_PROFILES:
        raise ValueError(f'unsupported Neos assistant format profile; got assistant_format_profile={assistant_format_profile!r}')
    adapter_enabled = assistant_format_profile == QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE
    if tool_schema_version != DEFAULT_NEOS_SCHEMA_VERSION:
        raise ValueError(f'unsupported Neos tool schema: {tool_schema_version!r}')
    adapter_version = QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION
    runtime_schema_version = str((getattr(runtime, 'config', None) or {}).get('tool_schema_version', tool_schema_version))
    if runtime_schema_version != tool_schema_version:
        raise ValueError(f'agent/runtime tool schema mismatch: {tool_schema_version!r} != {runtime_schema_version!r}')
    phase_started = time.perf_counter()
    input_images, input_paths = _load_inputs(image_inputs)
    timing_totals['input_image_load_ms'] += (time.perf_counter() - phase_started) * 1000
    codec_overrides = getattr(runtime, 'config', None)
    phase_started = time.perf_counter()
    registry = prepare_context_images(input_images, codec_overrides)
    timing_totals['input_image_prepare_ms'] += (time.perf_counter() - phase_started) * 1000
    source_registry = list(image_source_urls or [])[:len(registry)]
    source_registry.extend([None] * (len(registry) - len(source_registry)))
    phase_started = time.perf_counter()
    input_artifacts = []
    for index, (image, path) in enumerate(zip(input_images, input_paths), 1):
        ref = f'img_{index}'
        source_url = source_registry[index - 1]
        if path is not None:
            input_artifacts.append(describe_dataset_image(image, path, image_ref=ref, source_url=source_url))
        elif artifact_store is not None:
            input_artifacts.append(artifact_store.save_image(image, image_ref=ref, kind='input', source_url=source_url))
        else:
            input_artifacts.append(describe_unpersisted_image(image, image_ref=ref, kind='input'))
    timing_totals['input_artifact_ms'] += (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    user_content = [{'type': 'text', 'text': build_user_prompt(question, len(registry))}, *(_pil_content(image, codec_overrides) for image in registry)]
    if not getattr(model, 'official_multimodal_message_order', False):
        user_content = [*(_pil_content(image, codec_overrides) for image in registry), {'type': 'text', 'text': build_user_prompt(question, len(registry))}]
    messages = [{'role': 'system', 'content': NEOS_SYSTEM_PROMPT}, {'role': 'user', 'content': user_content}]
    timing_totals['initial_message_build_ms'] += (time.perf_counter() - phase_started) * 1000
    trajectory: list[dict[str, Any]] = []
    tool_schemas = get_neos_tool_schemas(tool_schema_version)
    final_answer = None
    termination_reason = 'max_turns'
    model_usage: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    model_calls = 0
    format_adapted_turns = 0
    format_recovery_model_calls = 0
    model_request_retry_count = 0
    model_request_retry_errors: Counter[str] = Counter()
    format_transform_counts: Counter[str] = Counter()
    protocol_metrics: Counter[str] = Counter()
    forced_finalization = False
    forced_finalization_succeeded = False
    prior_agent_wall_ms = 0.0
    resume_turn_index = 0
    turn_checkpoint_resume_count = 0
    input_image_count = len(input_paths)
    checkpoint_path = Path(turn_checkpoint_path).expanduser().resolve() if turn_checkpoint_path is not None else None
    checkpoint = _read_turn_checkpoint(checkpoint_path) if checkpoint_path else None
    if checkpoint is not None:
        messages = list(checkpoint['messages'])
        registry = _checkpoint_registry(messages)
        source_registry = list(checkpoint['source_registry'])
        if len(registry) != len(source_registry):
            raise ValueError(f'Agent turn checkpoint image registry mismatch: images={len(registry)} sources={len(source_registry)}')
        trajectory = list(checkpoint['trajectory'])
        input_artifacts = list(checkpoint.get('input_artifacts') or [])
        input_image_count = int(checkpoint.get('input_image_count') or 0)
        timing_totals = Counter(checkpoint.get('timing_totals') or {})
        final_answer = checkpoint.get('final_answer')
        termination_reason = str(checkpoint.get('termination_reason') or 'max_turns')
        model_usage = Counter(checkpoint.get('model_usage') or {})
        tool_counts = Counter(checkpoint.get('tool_counts') or {})
        model_calls = int(checkpoint.get('model_calls') or 0)
        format_adapted_turns = int(checkpoint.get('format_adapted_turns') or 0)
        format_recovery_model_calls = int(checkpoint.get('format_recovery_model_calls') or 0)
        model_request_retry_count = int(checkpoint.get('model_request_retry_count') or 0)
        model_request_retry_errors = Counter(checkpoint.get('model_request_retry_errors') or {})
        format_transform_counts = Counter(checkpoint.get('format_transform_counts') or {})
        protocol_metrics = Counter(checkpoint.get('protocol_metrics') or {})
        forced_finalization = bool(checkpoint.get('forced_finalization'))
        forced_finalization_succeeded = bool(checkpoint.get('forced_finalization_succeeded'))
        prior_agent_wall_ms = float(checkpoint.get('agent_wall_ms') or 0.0)
        resume_turn_index = int(checkpoint['resume_turn_index'])
        if not 0 <= resume_turn_index < max_turns:
            raise ValueError(f'invalid Agent checkpoint resume turn: {resume_turn_index}')
        turn_checkpoint_resume_count = int(checkpoint.get('turn_checkpoint_resume_count') or 0) + 1
        logger.info('Neos resuming Agent from turn checkpoint path=%s turn=%d', checkpoint_path, resume_turn_index + 1)

    def save_turn_checkpoint(turn_index: int) -> None:
        if checkpoint_path is None:
            return
        _write_turn_checkpoint(checkpoint_path, {'schema_version': AGENT_TURN_CHECKPOINT_SCHEMA_VERSION, 'created_at_utc': time.time(), 'question': question, 'max_turns': max_turns, 'assistant_format_profile': assistant_format_profile, 'tool_schema_version': tool_schema_version, 'resume_turn_index': turn_index, 'messages': messages, 'source_registry': source_registry, 'trajectory': trajectory, 'input_artifacts': input_artifacts, 'input_image_count': input_image_count, 'timing_totals': dict(timing_totals), 'final_answer': final_answer, 'termination_reason': termination_reason, 'model_usage': dict(model_usage), 'tool_counts': dict(tool_counts), 'model_calls': model_calls, 'format_adapted_turns': format_adapted_turns, 'format_recovery_model_calls': format_recovery_model_calls, 'model_request_retry_count': model_request_retry_count, 'model_request_retry_errors': dict(model_request_retry_errors), 'format_transform_counts': dict(format_transform_counts), 'protocol_metrics': dict(protocol_metrics), 'forced_finalization': forced_finalization, 'forced_finalization_succeeded': forced_finalization_succeeded, 'agent_wall_ms': prior_agent_wall_ms + (time.perf_counter() - agent_started) * 1000, 'turn_checkpoint_resume_count': turn_checkpoint_resume_count})
    for turn_index in range(resume_turn_index, max_turns):
        turn = turn_index + 1
        forced_final = turn == max_turns
        final_instruction = None
        if forced_final:
            forced_finalization = True
            final_instruction = build_final_turn_instruction(max_turns)
            request_messages = [*messages, {'role': 'user', 'content': final_instruction}]
            request_tools = None
            tool_choice = 'none'
        else:
            request_messages = messages
            request_tools = tool_schemas
            tool_choice = 'auto'
        format_recovery_attempts: list[dict[str, Any]] = []
        model_request_retry_attempts: list[dict[str, Any]] = []
        accepted_retry_instruction: str | None = None
        accepted_adaptation: dict[str, Any] | None = None
        output: dict[str, Any] = {'role': 'assistant', 'content': ''}
        provider_metadata = None
        parsed = parse_neos_assistant_message(output, tool_schema_version=tool_schema_version)
        model_latency_ms = 0.0
        parse_latency_ms = 0.0
        previous_error: str | None = None
        accepted_request_message_count = len(request_messages)
        maximum_attempts = 1 + (MAX_FORMAT_RECOVERY_RETRIES if adapter_enabled else 0)
        for format_attempt in range(maximum_attempts):
            attempt_messages = request_messages
            attempt_tool_choice = 'required' if format_attempt and request_tools is not None else tool_choice
            correction_message = None
            if format_attempt:
                correction_message = {'role': 'user', 'content': build_format_retry_instruction(previous_error, forced_final=forced_final)}
                attempt_messages = [*request_messages, _rejected_message_for_retry(output), correction_message]
                format_recovery_model_calls += 1
            accepted_request_message_count = len(attempt_messages)
            attempt_model_latency_ms = 0.0
            for request_attempt in range(1, MAX_TURN_MODEL_REQUEST_RETRIES + 1):
                model_started = time.perf_counter()
                try:
                    output = model.chat(attempt_messages, max_tokens=max_generation_tokens, temperature=0.0 if format_attempt else temperature, top_p=top_p, tools=request_tools, tool_choice=attempt_tool_choice)
                    request_latency_ms = (time.perf_counter() - model_started) * 1000
                    attempt_model_latency_ms += request_latency_ms
                    if request_attempt > 1:
                        model_request_retry_attempts.append({'request_attempt': request_attempt, 'format_attempt': format_attempt + 1, 'ok': True, 'latency_ms': round(request_latency_ms, 3)})
                    break
                except NeosModelRequestError as exc:
                    request_latency_ms = (time.perf_counter() - model_started) * 1000
                    attempt_model_latency_ms += request_latency_ms
                    model_request_retry_count += 1
                    model_request_retry_errors[type(exc.cause).__name__] += 1
                    model_request_retry_attempts.append({'request_attempt': request_attempt, 'format_attempt': format_attempt + 1, 'ok': False, 'backend': exc.backend, 'backend_attempts': exc.attempts, 'retryable': exc.retryable, 'error': f'{type(exc.cause).__name__}: {exc.cause}', 'latency_ms': round(request_latency_ms, 3)})
                    if not exc.retryable or request_attempt >= MAX_TURN_MODEL_REQUEST_RETRIES:
                        if exc.retryable:
                            timing_totals['model_inference_ms'] += attempt_model_latency_ms
                            save_turn_checkpoint(turn_index)
                        raise
                    delay = float(min(2 ** (request_attempt - 1), 30))
                    logger.warning('Neos retrying failed model request in-place turn=%d request_attempt=%d/%d backend=%s after: %s; sleep %.1fs', turn, request_attempt, MAX_TURN_MODEL_REQUEST_RETRIES, exc.backend, exc.cause, delay)
                    time.sleep(delay)
            else:
                raise AssertionError('model request retry loop exhausted')
            model_latency_ms += attempt_model_latency_ms
            timing_totals['model_inference_ms'] += attempt_model_latency_ms
            model_calls += 1
            parse_started = time.perf_counter()
            take_metadata = getattr(model, 'take_last_response_metadata', None)
            provider_metadata = take_metadata() if callable(take_metadata) else None
            _usage_from_provider(provider_metadata, model_usage)
            strict_parsed = parse_neos_assistant_message(output, tool_schema_version=tool_schema_version)
            if strict_parsed.valid:
                protocol_metrics['native_reasoning_parse_success'] += 1
                if strict_parsed.action_kind == 'tool_call':
                    protocol_metrics['native_tool_parse_success'] += 1
                    protocol_metrics['arguments_json_parse_success'] += 1
                    protocol_metrics['schema_validation_success'] += 1
                elif strict_parsed.action_kind == 'answer':
                    protocol_metrics['final_answer_parse_success'] += 1
            else:
                protocol_metrics['strict_parse_failed'] += 1
            parsed = strict_parsed
            if adapter_enabled and (not strict_parsed.valid):
                adapter = adapt_qwen35_native_assistant_message
                parsed = adapter(output, strict_parsed, question=question, available_image_refs=(f'img_{index}' for index in range(1, len(registry) + 1)), forced_final=forced_final)
            attempt_parse_latency_ms = (time.perf_counter() - parse_started) * 1000
            parse_latency_ms += attempt_parse_latency_ms
            timing_totals['assistant_parse_ms'] += attempt_parse_latency_ms
            if parsed.valid:
                accepted_retry_instruction = correction_message['content'] if correction_message else None
                accepted_adaptation = parsed.adaptation
                if format_attempt and accepted_adaptation is None:
                    accepted_adaptation = {'profile': assistant_format_profile, 'adapter_version': adapter_version, 'original_error': format_recovery_attempts[0]['format_error'], 'transforms': ['model_retry_after_format_error'], 'retry_attempts': format_attempt}
                break
            previous_error = parsed.error or strict_parsed.error
            format_recovery_attempts.append({'attempt': format_attempt + 1, 'format_error': previous_error, 'assistant_message': output, 'provider_metadata': provider_metadata, 'model_latency_ms': round(attempt_model_latency_ms, 3), 'assistant_parse_latency_ms': round(attempt_parse_latency_ms, 3)})
        if accepted_adaptation is not None:
            format_adapted_turns += 1
            protocol_metrics['compat_recovery_count'] += 1
        if accepted_adaptation is not None:
            for transform in accepted_adaptation.get('transforms') or []:
                format_transform_counts[str(transform)] += 1
        normalized_message = parsed.message or (output if isinstance(output, dict) else {'role': 'assistant', 'content': ''})
        assistant_record: dict[str, Any] = {'role': 'assistant', 'turn': turn, 'content': normalized_message.get('content', ''), 'reasoning_content': normalized_message.get('reasoning_content', ''), 'tool_calls': normalized_message.get('tool_calls', []), 'message': normalized_message, 'model_latency_ms': round(model_latency_ms, 3), 'assistant_parse_latency_ms': round(parse_latency_ms, 3), 'request_message_count': accepted_request_message_count, 'context_image_count': len(registry), 'response_chars': len(json.dumps(output, ensure_ascii=False, default=str)), 'strict_parse_valid': strict_parsed.valid, 'strict_parse_failed': not strict_parsed.valid, 'compat_used': accepted_adaptation is not None, 'compat_strategy': list((accepted_adaptation or {}).get('transforms') or []) if accepted_adaptation is not None else [], 'repair_attempts': max(0, len(format_recovery_attempts) - (0 if parsed.valid else 1)), 'repair_success': bool(parsed.valid and max(0, len(format_recovery_attempts) - (0 if parsed.valid else 1))), 'argument_normalizations': list(parsed.argument_normalizations or []), 'request_id': (provider_metadata or {}).get('response', {}).get('request_id') or (provider_metadata or {}).get('response', {}).get('id'), 'raw_completion': None, 'raw_completion_available': False, 'parsed_reasoning_content': normalized_message.get('reasoning_content', ''), 'parsed_content': normalized_message.get('content', ''), 'parsed_tool_calls': normalized_message.get('tool_calls', [])}
        protocol_metrics['argument_normalization_count'] += len(parsed.argument_normalizations or [])
        if accepted_adaptation is not None:
            assistant_record['format_adaptation'] = accepted_adaptation
            assistant_record['raw_assistant_message'] = output
        if format_recovery_attempts:
            assistant_record['format_recovery_attempts'] = format_recovery_attempts
        if model_request_retry_attempts:
            assistant_record['model_request_retry_attempts'] = model_request_retry_attempts
        if accepted_retry_instruction is not None:
            assistant_record['format_recovery_instruction'] = accepted_retry_instruction
        if forced_final:
            assistant_record['forced_finalization'] = True
            assistant_record['finalization_instruction'] = final_instruction
        if provider_metadata is not None:
            assistant_record['provider_metadata'] = provider_metadata
        if not parsed.valid:
            assistant_record['format_error'] = parsed.error
            trajectory.append(assistant_record)
            termination_reason = 'forced_finalization_failed' if forced_final else 'invalid_assistant_format'
            protocol_metrics['protocol_failure_count'] += 1
            break
        if parsed.action_kind == 'answer':
            final_answer = parsed.answer
            forced_finalization_succeeded = forced_final
            trajectory.append(assistant_record)
            termination_reason = 'final_answer'
            break
        if forced_final:
            assistant_record['format_error'] = 'tool use is disabled on the final turn'
            trajectory.append(assistant_record)
            termination_reason = 'forced_finalization_failed'
            break
        call = dict(parsed.tool_call or {})
        call_id = call['id']
        name = call['name']
        arguments = call['arguments']
        assistant_record['tool_call'] = call
        trajectory.append(assistant_record)
        tool_counts[name] += 1
        tool_request = AgentToolRequest(turn=turn, name=name, arguments=arguments, runtime=runtime, registry=tuple(registry), source_urls=tuple(source_registry))
        tool_result = (yield tool_request)
        if tool_result is None:
            tool_result = tool_request.execute()
        if not isinstance(tool_result, AgentToolResult):
            raise TypeError(f'Agent tool event must resume with AgentToolResult, got {type(tool_result)!r}')
        execution = tool_result.execution
        execution_error = tool_result.error
        runtime_type_validation_success = bool((execution.metadata or {}).get('runtime_type_validation_success')) and execution_error is None
        tool_execution_success = execution_error is None and (not bool((execution.metadata or {}).get('schema_validation_error'))) and (not str(execution.text or '').startswith('Error:'))
        protocol_metrics['runtime_type_validation_success'] += int(runtime_type_validation_success)
        protocol_metrics['tool_execution_success'] += int(tool_execution_success)
        latency_ms = tool_result.latency_ms
        timing_totals['tool_execution_ms'] += latency_ms
        postprocess_started = time.perf_counter()
        assigned_refs = [f'img_{len(registry) + index}' for index in range(1, len(execution.images) + 1)]
        observation = execution.text
        verify_execution_output_refs(name, execution.metadata, assigned_refs, registry_size=len(registry))
        full_observation = observation
        observation = _truncate(full_observation)
        output_sources = list(execution.metadata.get('image_source_urls') or [])
        output_sources.extend([None] * (len(execution.images) - len(output_sources)))
        output_kinds = list(execution.metadata.get('image_kinds') or [])
        output_kinds.extend([name] * (len(execution.images) - len(output_kinds)))
        image_artifacts = []
        for ref, image, source_url, kind in zip(assigned_refs, execution.images, output_sources, output_kinds):
            if artifact_store is not None:
                artifact = artifact_store.save_image(image, image_ref=ref, kind=str(kind), source_url=source_url)
            else:
                artifact = describe_unpersisted_image(image, image_ref=ref, kind=str(kind))
                if source_url:
                    artifact['source_url'] = source_url
            image_artifacts.append(artifact)
        context_images = prepare_context_images(execution.images, codec_overrides)
        registry.extend(context_images)
        source_registry.extend(output_sources[:len(execution.images)])
        tool_record = {'role': 'tool', 'turn': turn, 'tool_call_id': call_id, 'name': name, 'arguments': arguments, 'content': observation, 'raw_observation_chars': len(full_observation), 'observation_truncated': observation != full_observation, 'latency_ms': latency_ms, 'metadata': execution.metadata, 'argument_normalizations': list(parsed.argument_normalizations or []), 'runtime_argument_types': (execution.metadata or {}).get('runtime_argument_types'), 'runtime_type_validation_success': runtime_type_validation_success, 'tool_execution_success': tool_execution_success, 'images': image_artifacts}
        if execution_error:
            tool_record['error'] = execution_error
        trajectory.append(tool_record)
        messages.append(normalized_message)
        if context_images:
            tool_content: Any = [*(_pil_content(image, codec_overrides) for image in context_images), {'type': 'text', 'text': observation}]
        else:
            tool_content = observation
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name, 'content': tool_content})
        postprocess_latency_ms = (time.perf_counter() - postprocess_started) * 1000
        timing_totals['tool_postprocess_ms'] += postprocess_latency_ms
        tool_record['postprocess_latency_ms'] = round(postprocess_latency_ms, 3)
        scheduler_yield_started = time.perf_counter()
        yield AgentTurnComplete(turn=turn, next_turn=turn + 1, tool_name=name, model_calls=model_calls, tool_calls_executed=sum(tool_counts.values()))
        timing_totals['turn_scheduler_wait_ms'] += (time.perf_counter() - scheduler_yield_started) * 1000
    agent_wall_ms = prior_agent_wall_ms + (time.perf_counter() - agent_started) * 1000
    measured_ms = sum(timing_totals.values())
    timing = {'schema_version': TIMING_SCHEMA_VERSION, **{key: round(value, 3) for key, value in sorted(timing_totals.items())}, 'agent_wall_ms': round(agent_wall_ms, 3), 'unattributed_agent_ms': round(max(0.0, agent_wall_ms - measured_ms), 3)}
    result = {'trace_schema_version': TRACE_SCHEMA_VERSION, 'toolset_profile': NEOS_TOOLSET_PROFILE, 'available_tools': sorted(NEOS_TOOL_NAMES), 'protocol_version': NEOS_PROTOCOL_VERSION, 'tool_schema_version': tool_schema_version, 'visual_result_schema_version': VISUAL_RESULT_SCHEMA_VERSION, 'context_image_codec': codec_config(codec_overrides), 'assistant_format_profile': assistant_format_profile, 'assistant_format_adapter_contract_version': assistant_format_adapter_contract_version(assistant_format_profile), 'assistant_format_adapter_version': adapter_version if adapter_enabled else None, 'format_adapted_turns': format_adapted_turns, 'format_recovery_model_calls': format_recovery_model_calls, 'model_request_retry_count': model_request_retry_count, 'model_request_retry_errors': dict(sorted(model_request_retry_errors.items())), 'turn_checkpoint_resume_count': turn_checkpoint_resume_count, 'resumed_from_turn_checkpoint': bool(turn_checkpoint_resume_count), 'protocol_metrics': {**{key: int(protocol_metrics.get(key, 0)) for key in ('native_reasoning_parse_success', 'native_tool_parse_success', 'arguments_json_parse_success', 'schema_validation_success', 'argument_normalization_count', 'runtime_type_validation_success', 'tool_execution_success', 'final_answer_parse_success', 'strict_parse_failed', 'compat_recovery_count', 'protocol_failure_count')}, 'corrective_retry_count': format_recovery_model_calls}, 'format_adapter_transform_counts': dict(sorted(format_transform_counts.items())), 'question': question, 'final_answer': final_answer, 'termination_reason': termination_reason, 'n_turns': sum((item['role'] == 'assistant' for item in trajectory)), 'model_calls': model_calls, 'model_usage': dict(sorted(model_usage.items())), 'tool_call_counts': dict(sorted(tool_counts.items())), 'tool_calls_executed': sum(tool_counts.values()), 'forced_finalization': forced_finalization, 'forced_finalization_succeeded': forced_finalization_succeeded, 'n_input_images': input_image_count, 'n_images': len(registry), 'input_images': input_artifacts, 'timing': timing, 'trajectory': trajectory, 'raw_completion_observability': {'available': False, 'reason': 'current SGLang Chat Completions response is already parsed; no same-sample pre-parser hook is installed'}}
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    return result

def run_agent(**kwargs) -> dict[str, Any]:
    pass
    trajectory = iter_agent_turns(**kwargs)
    resume_value = None
    while True:
        try:
            event = trajectory.send(resume_value)
        except StopIteration as completed:
            return completed.value
        resume_value = event.execute() if isinstance(event, AgentToolRequest) else None
