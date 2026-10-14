from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Optional
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import SPEC_DECODE_EXTRA_KEYS, AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.workers.rollout.replica import TokenOutput
from neos.workflow.image_codec import prepare_context_images
from neos.workflow.native_adapter import adapt_qwen35_native_assistant_message, build_format_retry_instruction
from neos.workflow.protocol import MAX_ASSISTANT_TURNS, MAX_ROLLOUT_RESPONSE_TOKENS, MAX_TOOL_RESPONSE_LENGTH, NEOS_SYSTEM_PROMPT, QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, build_final_turn_instruction, parse_neos_assistant_message
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_TOOL_NAMES
from neos.workflow.turn_credit import encode_rollout_visuals
logger = logging.getLogger(__name__)
_THINK_RE = re.compile('^\\s*<think>(.*?)</think>\\s*(.*)$', re.DOTALL | re.IGNORECASE)
_SPECIAL_TAIL_RE = re.compile('(?:<\\|im_end\\|>|<\\|endoftext\\|>)+\\s*$')

def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    item = getattr(value, 'item', None)
    if callable(item):
        unwrapped = item()
        if isinstance(unwrapped, dict):
            return unwrapped
    return {}

def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts = []
    for item in value:
        if not isinstance(item, dict) or item.get('type') != 'text':
            return None
        parts.append(str(item.get('text') or ''))
    return '\n'.join(parts)
RL_ASSISTANT_PARSE_VERSION = 'qwen35-prefilled-think-visible-action'
COMPONENT_SPAN_VERSION = 'qwen35-generated-cot-act-token-spans'

def _split_reasoning(text: str, *, thinking_prefix_open: bool=False) -> tuple[str, str]:
    pass
    cleaned = _SPECIAL_TAIL_RE.sub('', str(text or '')).strip()
    explicit_open = cleaned.startswith('<think>')
    if explicit_open:
        cleaned = cleaned[len('<think>'):]
    if '</think>' in cleaned:
        reasoning, visible = cleaned.split('</think>', 1)
        return (reasoning.strip(), visible.strip())
    if thinking_prefix_open or explicit_open:
        return (cleaned.strip(), '')
    return ('', cleaned)

def _response_component_spans(token_ids: list[int], *, offset: int, thinking_prefix_open: bool, think_start_token_id: int, think_end_token_id: int) -> dict[str, dict[str, int]]:
    pass
    try:
        boundary = token_ids.index(think_end_token_id) + 1
    except ValueError:
        explicit_open = bool(token_ids and token_ids[0] == think_start_token_id)
        boundary = len(token_ids) if thinking_prefix_open or explicit_open else 0
    return {'cot': {'start': offset, 'end': offset + boundary}, 'act': {'start': offset + boundary, 'end': offset + len(token_ids)}}

def _merge_policy_step_bounds(trajectory_fields: dict[str, Any], generation_fields: dict[str, Any]) -> None:
    pass
    step = generation_fields.get('global_steps')
    if step is None:
        step = generation_fields.get('max_global_steps')
    if step is None:
        return
    step = int(step)
    current_min = trajectory_fields.get('min_global_steps')
    current_max = trajectory_fields.get('max_global_steps')
    trajectory_fields['min_global_steps'] = step if current_min is None else min(int(current_min), step)
    trajectory_fields['max_global_steps'] = step if current_max is None else max(int(current_max), step)

def _enabled(value: str | None) -> bool:
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}

@register('neos_agent')
class NeosAgentLoop(ToolAgentLoop):
    pass

    def __init__(self, *args, tools=None, **kwargs):
        super().__init__(*args, tools=tools, **kwargs)
        if self.tool_parser_name != 'qwen3_coder':
            raise ValueError('Neos Qwen3.5 RL requires qwen3_coder')
        if self.max_parallel_calls != 1:
            raise ValueError('Neos permits exactly one tool call per turn')
        if self.max_tool_response_length != MAX_TOOL_RESPONSE_LENGTH:
            raise ValueError('Neos tool-observation length mismatch')
        if set(self.tools) != set(NEOS_TOOL_NAMES):
            raise ValueError(f'Neos tool set mismatch: {sorted(self.tools)} != {sorted(NEOS_TOOL_NAMES)}')
        self.max_format_recovery_retries = int(os.getenv('NEOS_RL_FORMAT_RECOVERY_RETRIES', '2'))
        self.max_model_request_retries = int(os.getenv('NEOS_RL_MODEL_REQUEST_RETRIES', '8'))
        self.max_generation_tokens = int(os.getenv('NEOS_RL_MAX_GENERATION_TOKENS', str(MAX_ROLLOUT_RESPONSE_TOKENS)))
        if self.max_format_recovery_retries < 0 or self.max_model_request_retries <= 0 or self.max_generation_tokens <= 0:
            raise ValueError('invalid Neos RL retry configuration')
        self.turn_stop_token_ids = set(self.tool_parser.stop_token_ids or [])
        for token in ('<|im_end|>', '<|endoftext|>'):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                self.turn_stop_token_ids.add(token_id)
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if isinstance(eos_token_id, int) and eos_token_id >= 0:
            self.turn_stop_token_ids.add(eos_token_id)
        if not self.turn_stop_token_ids:
            raise ValueError('could not resolve a Qwen3.5 assistant-turn stop token')
        self.tool_call_start_token_id = self.tokenizer.convert_tokens_to_ids('<tool_call>')
        if not isinstance(self.tool_call_start_token_id, int):
            raise ValueError('could not resolve Qwen3.5 <tool_call> token')
        self.think_start_token_id = self.tokenizer.convert_tokens_to_ids('<think>')
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids('</think>')
        if not all((isinstance(value, int) and value >= 0 for value in (self.think_start_token_id, self.think_end_token_id))):
            raise ValueError('could not resolve Qwen3.5 reasoning boundary tokens')

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = [dict(message) for message in list(kwargs['raw_prompt'])]
        if not messages or messages[0].get('role') != 'system' or _text_content(messages[0].get('content')) != NEOS_SYSTEM_PROMPT:
            raise ValueError('RL sample does not contain the exact Neos system prompt')
        if os.getenv('NEOS_COEV_ROOT'):
            from neos.workflow.coevolution import rollout_system_prompt
            messages[0]['content'] = rollout_system_prompt(self.max_assistant_turns)
        extra_info = _mapping(kwargs.get('extra_info'))
        question = str(extra_info.get('question') or '').strip()
        if not question:
            raise ValueError('RL sample has no question in extra_info')
        turn_credit_capture = bool(_enabled(os.getenv('NEOS_TURN_CREDIT_ENABLED', '0')) and (extra_info.get('turn_credit_eligible') or os.getenv('NEOS_COEV_ROOT')) and (not bool(kwargs.get('validate', False))))
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = prepare_context_images(multi_modal_data.get('images'))
        videos = multi_modal_data.get('videos')
        audios = multi_modal_data.get('audios')
        metrics: dict[str, Any] = {}
        agent_data = AgentData(messages=messages, image_data=images, video_data=videos, audio_data=audios, mm_processor_kwargs=self._get_mm_processor_kwargs(audios), metrics=metrics, request_id=uuid4().hex, tools_kwargs=kwargs.get('tools_kwargs', {}))
        tool_selection = extra_info.get('tool_selection')
        if tool_selection:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
        else:
            selected = self.tools
        if set(selected) != set(NEOS_TOOL_NAMES):
            raise ValueError('per-sample tool selection changed the Neos tool set')
        agent_data._active_tools = selected
        agent_data._active_tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in selected.values()]
        initial_credit_trace: list[dict[str, Any]] = []
        if turn_credit_capture:
            for message in messages:
                role = str(message.get('role') or '')
                if role not in {'system', 'user'}:
                    continue
                initial_credit_trace.append({'role': role, 'content': message.get('content'), 'image_refs': [f'img_{index}' for index in range(1, len(images or []) + 1)] if role == 'user' else []})
        agent_data.extra_fields.update({'question': question, 'qid': str(extra_info.get('qid') or ''), 'ground_truth': str(extra_info.get('answer') or ''), 'final_answer': '', 'termination_reason': 'running', 'assistant_format_profile': QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, 'assistant_parse_version': RL_ASSISTANT_PARSE_VERSION, 'tool_schema_version': DEFAULT_NEOS_SCHEMA_VERSION, 'agent_trace': [], 'model_calls': 0, 'format_recovery_model_calls': 0, 'forced_finalization': False, 'min_global_steps': None, 'max_global_steps': None, 'rollout_request_id': agent_data.request_id, 'rollout_session_id': int(kwargs.get('session_id', -1)), 'turn_credit_eligible': bool(extra_info.get('turn_credit_eligible', False)), 'reference_trajectory_path': str(extra_info.get('reference_trajectory_path') or ''), 'policy_turn_spans': [], 'turn_credit_trace': initial_credit_trace, 'turn_credit_capture': turn_credit_capture})
        if os.getenv('NEOS_COEV_ROOT'):
            from neos.workflow.coevolution import rollout_context
            agent_data.extra_fields.update(rollout_context())
            agent_data.extra_fields['coev_system_prompt'] = messages[0]['content']
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                raise RuntimeError(f'invalid Neos AgentState: {state}')
        if agent_data.extra_fields['termination_reason'] == 'running':
            agent_data.extra_fields['termination_reason'] = 'response_length'
        if turn_credit_capture:
            agent_data.extra_fields['_turn_credit_visuals'] = await self.loop.run_in_executor(None, encode_rollout_visuals, agent_data.image_data)
        response_count = len(agent_data.response_mask)
        if os.getenv('NEOS_COEV_ROOT'):
            from neos.workflow.coevolution import save_rollout
            await self.loop.run_in_executor(None, save_rollout, agent_data)
        response_ids = agent_data.prompt_ids[-response_count:] if response_count else []
        prompt_ids = agent_data.prompt_ids[:-response_count] if response_count else agent_data.prompt_ids
        mm_data = {}
        if agent_data.image_data is not None:
            mm_data['images'] = agent_data.image_data
        if agent_data.video_data is not None:
            mm_data['videos'] = agent_data.video_data
        if agent_data.audio_data is not None:
            mm_data['audios'] = agent_data.audio_data
        return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids[:self.response_length], response_mask=agent_data.response_mask[:self.response_length], multi_modal_data=mm_data, mm_processor_kwargs=agent_data.mm_processor_kwargs, response_logprobs=agent_data.response_logprobs[:self.response_length] if agent_data.response_logprobs else None, num_turns=agent_data.assistant_turns, metrics=agent_data.metrics, extra_fields=agent_data.extra_fields)

    async def _decode_native_message(self, token_ids: list[int], *, agent_data: AgentData, forced_final: bool, format_attempt: int, thinking_prefix_open: bool=True):
        active_tools = getattr(agent_data, '_active_tools', self.tools)
        parser_tools = [] if forced_final else [tool.tool_schema for tool in active_tools.values()]
        raw_text = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=False))
        reasoning, visible = _split_reasoning(raw_text, thinking_prefix_open=thinking_prefix_open)
        ranges = _response_component_spans(token_ids, offset=0, thinking_prefix_open=thinking_prefix_open, think_start_token_id=self.think_start_token_id, think_end_token_id=self.think_end_token_id)
        visible_ids = token_ids[ranges['act']['start']:]
        if visible_ids:
            assistant_content, calls = await self.tool_parser.extract_tool_calls(visible_ids, parser_tools)
        else:
            assistant_content, calls = ('', [])
        if calls:
            message: dict[str, Any] = {'role': 'assistant', 'content': _SPECIAL_TAIL_RE.sub('', assistant_content).strip(), 'tool_calls': []}
            for index, call in enumerate(calls):
                message['tool_calls'].append({'id': f'call_rl_{agent_data.request_id[:10]}_{agent_data.assistant_turns}_{format_attempt}_{index}', 'type': 'function', 'function': {'name': call.name, 'arguments': call.arguments}})
        else:
            message = {'role': 'assistant', 'content': visible}
        if reasoning:
            message['reasoning_content'] = reasoning
        strict = parse_neos_assistant_message(message, tool_schema_version=DEFAULT_NEOS_SCHEMA_VERSION)
        parsed = strict
        if not strict.valid:
            parsed = adapt_qwen35_native_assistant_message(message, strict, question=agent_data.extra_fields['question'], available_image_refs=(f'img_{index}' for index in range(1, len(agent_data.image_data or []) + 1)), forced_final=forced_final)
        return (parsed, strict, raw_text)

    async def _append_retry_instruction(self, agent_data: AgentData, instruction: str) -> bool:
        message = {'role': 'user', 'content': instruction}
        response_ids = await self.apply_chat_template([message], images=None, videos=None, audios=None, remove_system_prompt=True)
        response_ids = self.turn_separator + response_ids
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return False
        agent_data.messages.append(message)
        if agent_data.extra_fields.get('turn_credit_capture'):
            agent_data.extra_fields['turn_credit_trace'].append({'role': 'environment_instruction', 'after_turn': agent_data.assistant_turns, 'content': instruction})
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return True

    async def _generate_once(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> TokenOutput:
        last_error: Exception | None = None
        for request_attempt in range(1, self.max_model_request_retries + 1):
            started = time.perf_counter()
            try:
                output = await self.server_manager.generate(request_id=agent_data.request_id, prompt_ids=agent_data.prompt_ids, sampling_params=sampling_params, image_data=agent_data.image_data, video_data=agent_data.video_data, audio_data=agent_data.audio_data, mm_processor_kwargs=agent_data.mm_processor_kwargs)
                agent_data.metrics['generate_sequences'] = float(agent_data.metrics.get('generate_sequences') or 0.0) + time.perf_counter() - started
                return output
            except Exception as exc:
                last_error = exc
                if request_attempt >= self.max_model_request_retries:
                    raise
                await asyncio.sleep(min(2 ** (request_attempt - 1), 30))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _merge_generation(agent_data: AgentData, output: TokenOutput) -> None:
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += output.token_ids
        agent_data.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        elif agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(output.token_ids)
        if agent_data.metrics.get('num_preempted') is None:
            agent_data.metrics['num_preempted'] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics['num_preempted'] += output.num_preempted or 0
        _merge_policy_step_bounds(agent_data.extra_fields, output.extra_fields)
        for key, value in output.extra_fields.items():
            if key in SPEC_DECODE_EXTRA_KEYS and key in agent_data.extra_fields:
                agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(value)

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool=False) -> AgentState:
        del ignore_termination
        turn = agent_data.assistant_turns + 1
        forced_final = bool(self.max_assistant_turns and turn >= self.max_assistant_turns)
        if forced_final:
            agent_data.extra_fields['forced_finalization'] = True
        agent_data.assistant_turns += 1
        active_tools = getattr(agent_data, '_active_tools', self.tools)
        parser_tools = [tool.tool_schema for tool in active_tools.values()]
        params = dict(sampling_params)
        params['stop_token_ids'] = sorted(set(params.get('stop_token_ids') or []) | self.turn_stop_token_ids)
        params['max_new_tokens'] = min(int(params.get('max_new_tokens') or self.max_generation_tokens), self.max_generation_tokens)
        if forced_final:
            logit_bias = dict(params.get('logit_bias') or {})
            logit_bias[str(self.tool_call_start_token_id)] = -100.0
            params['logit_bias'] = logit_bias
        last_error = None
        for format_attempt in range(self.max_format_recovery_retries + 1):
            if len(agent_data.response_mask) >= self.response_length:
                agent_data.extra_fields['termination_reason'] = 'response_length'
                return AgentState.TERMINATED
            attempt_params = dict(params)
            if format_attempt:
                attempt_params['temperature'] = 0.0
            span_start = len(agent_data.response_mask)
            prefix_tail = self.tokenizer.decode(agent_data.prompt_ids[-16:], skip_special_tokens=False)
            thinking_prefix_open = prefix_tail.rstrip().endswith('<think>')
            output = await self._generate_once(agent_data, attempt_params)
            self._merge_generation(agent_data, output)
            span_end = len(agent_data.response_mask)
            agent_data.extra_fields['model_calls'] += 1
            parsed, strict, raw_text = await self._decode_native_message(output.token_ids, agent_data=agent_data, forced_final=forced_final, format_attempt=format_attempt, thinking_prefix_open=thinking_prefix_open)
            agent_data.extra_fields['agent_trace'].append({'turn': turn, 'format_attempt': format_attempt + 1, 'forced_final': forced_final, 'strict_valid': strict.valid, 'accepted': parsed.valid, 'action_kind': parsed.action_kind, 'error': parsed.error, 'raw_response_chars': len(raw_text)})
            span_id = f'T{turn:02d}.A{format_attempt + 1}'
            span_record = {'span_id': span_id, 'turn': turn, 'format_attempt': format_attempt + 1, 'start': span_start, 'end': span_end, 'forced_final': forced_final, 'strict_valid': strict.valid, 'accepted': parsed.valid, 'action_kind': parsed.action_kind, 'component_span_version': COMPONENT_SPAN_VERSION, 'component_spans': _response_component_spans(output.token_ids, offset=span_start, thinking_prefix_open=thinking_prefix_open, think_start_token_id=self.think_start_token_id, think_end_token_id=self.think_end_token_id)}
            agent_data.extra_fields['policy_turn_spans'].append(span_record)
            if agent_data.extra_fields.get('turn_credit_capture'):
                trace_record: dict[str, Any] = {'role': 'assistant', **span_record, 'reasoning': str(parsed.reasoning or ''), 'parse_error': parsed.error}
                if parsed.action_kind == 'tool_call':
                    trace_record['tool_call'] = parsed.tool_call
                elif parsed.action_kind == 'answer':
                    trace_record['answer'] = str(parsed.answer or '')
                else:
                    trace_record['raw_response'] = raw_text
                agent_data.extra_fields['turn_credit_trace'].append(trace_record)
            if parsed.valid:
                if parsed.action_kind == 'answer':
                    agent_data.extra_fields['final_answer'] = str(parsed.answer or '')
                    agent_data.extra_fields['termination_reason'] = 'final_answer'
                    return AgentState.TERMINATED
                if not forced_final:
                    call = parsed.tool_call or {}
                    agent_data.tool_calls = [FunctionCall(name=str(call['name']), arguments=json.dumps(call['arguments'], ensure_ascii=False, separators=(',', ':')), tool_call_id=str(call['id']))]
                    return AgentState.PROCESSING_TOOLS
                last_error = 'The reserved final turn attempted a tool call even though tools are disabled. Return the concise final answer directly.'
            else:
                last_error = parsed.error
            if format_attempt >= self.max_format_recovery_retries:
                break
            instruction = build_format_retry_instruction(last_error, forced_final=forced_final)
            if not await self._append_retry_instruction(agent_data, instruction):
                break
            agent_data.extra_fields['format_recovery_model_calls'] += 1
        agent_data.extra_fields['termination_reason'] = 'forced_finalization_failed' if forced_final else 'invalid_assistant_format'
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        pass
        trace = agent_data.extra_fields.get('turn_credit_trace', [])
        trace_start = len(trace)
        state = await super()._handle_processing_tools_state(agent_data)
        delivered = state == AgentState.GENERATING
        for event in trace[trace_start:]:
            if event.get('role') == 'tool':
                event['observation_delivered_to_policy'] = delivered
                if not delivered:
                    event['observation_omitted_reason'] = 'tool_response_not_inserted_before_termination'
        return state

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData):
        from neos.workflow.coevolution import mei_call_context
        with mei_call_context(agent_data.extra_fields):
            response, reward, metadata = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        agent_data.extra_fields['agent_trace'].append({'turn': agent_data.assistant_turns, 'role': 'tool', 'name': tool_call.name, 'arguments': tool_call.arguments, 'observation_chars': len(str(response.text or '')), 'image_count': len(response.image) if isinstance(response.image, list) else int(response.image is not None), 'metadata': metadata})
        if agent_data.extra_fields.get('turn_credit_capture'):
            agent_data.extra_fields['turn_credit_trace'].append({'role': 'tool', 'turn': agent_data.assistant_turns, 'name': tool_call.name, 'arguments': tool_call.arguments, 'content': str(response.text or ''), 'output_refs': list((metadata or {}).get('output_refs') or []), 'reused_refs': list((metadata or {}).get('reused_refs') or []), 'image_source_urls': list((metadata or {}).get('image_source_urls') or []), 'schema_validation_error': bool((metadata or {}).get('schema_validation_error'))})
        if self.max_assistant_turns and agent_data.assistant_turns == self.max_assistant_turns - 1:
            response.text = str(response.text or '') + '\n\n[Environment instruction]\n' + build_final_turn_instruction(self.max_assistant_turns)
            if agent_data.extra_fields.get('turn_credit_capture'):
                agent_data.extra_fields['turn_credit_trace'][-1]['content'] = response.text
        return (response, reward, metadata)
