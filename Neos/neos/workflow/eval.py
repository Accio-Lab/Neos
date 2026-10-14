from __future__ import annotations
from neos.identifiers import path_namespace, file_key
import argparse
import json
import logging
import os
import re
import stat
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from neos.eval import secrets
from neos.configuration import required_setting, evaluation_output_path
from neos.eval.answer_judge import judge_final_answer
from neos.eval.datasets import load_bcvl, load_browsecomp_v3, load_fvqa, load_infoseek, load_livevqa, load_mmbrowsecomp, load_mmsearch, load_mmsearch_plus, load_simplevqa, load_vdr, load_visbrowse
from neos.eval.benchmarks import BENCHMARK_BY_ID, SELECTION_GROUPS, canonical_benchmark_selection, expand_benchmark_selection
from neos.eval.gateway import FatalLLMGatewayError, LLMGateway, RetryExhaustedLLMGatewayError, SearchGateway
from neos.eval.durable_jsonl import DurableJSONLPublisher, repair_truncated_jsonl_tail
from neos.eval.models import ModelContextLengthExceededError
from neos.eval.trace_artifacts import EvaluationArtifactStore
from neos.eval.text_search_data import TEXT_SEARCH_LOADERS
from neos.workflow.agent import AgentToolRequest, AgentToolResult, AgentTurnComplete, iter_agent_turns, run_agent
from neos.workflow.mei import NeosMEI
from neos.workflow.image_codec import codec_config
from neos.workflow.models import load_model_backend
from neos.workflow.protocol import ASSISTANT_FORMAT_PROFILES, MAX_ASSISTANT_TURNS, MAX_CONTEXT_TOKENS, MAX_ROLLOUT_RESPONSE_TOKENS, MAX_TOOL_RESPONSE_LENGTH, NEOS_PROTOCOL_VERSION, QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, assistant_format_adapter_contract_version
from neos.workflow.runtime import NeosRuntime
from neos.workflow.toolsets import NEOS_TOOLSET_PROFILE
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_SCHEMA_VERSIONS, NEOS_TOOL_NAMES
from neos.workflow.timing import build_timing_summary
logger = logging.getLogger(__name__)
_RETRYABLE_FORMAT_FAILURE_TERMINATIONS = frozenset({'invalid_assistant_format'})
_NON_RETRYABLE_SCORED_FAILURE_TERMINATIONS = frozenset({'forced_finalization_failed'})
SELECTIVE_FORMAT_FAILURE_RESAMPLING = 'selective_resample'
SCORE_FIRST_FORMAT_FAILURE_OUTCOME = 'score_first_attempt'
FORMAT_FAILURE_ROLLOUT_POLICIES = (SELECTIVE_FORMAT_FAILURE_RESAMPLING, SCORE_FIRST_FORMAT_FAILURE_OUTCOME)
_FORMAT_FAILURE_RESAMPLING_PROFILES = frozenset({QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE})
JUDGE_ONLY_TRANSPORT_RETRIES = 3
JUDGE_ONLY_RETRY_SCHEMA_VERSION = 'neos-judge-only-retry'

class JudgeOnlyRetryExhaustedError(RuntimeError):
    pass

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        rendered = attempts[-1].get('error') if attempts else 'unknown judge error'
        super().__init__(str(rendered))

def _task_repeat(value: dict) -> int:
    repeat = value.get('evaluation_repeat', 1)
    if isinstance(repeat, bool):
        raise ValueError('evaluation_repeat must be an integer')
    return int(repeat or 1)

def _task_key(value: dict) -> tuple[str, str, int]:
    pass
    return (str(value.get('bench') or ''), str(value.get('sid') or ''), _task_repeat(value))

def _is_judge_only_pending(row: dict) -> bool:
    pass
    return bool(row.get('judge_only_pending') and row.get('error') and isinstance(row.get('trajectory'), list) and str(row.get('final_answer') or '').strip())

def _judge_with_transport_retries(judge: LLMGateway, *, question: str, correct_answer: str, final_answer: str | None) -> tuple[dict, list[dict], float]:
    pass
    started = time.perf_counter()
    transport_attempts: list[dict] = []
    for attempt in range(1, JUDGE_ONLY_TRANSPORT_RETRIES + 1):
        attempt_started = time.perf_counter()
        try:
            judgement = judge_final_answer(judge, question, correct_answer, final_answer)
            transport_attempts.append({'attempt': attempt, 'ok': True, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3)})
            return (judgement, transport_attempts, (time.perf_counter() - started) * 1000)
        except FatalLLMGatewayError:
            raise
        except RetryExhaustedLLMGatewayError as exc:
            transport_attempts.append({'attempt': attempt, 'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3)})
            if attempt < JUDGE_ONLY_TRANSPORT_RETRIES:
                delay = float(2 ** (attempt - 1))
                logger.warning('Neos judge-only retry %d/%d after: %s; sleep %.1fs', attempt, JUDGE_ONLY_TRANSPORT_RETRIES, exc, delay)
                time.sleep(delay)
    raise JudgeOnlyRetryExhaustedError(transport_attempts)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 384)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 384)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def _read_jsonl(path: Path, attempts: int=30, *, repair_truncated_tail: bool=False) -> list[dict]:
    pass
    if not path.exists():
        return []
    if repair_truncated_tail:
        repair = repair_truncated_jsonl_tail(path)
        if repair.get('repaired'):
            logger.warning('discarded %d-byte truncated JSONL tail from %s; local quarantine=%s', int(repair.get('discarded_bytes') or 0), path, repair.get('quarantine_path'))
    try:
        import orjson
        parse_line = orjson.loads
    except ImportError:
        parse_line = json.loads
    last_error: ValueError | OSError | None = None
    for attempt in range(attempts):
        try:
            snapshot = path.read_bytes()
            return [parse_line(line) for line in snapshot.splitlines() if line.strip()]
        except (ValueError, OSError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0)
    assert last_error is not None
    raise last_error

def _historical_worker_ms(row: dict) -> float:
    evaluation = row.get('evaluation_timing') or {}
    value = float(evaluation.get('sample_worker_wall_ms', 0.0) or 0.0)
    if value > 0:
        return value
    timing = row.get('timing') or {}
    value = float(timing.get('agent_wall_ms', 0.0) or 0.0)
    if value > 0:
        return value
    return float(row.get('n_turns', 0) or 0) * 1000.0

def _load_scheduling_hints(path: Path) -> dict[tuple[str, str], float]:
    hints: dict[tuple[str, str], float] = {}
    try:
        import orjson
        parse_line = orjson.loads
    except ImportError:
        parse_line = json.loads
    with path.open('rb') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = parse_line(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid scheduling hint JSON at {path}:{line_number}') from exc
            key = (str(row.get('bench') or ''), str(row.get('sid') or ''))
            score = _historical_worker_ms(row)
            if all(key) and score > hints.get(key, 0.0):
                hints[key] = score
    return hints

def _longest_predicted_first(samples: list[dict], hints: dict[tuple[str, str], float]) -> tuple[list[dict], int]:
    indexed = list(enumerate(samples))
    matched = sum(((str(sample.get('bench')), str(sample.get('sid'))) in hints for _, sample in indexed))
    indexed.sort(key=lambda item: (-hints.get((str(item[1].get('bench')), str(item[1].get('sid'))), 0.0), item[0]))
    return ([sample for _, sample in indexed], matched)

def _interleave_benchmark_samples(samples: list[dict]) -> list[dict]:
    pass
    queues: dict[str, deque] = {}
    for sample in samples:
        queues.setdefault(str(sample['bench']), deque()).append(sample)
    result = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                result.append(queue.popleft())
    return result

def _images(sample: dict) -> list[Path]:
    values = sample.get('image_paths')
    if values is None:
        value = sample.get('image_path')
        values = [] if value is None else [value]
    elif isinstance(values, (str, Path)):
        values = [values]
    return [Path(value) for value in values]

def _source_urls(sample: dict, count: int) -> list[str | None]:
    values = sample.get('image_source_urls') or []
    if isinstance(values, str):
        values = [values]
    return [str(values[index]) if index < len(values) and str(values[index]).startswith(('http://', 'https://')) else None for index in range(count)]

def _is_format_failure(row: dict) -> bool:
    pass
    termination_reason = row.get('termination_reason')
    if termination_reason in _NON_RETRYABLE_SCORED_FAILURE_TERMINATIONS:
        return False
    if termination_reason in _RETRYABLE_FORMAT_FAILURE_TERMINATIONS:
        return True
    return any((event.get('role') == 'assistant' and event.get('format_error') for event in row.get('trajectory') or []))

def _infer_format_failure_rollout_policy(row: dict) -> str:
    pass
    if row.get('assistant_format_profile') in _FORMAT_FAILURE_RESAMPLING_PROFILES:
        return SELECTIVE_FORMAT_FAILURE_RESAMPLING
    return SCORE_FIRST_FORMAT_FAILURE_OUTCOME

def _format_failure_rollout_policy_for_row(row: dict) -> str:
    value = row.get('format_failure_rollout_policy')
    if value in FORMAT_FAILURE_ROLLOUT_POLICIES:
        return str(value)
    return _infer_format_failure_rollout_policy(row)

def _should_archive_existing_row(row: dict, *, retry_source_run: Path | None, assistant_format_profile: str, format_failure_rollout_policy: str) -> bool:
    pass
    if row.get('error'):
        return True
    if not _is_format_failure(row):
        return False
    if retry_source_run is not None:
        return True
    return format_failure_rollout_policy == SELECTIVE_FORMAT_FAILURE_RESAMPLING and assistant_format_profile in _FORMAT_FAILURE_RESAMPLING_PROFILES

def _backfill_empty_adapter_transforms(row: dict) -> int:
    pass
    changed = 0
    for event in row.get('trajectory') or []:
        if not isinstance(event, dict):
            continue
        adaptation = event.get('format_adaptation')
        if not isinstance(adaptation, dict) or adaptation.get('transforms'):
            continue
        adaptation['transforms'] = ['normalize_native_tool_call']
        changed += 1
    if changed:
        counts = Counter(row.get('format_adapter_transform_counts') or {})
        counts['normalize_native_tool_call'] += changed
        row['format_adapter_transform_counts'] = dict(sorted(counts.items()))
    return changed

def _parse_model_list(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    values = value.split(',') if isinstance(value, str) else value
    return tuple(dict.fromkeys((item.strip() for item in values if item.strip())))

def _mei_route_audit(trace: dict) -> dict:
    pass
    actual_models: Counter = Counter()
    gateway_calls = 0
    fallback_calls = 0
    failed_gateway_calls = 0

    def visit(value) -> None:
        nonlocal gateway_calls, fallback_calls, failed_gateway_calls
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'mei_timing' and isinstance(child, dict):
                    gateway_call = child.get('gateway_call')
                    if isinstance(gateway_call, dict):
                        gateway_calls += 1
                        actual_model = gateway_call.get('actual_model')
                        if actual_model:
                            actual_models[str(actual_model)] += 1
                        else:
                            failed_gateway_calls += 1
                        fallback_calls += int(bool(gateway_call.get('fallback_used')))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(trace)
    return {'mei_gateway_calls': gateway_calls, 'mei_actual_models': dict(sorted(actual_models.items())), 'mei_fallback_calls': fallback_calls, 'mei_failed_gateway_calls': failed_gateway_calls}

def _summary(rows: list[dict], selected: tuple[str, ...]) -> dict:
    per_benchmark = {}
    for benchmark in selected:
        bench_rows = [row for row in rows if row.get('bench') == benchmark]
        skipped = [row for row in bench_rows if row.get('skipped')]
        completed = [row for row in bench_rows if not row.get('error') and (not row.get('skipped'))]
        scored = [row for row in completed if not row.get('reference_missing')]
        correct = sum((bool(row.get('correct')) for row in scored))
        protocol_totals = sum((Counter(row.get('protocol_metrics') or {}) for row in completed), Counter())
        mei_models = Counter((str(row.get('mei_model')) for row in completed if row.get('mei_model') is not None))
        mei_fallback_chains = Counter((' -> '.join([str(row.get('mei_model')), *(str(model) for model in row.get('mei_fallback_models') or [])]) for row in completed if row.get('mei_model') is not None))
        mei_actual_models = sum((Counter(row.get('mei_actual_models') or {}) for row in completed), Counter())
        judge_models = Counter((str(row.get('judge_model')) for row in completed if row.get('judge_model') is not None))
        per_benchmark[benchmark] = {'total': len(bench_rows), 'completed': len(completed), 'skipped': len(skipped), 'skip_reasons': dict(sorted(Counter((row.get('skip_reason') or 'unknown' for row in skipped)).items())), 'errors': sum((bool(row.get('error')) for row in bench_rows)), 'scored': len(scored), 'correct': correct, 'accuracy': correct / len(scored) if scored else None, 'mei_models': dict(sorted(mei_models.items())), 'mei_fallback_chains': dict(sorted(mei_fallback_chains.items())), 'mei_actual_models': dict(sorted(mei_actual_models.items())), 'mei_gateway_calls': sum((int(row.get('mei_gateway_calls') or 0) for row in completed)), 'mei_fallback_calls': sum((int(row.get('mei_fallback_calls') or 0) for row in completed)), 'mei_failed_gateway_calls': sum((int(row.get('mei_failed_gateway_calls') or 0) for row in completed)), 'judge_models': dict(sorted(judge_models.items())), 'final_answers': sum((row.get('termination_reason') == 'final_answer' for row in completed)), 'forced_finalization_failures': sum((row.get('termination_reason') == 'forced_finalization_failed' for row in completed)), 'format_failures': sum((_is_format_failure(row) for row in completed)), 'format_adapted_rows': sum((int(row.get('format_adapted_turns') or 0) > 0 for row in completed)), 'format_adapted_turns': sum((int(row.get('format_adapted_turns') or 0) for row in completed)), 'format_recovery_model_calls': sum((int(row.get('format_recovery_model_calls') or 0) for row in completed)), 'average_turns': sum((row.get('n_turns', 0) for row in completed)) / len(completed) if completed else None, 'tool_calls': dict(sorted(sum((Counter(row.get('tool_call_counts') or {}) for row in completed), Counter()).items())), 'protocol_metrics': {key: int(protocol_totals.get(key, 0)) for key in ('native_reasoning_parse_success', 'native_tool_parse_success', 'arguments_json_parse_success', 'schema_validation_success', 'argument_normalization_count', 'runtime_type_validation_success', 'tool_execution_success', 'final_answer_parse_success', 'strict_parse_failed', 'compat_recovery_count', 'corrective_retry_count', 'protocol_failure_count')}, 'average_tool_steps': sum((int(row.get('tool_calls_executed') or 0) for row in completed)) / len(completed) if completed else None, 'average_model_calls': sum((int(row.get('model_calls') or 0) for row in completed)) / len(completed) if completed else None}
    return {'schema_version': 'neos-eval-summary', 'protocol_version': NEOS_PROTOCOL_VERSION, 'updated_at_utc': _utc_now(), 'benchmarks': per_benchmark, 'total': sum((item['total'] for item in per_benchmark.values())), 'completed': sum((item['completed'] for item in per_benchmark.values())), 'skipped': sum((item['skipped'] for item in per_benchmark.values())), 'errors': sum((item['errors'] for item in per_benchmark.values()))}

def _unified_repeat_summary(rows: list[dict], selected: tuple[str, ...], repeat_count: int) -> dict:
    pass
    benchmarks = {}
    for benchmark in selected:
        bench_rows = [row for row in rows if row.get('bench') == benchmark]
        sid_sets = {repeat: {str(row.get('sid')) for row in bench_rows if _task_repeat(row) == repeat} for repeat in range(1, repeat_count + 1)}
        canonical_sids = set().union(*sid_sets.values()) if sid_sets else set()
        per_repeat = []
        correct_by_sid: Counter[str] = Counter()
        for repeat in range(1, repeat_count + 1):
            repeat_rows = [row for row in bench_rows if _task_repeat(row) == repeat]
            correct = sum((row.get('correct') is True for row in repeat_rows))
            for row in repeat_rows:
                if row.get('correct') is True:
                    correct_by_sid[str(row.get('sid'))] += 1
            denominator = len(canonical_sids)
            per_repeat.append({'repeat': repeat, 'rows': len(repeat_rows), 'unique_sids': len(sid_sets[repeat]), 'correct': correct, 'errors': sum((bool(row.get('error')) for row in repeat_rows)), 'skipped': sum((bool(row.get('skipped')) for row in repeat_rows)), 'fixed_denominator_total': denominator, 'fixed_denominator_accuracy': correct / denominator if denominator else None})
        pooled_correct = sum((item['correct'] for item in per_repeat))
        pooled_total = len(canonical_sids) * repeat_count
        histogram = Counter((correct_by_sid.get(sid, 0) for sid in canonical_sids))
        passed = sum((correct_by_sid.get(sid, 0) > 0 for sid in canonical_sids))
        benchmarks[benchmark] = {'repeat_count': repeat_count, 'canonical_unique_sids': len(canonical_sids), 'all_repeats_share_sid_set': all((values == canonical_sids for values in sid_sets.values())), 'runs': per_repeat, 'average': {'pooled_correct': pooled_correct, 'pooled_total': pooled_total, 'accuracy_mean': pooled_correct / pooled_total if pooled_total else None}, 'pass': {'at_least_one_correct': passed, 'total': len(canonical_sids), 'pass_rate': passed / len(canonical_sids) if canonical_sids else None, 'correct_repeat_count_histogram': {str(key): histogram.get(key, 0) for key in range(repeat_count + 1)}}}
    return {'schema_version': 'neos-unified-repeat-summary', 'created_at_utc': _utc_now(), 'repeat_count': repeat_count, 'benchmarks': benchmarks}

def _generation_limit_for_run(existing_rows: list[dict]) -> int:
    pass
    limits = {int(row['max_generation_tokens']) for row in existing_rows if not row.get('error') and row.get('max_generation_tokens') is not None}
    if len(limits) > 1:
        raise RuntimeError(f'cannot resume trajectories with mixed max_generation_tokens: {sorted(limits)}')
    if limits:
        return limits.pop()
    return MAX_ROLLOUT_RESPONSE_TOKENS

def main() -> None:
    evaluator_started = time.perf_counter()
    invocation_started_utc = _utc_now()
    phase_timings_ms: dict[str, float] = {}
    parser = argparse.ArgumentParser(description='Neos multimodal agentic evaluation')
    parser.add_argument('--model', default=os.getenv('NEOS_EVAL_MODEL_SPEC', ''))
    parser.add_argument('--bench', type=canonical_benchmark_selection, choices=[*BENCHMARK_BY_ID, *SELECTION_GROUPS], default='all')
    parser.add_argument('--out')
    parser.add_argument('--run-id', default=os.getenv('NEOS_EVAL_RUN_ID', ''), help='Optional run name; omitted names create a new output directory automatically.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--execution-scheduler', choices=('trajectory', 'turn_event'), default='trajectory', help='trajectory keeps one worker for a whole rollout; turn_event uses separate model/tool/judge event pools and cooperatively requeues the trajectory after every completed turn.')
    parser.add_argument('--trajectory-window', type=int, default=0, help='Maximum logical trajectories admitted by turn_event scheduling. Zero uses 2x --workers.')
    parser.add_argument('--tool-event-workers', type=int, default=0, help='Tool-event pool size; zero uses --workers.')
    parser.add_argument('--postprocess-event-workers', type=int, default=0, help='Tool-result rendering/image/checkpoint pool size; zero uses the tool-event pool size. These workers never consume model slots.')
    parser.add_argument('--judge-event-workers', type=int, default=0, help='Judge-event pool size; zero uses min(--workers, 256).')
    parser.add_argument('--repeat-count', type=int, default=1, help='Evaluate every selected canonical sample this many times inside one global queue. Rows retain evaluation_repeat provenance.')
    parser.add_argument('--scheduling-hints', type=Path, help='Prior trajectories.jsonl used only to order pending samples by historical sample_worker_wall_ms, longest predicted first. This reduces straggler tails without changing benchmark contents.')
    parser.add_argument('--interleave-benchmarks', action='store_true', help="Opt-in round-robin pending admission across benchmarks; preserve each benchmark's order.")
    parser.add_argument('--max-turns', type=int, default=MAX_ASSISTANT_TURNS)
    parser.add_argument('--context-length', type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument('--model-native-context-length', type=int, default=0, help='Native limit read from the local model config. Inputs are skipped only after this limit is exhausted. Defaults to --context-length.')
    parser.add_argument('--assistant-format-profile', choices=ASSISTANT_FORMAT_PROFILES, default='openai_native', help="Native OpenAI tools/messages protocol. Tokenizer-specific syntax is owned by the model's official chat template and server parser.")
    parser.add_argument('--tool-schema-version', choices=NEOS_SCHEMA_VERSIONS, default=DEFAULT_NEOS_SCHEMA_VERSION, help='Native array-based tool argument schema.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dataset-variant', default='', help='Stable identifier for a non-canonical benchmark split. This is provenance/display metadata and does not alter sample loading.')
    parser.add_argument('--limit-per-benchmark', type=int, default=0, help='Optional sample limit applied independently to every selected benchmark. Zero keeps every row.')
    parser.add_argument('--shard', default='0/1')
    parser.add_argument('--retry-source-run', type=Path, help='Continue from an immutable source run. Re-evaluate source rows that terminated with an assistant-format failure and evaluate canonical samples not yet present in an incomplete source run. Valid source rows are not repeated.')
    parser.add_argument('--format-failure-rollout-policy', choices=FORMAT_FAILURE_ROLLOUT_POLICIES, default=SELECTIVE_FORMAT_FAILURE_RESAMPLING, help='selective_resample preserves the historical compatibility-run behavior of replacing terminal ordinary-turn format failures on a later evaluator pass. score_first_attempt keeps the adapter and same-turn corrective calls, but permanently scores the first trajectory outcome without whole-rollout replacement.')
    parser.add_argument('--mei-model', default=os.getenv('NEOS_MEI_MODEL_SPEC', 'qwen3.7-flash'))
    parser.add_argument('--mei-fallback-models', default=os.getenv('NEOS_MEI_FALLBACK_MODELS', ''), help='Comma-separated ordered MEI fallbacks; no fallback is configured by default.')
    parser.add_argument('--disable-mei-vlm', action='store_true', help='Metadata-only degraded diagnostic; not a paper-faithful run.')
    parser.add_argument('--degraded-skip-image-staging-preflight', action='store_true')
    args = parser.parse_args()
    if not args.model:
        parser.error('Fill NEOS_EVAL_MODEL_SPEC in config.local.json, or pass --model')
    args.mei_fallback_models = tuple((model for model in _parse_model_list(args.mei_fallback_models) if model != args.mei_model))
    if args.workers <= 0 or args.max_turns <= 0 or args.repeat_count <= 0:
        parser.error('--workers and --max-turns must be positive')
    if min(args.trajectory_window, args.tool_event_workers, args.postprocess_event_workers, args.judge_event_workers) < 0:
        parser.error('event worker/window values must be non-negative')
    if args.repeat_count > 1 and args.retry_source_run is not None:
        parser.error('--repeat-count > 1 cannot be combined with --retry-source-run')
    if args.limit < 0 or args.limit_per_benchmark < 0:
        parser.error('--limit and --limit-per-benchmark must be non-negative')
    if args.interleave_benchmarks and args.scheduling_hints is not None:
        parser.error('--interleave-benchmarks and --scheduling-hints select different ordering policies')
    if args.dataset_variant and (not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.dataset_variant)):
        parser.error('--dataset-variant may contain only letters, digits, dot, underscore, and dash')
    if args.limit and args.limit_per_benchmark:
        parser.error('--limit and --limit-per-benchmark are mutually exclusive')
    if args.max_turns != MAX_ASSISTANT_TURNS:
        parser.error(f'Neos canonical train/infer contract requires --max-turns {MAX_ASSISTANT_TURNS}; got {args.max_turns}')
    if args.format_failure_rollout_policy == SCORE_FIRST_FORMAT_FAILURE_OUTCOME and args.retry_source_run is not None:
        parser.error('score_first_attempt cannot be combined with --retry-source-run')
    if args.tool_schema_version != DEFAULT_NEOS_SCHEMA_VERSION:
        parser.error(f'unsupported Neos tool schema: {args.tool_schema_version!r}')
    model_native_context_length = args.model_native_context_length or args.context_length
    if args.context_length <= 0 or model_native_context_length <= 0:
        parser.error('context lengths must be positive')
    if args.context_length != model_native_context_length:
        parser.error('Neos native-context policy requires --context-length to equal --model-native-context-length')
    try:
        shard_index, shard_count = (int(value) for value in args.shard.split('/', 1))
    except Exception:
        parser.error('--shard must be k/n')
    if not 0 <= shard_index < shard_count:
        parser.error('--shard requires 0 <= k < n')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    selected = expand_benchmark_selection(args.bench)
    phase_started = time.perf_counter()
    try:
        output_dir = evaluation_output_path(benchmark=args.bench, repeat_count=args.repeat_count, run_id=args.run_id, out=args.out)
    except ValueError as exc:
        parser.error(str(exc))
    args.out = str(output_dir)
    args.run_id = output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Evaluation run: %s; output: %s', args.run_id, output_dir)
    retry_source_run = args.retry_source_run.expanduser().resolve() if args.retry_source_run is not None else None
    retry_source_rows: list[dict] = []
    retry_source_keys: set[tuple[str, str]] = set()
    retry_source_selected_keys: set[tuple[str, str]] = set()
    retry_unseen_keys: set[tuple[str, str]] = set()
    if retry_source_run is not None:
        if retry_source_run == output_dir:
            parser.error('--retry-source-run must differ from --out')
        retry_source_rows = _read_jsonl(retry_source_run / 'trajectories.jsonl')
        retry_source_keys = {(str(row.get('bench')), str(row.get('sid'))) for row in retry_source_rows if _is_format_failure(row) and str(row.get('bench')) in selected}
        retry_source_selected_keys = {(str(row.get('bench')), str(row.get('sid'))) for row in retry_source_rows if str(row.get('bench')) in selected}
    invocation_path = output_dir / 'evaluation_invocations.jsonl'
    existing_invocations = _read_jsonl(invocation_path)
    trajectory_path = output_dir / 'trajectories.jsonl'
    if not trajectory_path.exists():
        descriptor = os.open(trajectory_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
        os.close(descriptor)
    elif stat.S_IMODE(trajectory_path.stat().st_mode) != 384:
        os.chmod(trajectory_path, 384)
    artifact_store = EvaluationArtifactStore()
    existing_snapshot = _read_jsonl(trajectory_path, repair_truncated_tail=True)
    trajectory_publisher = DurableJSONLPublisher(trajectory_path, existing_rows=existing_snapshot, before_publish=artifact_store.publish_row_artifacts)
    trajectory_publisher.flush(float(os.getenv('NEOS_EVAL_RECOVERY_FLUSH_TIMEOUT_SECONDS', '300')))
    if trajectory_publisher.stats()['published_rows']:
        existing_snapshot = _read_jsonl(trajectory_path, repair_truncated_tail=True)
    phase_timings_ms['output_and_resume_snapshot'] = (time.perf_counter() - phase_started) * 1000
    run_max_generation_tokens = _generation_limit_for_run(existing_snapshot)
    phase_started = time.perf_counter()
    model = load_model_backend(args.model)
    ensure_template = model.ensure_neos_chat_template
    ensure_template(expected_context_length=args.context_length)
    backend_manifest_fn = getattr(model, 'backend_manifest', None)
    model_backend = backend_manifest_fn() if callable(backend_manifest_fn) else {'type': type(model).__qualname__}
    if args.assistant_format_profile in {QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE}:
        replicas = model_backend.get('replicas') or [model_backend]
        families = {str(replica.get('model_family') or '') for replica in replicas if isinstance(replica, dict)}
        if families != {'qwen3.5'}:
            raise RuntimeError(f'{args.assistant_format_profile} is restricted to verified local Qwen3.5 replicas; observed model families={sorted(families)!r}')
    phase_timings_ms['model_backend_init_and_template_preflight'] = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    search = SearchGateway()
    search.require_interfaces()
    search_status = search.config_status()
    if args.degraded_skip_image_staging_preflight:
        image_preflight = {'ok': False, 'skipped_degraded': True}
    else:
        if not search_status['local_reverse_image_ready']:
            raise RuntimeError('Neos ReverseImageSearch requires Neos image staging; configure OSS or use the explicit degraded preflight flag.')
        image_preflight = search.live_image_staging_preflight()
    artifact_status = artifact_store.preflight()
    mei = False if args.disable_mei_vlm else NeosMEI(model=args.mei_model, fallback_models=args.mei_fallback_models)
    runtime = NeosRuntime(client=search, mei=mei, config={'tool_schema_version': args.tool_schema_version})
    image_codec = codec_config(runtime.config)
    judge = LLMGateway(model=secrets.JUDGE_MODEL)
    phase_timings_ms['tool_mei_judge_and_live_preflights'] = (time.perf_counter() - phase_started) * 1000
    manifest = {'schema_version': 'neos-evaluation-run', 'status': 'running', 'started_at_utc': _utc_now(), 'model': args.model, 'benchmark_selection': args.bench, 'dataset_variant': args.dataset_variant or None, 'benchmark_ids': list(selected), 'limit': args.limit, 'limit_per_benchmark': args.limit_per_benchmark, 'shard': args.shard, 'workers': args.workers, 'execution_scheduler': args.execution_scheduler, 'trajectory_window': args.trajectory_window or args.workers * 2 if args.execution_scheduler == 'turn_event' else args.workers, 'tool_event_workers': args.tool_event_workers or args.workers if args.execution_scheduler == 'turn_event' else None, 'postprocess_event_workers': args.postprocess_event_workers or args.tool_event_workers or args.workers if args.execution_scheduler == 'turn_event' else None, 'judge_event_workers': args.judge_event_workers or min(args.workers, 256) if args.execution_scheduler == 'turn_event' else None, 'repeat_count': args.repeat_count, 'global_repeat_benchmark_queue': bool(args.repeat_count > 1 or len(selected) > 1), 'max_turns': args.max_turns, 'context_length_tokens': args.context_length, 'model_native_context_length_tokens': model_native_context_length, 'context_limit_policy': 'skip_only_when_native_context_is_exhausted', 'assistant_format_profile': args.assistant_format_profile, 'assistant_format_adapter_contract_version': assistant_format_adapter_contract_version(args.assistant_format_profile), 'format_failure_rollout_policy': args.format_failure_rollout_policy, 'whole_rollout_format_failure_replacement': args.format_failure_rollout_policy == SELECTIVE_FORMAT_FAILURE_RESAMPLING, 'max_generation_tokens': run_max_generation_tokens, 'max_tool_response_chars': MAX_TOOL_RESPONSE_LENGTH, 'protocol_version': NEOS_PROTOCOL_VERSION, 'toolset_profile': NEOS_TOOLSET_PROFILE, 'available_tools': sorted(NEOS_TOOL_NAMES), 'tool_schema_version': args.tool_schema_version, 'context_image_codec': image_codec, 'model_backend': model_backend, 'mei_vlm_enabled': not args.disable_mei_vlm, 'mei_model': None if args.disable_mei_vlm else args.mei_model, 'mei_fallback_models': None if args.disable_mei_vlm else list(args.mei_fallback_models), 'judge_model': secrets.JUDGE_MODEL, 'image_staging_preflight': image_preflight, 'artifact_store': artifact_status, 'paper': os.getenv('NEOS_PAPER_URL', '')}
    if retry_source_run is not None:
        manifest['format_retry'] = {'source_run': str(retry_source_run), 'source_trajectory_file': str((retry_source_run / 'trajectories.jsonl').resolve()), 'selected_failure_rows': len(retry_source_keys), 'selected_unseen_rows': None, 'source_run_is_immutable': True, 'continuation_policy': 'retryable_format_failures_plus_unseen'}
    _atomic_json(output_dir / 'run_manifest.json', manifest)
    phase_started = time.perf_counter()
    loaders = {**TEXT_SEARCH_LOADERS, 'mmsearch': load_mmsearch, 'browsecomp_vl': load_bcvl, 'mm_browsecomp': load_mmbrowsecomp, 'mmsearch_plus': load_mmsearch_plus, 'simplevqa': load_simplevqa, 'livevqa': load_livevqa, 'fvqa': load_fvqa, 'infoseek': load_infoseek, 'vdr': load_vdr, 'browsecomp_v3': load_browsecomp_v3, 'visbrowse': load_visbrowse}
    all_samples = []
    for benchmark in selected:
        loaded = loaders[benchmark]()
        if args.limit_per_benchmark:
            loaded = loaded[:args.limit_per_benchmark]
        all_samples.extend(loaded)
    all_samples = [sample for index, sample in enumerate(all_samples) if index % shard_count == shard_index]
    if retry_source_run is not None:
        canonical_keys = {(str(sample['bench']), str(sample['sid'])) for sample in all_samples}
        retry_unseen_keys = canonical_keys - retry_source_selected_keys
        retry_pending_keys = retry_source_keys | retry_unseen_keys
        if not retry_pending_keys:
            parser.error('--retry-source-run has no selected format failures or unseen samples')
        all_samples = [sample for sample in all_samples if (str(sample['bench']), str(sample['sid'])) in retry_pending_keys]
        manifest['format_retry']['selected_unseen_rows'] = len(retry_unseen_keys)
        manifest['format_retry']['selected_pending_rows'] = len(retry_pending_keys)
        _atomic_json(output_dir / 'run_manifest.json', manifest)
    if args.limit:
        all_samples = all_samples[:args.limit]
    canonical_sample_count = len(all_samples)
    if args.repeat_count > 1:
        all_samples = [{**sample, 'evaluation_repeat': repeat} for sample in all_samples for repeat in range(1, args.repeat_count + 1)]
    eligible_sample_keys = {_task_key(sample) for sample in all_samples}
    manifest['canonical_samples_before_repeat_expansion'] = canonical_sample_count
    manifest['expanded_sample_tasks'] = len(all_samples)
    _atomic_json(output_dir / 'run_manifest.json', manifest)
    phase_timings_ms['benchmark_dataset_load'] = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    existing = existing_snapshot
    backfilled_adapter_turns = sum((_backfill_empty_adapter_transforms(row) for row in existing))
    judge_only_existing = [row for row in existing if _is_judge_only_pending(row)]
    failed_existing = [row for row in existing if not _is_judge_only_pending(row) and _should_archive_existing_row(row, retry_source_run=retry_source_run, assistant_format_profile=args.assistant_format_profile, format_failure_rollout_policy=args.format_failure_rollout_policy)]
    out_of_scope_existing = [row for row in existing if _task_key(row) not in eligible_sample_keys] if retry_source_run is not None else []
    retry_existing_ids = {id(row) for row in [*judge_only_existing, *failed_existing, *out_of_scope_existing]}
    if retry_existing_ids or backfilled_adapter_turns:
        backup_path = output_dir / 'resume_backups' / f"trajectories-before-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        _atomic_jsonl(backup_path, existing)
        existing = [row for row in existing if id(row) not in retry_existing_ids]
        _atomic_jsonl(trajectory_path, existing)
        if backfilled_adapter_turns:
            logger.info('backfilled %d empty native-adapter transform trace(s)', backfilled_adapter_turns)
        logger.info('Neos resume selected %d judge-only row(s), archived %d other retryable error row(s), and %d out-of-scope row(s) to %s', len(judge_only_existing), len(failed_existing), len(out_of_scope_existing), backup_path)
    trajectory_publisher.reset_known_rows(existing)
    completed_keys = {_task_key(row) for row in existing if not row.get('error')}
    judge_only_keys = {_task_key(row) for row in judge_only_existing}
    pending = [sample for sample in all_samples if _task_key(sample) not in completed_keys and _task_key(sample) not in judge_only_keys]
    duration_hints: dict[tuple[str, str], float] = {}
    if args.scheduling_hints is not None:
        scheduling_path = args.scheduling_hints.expanduser().resolve()
        if not scheduling_path.is_file():
            raise FileNotFoundError(f'scheduling hints are unavailable: {scheduling_path}')
        duration_hints = _load_scheduling_hints(scheduling_path)
        pending, matched_hints = _longest_predicted_first(pending, duration_hints)
        manifest['sample_scheduling'] = {'policy': 'historical_longest_predicted_first', 'hints_path': str(scheduling_path), 'hints_rows': len(duration_hints), 'pending_rows': len(pending), 'judge_only_pending_rows': len(judge_only_existing), 'matched_pending_rows': matched_hints, 'unknown_rows_keep_canonical_relative_order': True, 'result_semantics_unchanged': True}
        _atomic_json(output_dir / 'run_manifest.json', manifest)
        logger.info('Neos scheduling=historical_longest_predicted_first matched=%d/%d hints=%s', matched_hints, len(pending), scheduling_path)
    elif args.interleave_benchmarks:
        pending = _interleave_benchmark_samples(pending)
        manifest['sample_scheduling'] = {'policy': 'round_robin_benchmarks', 'benchmark_order': list(dict.fromkeys((str(row['bench']) for row in pending))), 'pending_rows': len(pending), 'judge_only_pending_rows': len(judge_only_existing), 'per_benchmark_order_preserved': True}
        _atomic_json(output_dir / 'run_manifest.json', manifest)
        logger.info('Neos scheduling=round_robin_benchmarks pending=%d', len(pending))
    else:
        manifest['sample_scheduling'] = {'policy': 'canonical_input_order', 'pending_rows': len(pending), 'judge_only_pending_rows': len(judge_only_existing)}
        _atomic_json(output_dir / 'run_manifest.json', manifest)
    logger.info('Neos selected=%d existing=%d pending=%d agent_pending=%d judge_only_pending=%d', len(all_samples), len(existing), len(pending) + len(judge_only_existing), len(pending), len(judge_only_existing))
    phase_timings_ms['resume_compatibility_and_pending_selection'] = (time.perf_counter() - phase_started) * 1000
    checkpoint_base = Path(os.getenv('NEOS_EVAL_TURN_CHECKPOINT_ROOT', '/tmp/neos-eval-turn-checkpoints')).expanduser().resolve()
    checkpoint_token = path_namespace(output_dir)
    turn_checkpoint_root = checkpoint_base / checkpoint_token
    turn_checkpoint_root.mkdir(parents=True, exist_ok=True, mode=448)
    manifest['turn_checkpoint_store'] = {'mode': 'instance-local-recoverable', 'root': str(turn_checkpoint_root), 'shared_cpfs_writes': False}

    def turn_checkpoint_path(sample: dict) -> Path:
        key = f"{sample.get('bench')}\x00{sample.get('sid')}\x00{_task_repeat(sample)}"
        digest = file_key([sample.get('bench'), sample.get('sid'), _task_repeat(sample)])
        return turn_checkpoint_root / f'{digest}.json'

    def trajectory_route_key(sample: dict) -> str:
        bench, sid, repeat = _task_key(sample)
        return f'{output_dir.name}\x00{bench}\x00{sid}\x00repeat{repeat}'

    def trajectory_route_weight(sample: dict) -> float:
        return max(1.0, float(duration_hints.get((str(sample.get('bench')), str(sample.get('sid'))), 1.0)))

    def apply_judge_to_completed_trace(result: dict, *, sample_started: float, image_path_resolution_ms: float, judge_only_resume: bool) -> dict:
        pass
        reference_missing = bool(result.get('reference_missing'))
        judge_started = time.perf_counter()
        transport_attempts: list[dict] = []
        if reference_missing:
            judgement = {'correct': None, 'judgement': 'reference missing; excluded from scored accuracy', 'judge_attempts': [], 'judge_parse_failed': False}
            judge_latency_ms = None
            judge_skipped_reason = 'reference_missing'
        else:
            try:
                judgement, transport_attempts, judge_latency_ms = _judge_with_transport_retries(judge, question=str(result.get('question') or ''), correct_answer='; '.join((str(value) for value in result.get('gt_answer') or result.get('answers') or [])), final_answer=result.get('final_answer'))
            except JudgeOnlyRetryExhaustedError as exc:
                judge_latency_ms = (time.perf_counter() - judge_started) * 1000
                history = list(result.get('judge_transport_attempts') or [])
                history.extend(exc.attempts)
                result.update({'error': f'{type(exc).__name__}: {exc}', 'judge_only_pending': True, 'judge_only_retry_schema_version': JUDGE_ONLY_RETRY_SCHEMA_VERSION, 'judge_transport_attempts': history, 'judge_model': judge.model, 'correct': None, 'judge_correct': None, 'judgement': None, 'judge_raw_response': None, 'judge_attempts': [], 'judge_parse_failed': True})
                timing = dict(result.get('evaluation_timing') or {})
                prior_worker_ms = float(timing.get('sample_worker_wall_ms') or 0.0)
                current_worker_ms = (time.perf_counter() - sample_started) * 1000
                timing.update({'image_path_resolution_ms': round(image_path_resolution_ms, 3), 'judge_latency_ms': round(judge_latency_ms, 3), 'judge_skipped_reason': 'judge_transport_exhausted', 'judge_only_retry_wall_ms': round(current_worker_ms, 3), 'sample_worker_wall_ms': round(prior_worker_ms + current_worker_ms if judge_only_resume else current_worker_ms, 3)})
                result['evaluation_timing'] = timing
                logger.error('Neos preserving completed Agent trace for judge-only retry bench=%s sid=%s', result.get('bench'), result.get('sid'))
                return result
            judge_skipped_reason = 'missing_final_answer' if not str(result.get('final_answer') or '').strip() else None
        result.pop('error', None)
        result.pop('judge_only_pending', None)
        result.update({'judge_only_retry_schema_version': JUDGE_ONLY_RETRY_SCHEMA_VERSION, 'judge_model': judge.model, 'correct': judgement.get('correct'), 'judge_correct': judgement.get('correct'), 'judgement': judgement.get('judgement'), 'judge_raw_response': judgement.get('judgement'), 'judge_attempts': judgement.get('judge_attempts') or [], 'judge_parse_failed': judgement.get('judge_parse_failed', False)})
        history = list(result.get('judge_transport_attempts') or [])
        history.extend(transport_attempts)
        result['judge_transport_attempts'] = history
        result['judge_only_retry_count'] = int(result.get('judge_only_retry_count') or 0) + int(judge_only_resume) + max(0, len(transport_attempts) - 1)
        timing = dict(result.get('evaluation_timing') or {})
        prior_worker_ms = float(timing.get('sample_worker_wall_ms') or 0.0)
        current_worker_ms = (time.perf_counter() - sample_started) * 1000
        timing.update({'image_path_resolution_ms': round(image_path_resolution_ms, 3), 'judge_latency_ms': round(judge_latency_ms, 3) if judge_latency_ms is not None else None, 'judge_skipped_reason': judge_skipped_reason, 'sample_worker_wall_ms': round(prior_worker_ms + current_worker_ms if judge_only_resume else current_worker_ms, 3)})
        if judge_only_resume:
            timing['judge_only_retry_wall_ms'] = round(current_worker_ms, 3)
            logger.info('Neos judge-only retry complete bench=%s sid=%s', result.get('bench'), result.get('sid'))
        result['evaluation_timing'] = timing
        return result

    def process_judge_only(row: dict) -> dict:
        sample_started = time.perf_counter()
        result = dict(row)
        image_path_resolution_ms = float((result.get('evaluation_timing') or {}).get('image_path_resolution_ms', 0.0) or 0.0)
        return apply_judge_to_completed_trace(result, sample_started=sample_started, image_path_resolution_ms=image_path_resolution_ms, judge_only_resume=True)

    def build_result_from_trace(sample: dict, trace: dict, *, route_info: dict) -> dict:
        mei_route_audit = _mei_route_audit(trace)
        return {'sid': str(sample['sid']), 'bench': sample['bench'], 'evaluation_repeat': _task_repeat(sample), 'answers': sample.get('answers') or [], 'gt_answer': sample.get('answers') or [], 'reference_missing': bool(sample.get('reference_missing')), 'area': sample.get('area'), 'subfield': sample.get('subfield'), 'domain': sample.get('domain'), 'category': sample.get('category'), 'subtask': sample.get('subtask'), 'level': sample.get('level'), 'answer_type': sample.get('answer_type'), 'input_variant': sample.get('input_variant'), 'evaluation_run_id': output_dir.name, 'format_failure_rollout_policy': args.format_failure_rollout_policy, 'context_length_tokens': args.context_length, 'model_native_context_length_tokens': model_native_context_length, 'max_generation_tokens': run_max_generation_tokens, 'max_tool_response_chars': MAX_TOOL_RESPONSE_LENGTH, 'judge_model': judge.model, 'model_backend': model_backend, 'trajectory_routing': route_info, 'tool_schema_version': args.tool_schema_version, 'mei_model': None if args.disable_mei_vlm else args.mei_model, 'mei_fallback_models': None if args.disable_mei_vlm else list(args.mei_fallback_models), **mei_route_audit, **trace}

    def process(sample: dict) -> dict:
        sample_started = time.perf_counter()
        route_key = trajectory_route_key(sample)
        route_info: dict = {}
        terminal_route_release = getattr(model, 'release_trajectory', None)
        try:
            image_resolve_started = time.perf_counter()
            images = _images(sample)
            image_path_resolution_ms = (time.perf_counter() - image_resolve_started) * 1000
            session_factory = getattr(model, 'trajectory_session', None)
            session = session_factory(route_key) if callable(session_factory) else nullcontext(None)
            with session as assigned_route:
                route_info = dict(assigned_route or {})
                trace = run_agent(question=sample['question'], image_inputs=images, image_source_urls=_source_urls(sample, len(images)), model=model, runtime=runtime, max_turns=args.max_turns, artifact_store=artifact_store, max_generation_tokens=run_max_generation_tokens, assistant_format_profile=args.assistant_format_profile, tool_schema_version=args.tool_schema_version, turn_checkpoint_path=turn_checkpoint_path(sample))
            result = build_result_from_trace(sample, trace, route_info=route_info)
            return apply_judge_to_completed_trace(result, sample_started=sample_started, image_path_resolution_ms=image_path_resolution_ms, judge_only_resume=False)
        except FatalLLMGatewayError:
            raise
        except ModelContextLengthExceededError as exc:
            logger.warning('Neos discarding native-context overflow bench=%s sid=%s input_tokens=%d native_context_tokens=%d', sample['bench'], sample['sid'], exc.input_tokens, model_native_context_length)
            return {'sid': str(sample['sid']), 'bench': sample['bench'], 'evaluation_repeat': _task_repeat(sample), 'question': sample.get('question'), 'answers': sample.get('answers') or [], 'gt_answer': sample.get('answers') or [], 'evaluation_run_id': output_dir.name, 'skipped': True, 'skip_reason': 'model_native_context_exceeded', 'input_context_tokens': exc.input_tokens, 'context_length_tokens': args.context_length, 'model_native_context_length_tokens': model_native_context_length, 'max_generation_tokens': run_max_generation_tokens, 'evaluation_timing': {'judge_skipped_reason': 'model_native_context_exceeded', 'sample_worker_wall_ms': round((time.perf_counter() - sample_started) * 1000, 3)}}
        except Exception as exc:
            logger.exception('Neos sample failed bench=%s sid=%s', sample['bench'], sample['sid'])
            return {'sid': str(sample['sid']), 'bench': sample['bench'], 'evaluation_repeat': _task_repeat(sample), 'question': sample.get('question'), 'answers': sample.get('answers') or [], 'error': f'{type(exc).__name__}: {exc}', 'evaluation_timing': {'sample_worker_wall_ms': round((time.perf_counter() - sample_started) * 1000, 3)}}
        finally:
            if callable(terminal_route_release):
                terminal_route_release(route_key)

    def release_job_route(job: dict) -> None:
        release = getattr(model, 'release_trajectory', None)
        if callable(release) and (not job.get('route_released')):
            release(job['route_key'])
            job['route_released'] = True

    def advance_agent_job(job: dict, resume_value=None, *, bind_model_route: bool=True) -> dict:
        pass
        sample = job['sample']
        try:
            if job.get('generator') is None:
                image_resolve_started = time.perf_counter()
                images = _images(sample)
                job['image_path_resolution_ms'] = (time.perf_counter() - image_resolve_started) * 1000
                job['generator'] = iter_agent_turns(question=sample['question'], image_inputs=images, image_source_urls=_source_urls(sample, len(images)), model=model, runtime=runtime, max_turns=args.max_turns, artifact_store=artifact_store, max_generation_tokens=run_max_generation_tokens, assistant_format_profile=args.assistant_format_profile, tool_schema_version=args.tool_schema_version, turn_checkpoint_path=turn_checkpoint_path(sample))
            session_factory = getattr(model, 'trajectory_session', None) if bind_model_route else None
            session = session_factory(job['route_key']) if callable(session_factory) else nullcontext(None)
            try:
                with session as assigned_route:
                    if assigned_route and (not job.get('route_info')):
                        job['route_info'] = dict(assigned_route)
                    event = job['generator'].send(resume_value)
            except StopIteration as completed:
                release_job_route(job)
                return {'kind': 'trace_complete', 'result': build_result_from_trace(sample, completed.value, route_info=dict(job.get('route_info') or {}))}
            if isinstance(event, AgentToolRequest):
                return {'kind': 'tool_request', 'event': event}
            if isinstance(event, AgentTurnComplete):
                return {'kind': 'turn_complete', 'event': event}
            raise TypeError(f'unsupported Agent cooperative event: {event!r}')
        except FatalLLMGatewayError:
            release_job_route(job)
            raise
        except ModelContextLengthExceededError as exc:
            release_job_route(job)
            logger.warning('Neos discarding native-context overflow bench=%s sid=%s repeat=%d input_tokens=%d native_context_tokens=%d', sample['bench'], sample['sid'], _task_repeat(sample), exc.input_tokens, model_native_context_length)
            return {'kind': 'row_complete', 'row': {'sid': str(sample['sid']), 'bench': sample['bench'], 'evaluation_repeat': _task_repeat(sample), 'question': sample.get('question'), 'answers': sample.get('answers') or [], 'gt_answer': sample.get('answers') or [], 'evaluation_run_id': output_dir.name, 'skipped': True, 'skip_reason': 'model_native_context_exceeded', 'input_context_tokens': exc.input_tokens, 'context_length_tokens': args.context_length, 'model_native_context_length_tokens': model_native_context_length, 'max_generation_tokens': run_max_generation_tokens, 'trajectory_routing': dict(job.get('route_info') or {}), 'evaluation_timing': {'judge_skipped_reason': 'model_native_context_exceeded', 'sample_worker_wall_ms': round((time.perf_counter() - job['sample_started']) * 1000, 3)}}}
        except Exception as exc:
            release_job_route(job)
            logger.exception('Neos cooperative sample failed bench=%s sid=%s repeat=%d', sample['bench'], sample['sid'], _task_repeat(sample))
            return {'kind': 'row_complete', 'row': {'sid': str(sample['sid']), 'bench': sample['bench'], 'evaluation_repeat': _task_repeat(sample), 'question': sample.get('question'), 'answers': sample.get('answers') or [], 'error': f'{type(exc).__name__}: {exc}', 'trajectory_routing': dict(job.get('route_info') or {}), 'evaluation_timing': {'sample_worker_wall_ms': round((time.perf_counter() - job['sample_started']) * 1000, 3)}}}

    def advance_postprocess_job(job: dict, tool_result: AgentToolResult) -> dict:
        pass
        outcome = advance_agent_job(job, tool_result, bind_model_route=False)
        if outcome.get('kind') not in {'turn_complete', 'row_complete', 'trace_complete'}:
            raise RuntimeError(f"tool postprocess crossed an unexpected cooperative boundary: {outcome.get('kind')!r}")
        return outcome

    def judge_completed_job(job: dict, result: dict) -> dict:
        return apply_judge_to_completed_trace(result, sample_started=job['sample_started'], image_path_resolution_ms=float(job.get('image_path_resolution_ms') or 0.0), judge_only_resume=False)
    completed_now = 0
    completed_rows: list[dict] = []
    persistence_path = output_dir / 'persistence_timing.jsonl'
    existing_persistence_events = _read_jsonl(persistence_path, repair_truncated_tail=True)
    persistence_events: list[dict] = []
    evaluation_loop_started = time.perf_counter()
    total_pending_work = len(pending) + len(judge_only_existing)

    def persist_completed_row(row: dict) -> None:
        nonlocal completed_now
        persistence_started = time.perf_counter()
        serialize_started = time.perf_counter()
        serialized = json.dumps(row, ensure_ascii=False) + '\n'
        json_serialize_ms = (time.perf_counter() - serialize_started) * 1000
        publish_info = trajectory_publisher.enqueue(row)
        local_write_ms = float(publish_info.get('local_write_ms') or 0.0)
        persistence_event = {'timestamp_utc': _utc_now(), 'sid': str(row.get('sid')), 'bench': row.get('bench'), 'evaluation_repeat': _task_repeat(row), 'serialized_bytes': len(serialized.encode('utf-8')), 'json_serialize_ms': round(json_serialize_ms, 3), 'storage_mode': 'local-spool-batched-publish', 'local_spool_write_and_fsync_ms': round(local_write_ms, 3), 'shared_publish_deferred': True, 'shared_pending_rows': trajectory_publisher.pending_count(), 'persistence_wall_ms': round((time.perf_counter() - persistence_started) * 1000, 3)}
        persistence_events.append(persistence_event)
        completed_rows.append(row)
        completed_now += 1
        if completed_now == 1 or completed_now % 10 == 0:
            logger.info('progress %d/%d', completed_now, total_pending_work)

    def run_turn_event_scheduler() -> None:
        model_workers = args.workers
        tool_workers = args.tool_event_workers or args.workers
        postprocess_workers = args.postprocess_event_workers or tool_workers
        judge_workers = args.judge_event_workers or min(args.workers, 256)
        trajectory_window = args.trajectory_window or args.workers * 2
        trajectory_window = max(trajectory_window, args.workers)
        pending_samples = deque(pending)
        ready_model: deque[tuple[dict, AgentToolResult | None]] = deque()
        active_trajectories = 0
        model_inflight = 0
        postprocess_inflight = 0
        replica_count = max(1, len(getattr(model, 'backends', ()) or (None,)))
        base_replica_workers, extra_replica_workers = divmod(model_workers, replica_count)
        model_worker_capacity_per_replica = [base_replica_workers + int(index < extra_replica_workers) for index in range(replica_count)]
        model_inflight_per_replica = [0 for _ in range(replica_count)]
        assign_trajectory_fn = getattr(model, 'assign_trajectory', None)
        trajectory_replica_fn = getattr(model, 'trajectory_replica', None)
        future_meta: dict[Future, tuple[str, dict | None]] = {}
        scheduler_events: Counter[str] = Counter()
        interval_tool_completed: Counter[str] = Counter()
        interval_tool_latency_ms: Counter[str] = Counter()
        interval_tool_latency_max_ms: dict[str, float] = {}
        interval_tool_stage_sum: dict[str, Counter[str]] = {}
        interval_tool_stage_max: dict[str, dict[str, float]] = {}
        next_scheduler_status = time.monotonic() + 30.0

        def admit() -> None:
            nonlocal active_trajectories
            while pending_samples and active_trajectories < trajectory_window:
                sample = pending_samples.popleft()
                ready_model.append(({'sample': sample, 'sample_started': time.perf_counter(), 'route_key': trajectory_route_key(sample), 'route_info': {}, 'route_released': False, 'generator': None, 'route_weight': trajectory_route_weight(sample), 'assigned_replica': None}, None))
                active_trajectories += 1

        def submit_ready_model(executor: ThreadPoolExecutor) -> None:
            nonlocal model_inflight
            while ready_model and model_inflight < model_workers:
                submitted_this_scan = False
                scan_size = len(ready_model)
                for _ in range(scan_size):
                    if model_inflight >= model_workers:
                        break
                    job, resume_value = ready_model.popleft()
                    replica = job.get('assigned_replica')
                    if replica is None and callable(trajectory_replica_fn):
                        replica = trajectory_replica_fn(job['route_key'])
                    if replica is None and callable(assign_trajectory_fn):
                        replica = assign_trajectory_fn(job['route_key'], estimated_weight=job.get('route_weight', 1.0), replica_load=model_inflight_per_replica)
                    if replica is None:
                        replica = 0
                    replica = int(replica)
                    if not 0 <= replica < replica_count:
                        raise RuntimeError(f'invalid trajectory replica assignment: {replica}')
                    job['assigned_replica'] = replica
                    if model_inflight_per_replica[replica] >= model_worker_capacity_per_replica[replica]:
                        ready_model.append((job, resume_value))
                        continue
                    future = executor.submit(advance_agent_job, job, resume_value)
                    job['model_inflight_replica'] = replica
                    future_meta[future] = ('model', job)
                    model_inflight += 1
                    model_inflight_per_replica[replica] += 1
                    submitted_this_scan = True
                if not submitted_this_scan:
                    break
        logger.info('Neos scheduler=turn_event global_queue=true model_workers=%d tool_workers=%d postprocess_workers=%d judge_workers=%d trajectory_window=%d repeats=%d benchmarks=%d', model_workers, tool_workers, postprocess_workers, judge_workers, trajectory_window, args.repeat_count, len(selected))
        with ThreadPoolExecutor(max_workers=model_workers) as model_executor, ThreadPoolExecutor(max_workers=tool_workers) as tool_executor, ThreadPoolExecutor(max_workers=postprocess_workers) as postprocess_executor, ThreadPoolExecutor(max_workers=judge_workers) as judge_executor:
            for row in judge_only_existing:
                future = judge_executor.submit(process_judge_only, row)
                future_meta[future] = ('judge_only', None)
            admit()
            submit_ready_model(model_executor)
            while future_meta or ready_model or pending_samples:
                if not future_meta:
                    admit()
                    submit_ready_model(model_executor)
                    if not future_meta:
                        raise RuntimeError('turn-event scheduler made no progress')
                completed, _ = wait(tuple(future_meta), return_when=FIRST_COMPLETED)
                for future in completed:
                    kind, job = future_meta.pop(future)
                    if kind == 'model':
                        model_inflight -= 1
                        assert job is not None
                        replica = int(job.pop('model_inflight_replica'))
                        model_inflight_per_replica[replica] -= 1
                        if model_inflight_per_replica[replica] < 0:
                            raise AssertionError('negative per-replica model inflight count')
                    elif kind == 'postprocess':
                        postprocess_inflight -= 1
                        if postprocess_inflight < 0:
                            raise AssertionError('negative postprocess inflight count')
                    try:
                        outcome = future.result()
                    except FatalLLMGatewayError:
                        for pending_future in future_meta:
                            pending_future.cancel()
                        raise
                    if kind == 'model':
                        assert job is not None
                        outcome_kind = outcome['kind']
                        if outcome_kind == 'tool_request':
                            job['current_tool_name'] = outcome['event'].name
                            tool_future = tool_executor.submit(outcome['event'].execute)
                            future_meta[tool_future] = ('tool', job)
                            scheduler_events['tool_requests'] += 1
                        elif outcome_kind == 'turn_complete':
                            ready_model.append((job, None))
                            scheduler_events['turns_completed'] += 1
                        elif outcome_kind == 'trace_complete':
                            judge_future = judge_executor.submit(judge_completed_job, job, outcome['result'])
                            future_meta[judge_future] = ('judge', job)
                            scheduler_events['traces_completed'] += 1
                        elif outcome_kind == 'row_complete':
                            persist_completed_row(outcome['row'])
                            active_trajectories -= 1
                            admit()
                        else:
                            raise AssertionError(outcome_kind)
                    elif kind == 'tool':
                        assert job is not None
                        if not isinstance(outcome, AgentToolResult):
                            raise TypeError(f'tool event returned unsupported result: {type(outcome)!r}')
                        completed_tool_name = str(job.get('current_tool_name') or 'unknown')
                        interval_tool_completed[completed_tool_name] += 1
                        interval_tool_latency_ms[completed_tool_name] += float(outcome.latency_ms)
                        interval_tool_latency_max_ms[completed_tool_name] = max(interval_tool_latency_max_ms.get(completed_tool_name, 0.0), float(outcome.latency_ms))
                        execution_metadata = getattr(outcome.execution, 'metadata', None)
                        execution_timing = execution_metadata.get('timing') if isinstance(execution_metadata, dict) else None
                        if isinstance(execution_timing, dict):
                            stage_sum = interval_tool_stage_sum.setdefault(completed_tool_name, Counter())
                            stage_max = interval_tool_stage_max.setdefault(completed_tool_name, {})
                            for stage, raw_value in execution_timing.items():
                                if not (stage == 'parallel_wall_ms' or stage.endswith('_ms_sum')) or not isinstance(raw_value, (int, float)):
                                    continue
                                value = float(raw_value)
                                stage_sum[stage] += value
                                stage_max[stage] = max(stage_max.get(stage, 0.0), value)
                        job.pop('current_tool_name', None)
                        postprocess_future = postprocess_executor.submit(advance_postprocess_job, job, outcome)
                        future_meta[postprocess_future] = ('postprocess', job)
                        postprocess_inflight += 1
                        scheduler_events['tool_results'] += 1
                        scheduler_events['postprocess_submitted'] += 1
                    elif kind == 'postprocess':
                        assert job is not None
                        outcome_kind = outcome['kind']
                        if outcome_kind == 'turn_complete':
                            ready_model.append((job, None))
                            scheduler_events['turns_completed'] += 1
                            scheduler_events['postprocess_completed'] += 1
                        elif outcome_kind == 'trace_complete':
                            judge_future = judge_executor.submit(judge_completed_job, job, outcome['result'])
                            future_meta[judge_future] = ('judge', job)
                            scheduler_events['traces_completed'] += 1
                        elif outcome_kind == 'row_complete':
                            persist_completed_row(outcome['row'])
                            active_trajectories -= 1
                            admit()
                        else:
                            raise AssertionError(outcome_kind)
                    elif kind == 'judge':
                        assert job is not None
                        persist_completed_row(outcome)
                        active_trajectories -= 1
                        admit()
                    elif kind == 'judge_only':
                        persist_completed_row(outcome)
                    else:
                        raise AssertionError(kind)
                submit_ready_model(model_executor)
                if time.monotonic() >= next_scheduler_status:
                    future_kinds = Counter((metadata[0] for metadata in future_meta.values()))
                    tools_by_name = Counter((str((job or {}).get('current_tool_name') or 'unknown') for kind, job in future_meta.values() if kind == 'tool'))
                    route_snapshot_fn = getattr(model, 'trajectory_route_snapshot', None)
                    route_snapshot = route_snapshot_fn() if callable(route_snapshot_fn) else None
                    if isinstance(route_snapshot, dict):
                        route_snapshot = {**route_snapshot, 'model_inflight_per_replica': list(model_inflight_per_replica), 'model_worker_capacity_per_replica': list(model_worker_capacity_per_replica), 'ready_model_unassigned': sum((job.get('assigned_replica') is None for job, _resume in ready_model)), 'ready_model_per_replica': [sum((job.get('assigned_replica') == index for job, _resume in ready_model)) for index in range(replica_count)]}
                    interval_tool_latency = {name: {'avg_ms': round(interval_tool_latency_ms[name] / count, 3), 'max_ms': round(interval_tool_latency_max_ms.get(name, 0.0), 3)} for name, count in sorted(interval_tool_completed.items()) if count}
                    interval_tool_stages = {name: {stage: {'avg_ms': round(total / max(1, interval_tool_completed[name]), 3), 'max_ms': round(interval_tool_stage_max.get(name, {}).get(stage, 0.0), 3)} for stage, total in sorted(stages.items())} for name, stages in sorted(interval_tool_stage_sum.items())}
                    provider_status_fn = getattr(search, 'provider_status', None)
                    provider_status = provider_status_fn() if callable(provider_status_fn) else None
                    logger.info('Neos scheduler_status active=%d pending_new=%d ready_model=%d model_inflight=%d tool_futures=%d postprocess_futures=%d judge_futures=%d tools_by_name=%s events=%s tool_completed_30s=%s tool_latency_30s=%s tool_stages_30s=%s providers=%s routes=%s', active_trajectories, len(pending_samples), len(ready_model), model_inflight, future_kinds.get('tool', 0), future_kinds.get('postprocess', 0), future_kinds.get('judge', 0) + future_kinds.get('judge_only', 0), json.dumps(dict(sorted(tools_by_name.items()))), json.dumps(dict(sorted(scheduler_events.items()))), json.dumps(dict(sorted(interval_tool_completed.items()))), json.dumps(interval_tool_latency, sort_keys=True), json.dumps(interval_tool_stages, sort_keys=True), json.dumps(provider_status, sort_keys=True), json.dumps(route_snapshot, sort_keys=True))
                    interval_tool_completed.clear()
                    interval_tool_latency_ms.clear()
                    interval_tool_latency_max_ms.clear()
                    interval_tool_stage_sum.clear()
                    interval_tool_stage_max.clear()
                    next_scheduler_status = time.monotonic() + 30.0
    try:
        if args.execution_scheduler == 'turn_event':
            run_turn_event_scheduler()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_judge_only, row): ('judge_only', row) for row in judge_only_existing}
                futures.update({executor.submit(process, sample): ('agent', sample) for sample in pending})
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except FatalLLMGatewayError:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                    persist_completed_row(row)
    finally:
        trajectory_publisher.close()
    phase_timings_ms['parallel_agent_and_judge_evaluation'] = (time.perf_counter() - evaluation_loop_started) * 1000
    rows = existing + completed_rows
    summary = _summary(rows, selected)
    summary_started = time.perf_counter()
    _atomic_json(output_dir / 'summary.json', summary)
    unified_repeat_summary_path: Path | None = None
    if args.repeat_count > 1:
        unified_repeat_summary_path = output_dir / 'unified_repeat_summary.json'
        _atomic_json(unified_repeat_summary_path, _unified_repeat_summary(rows, selected, args.repeat_count))
    combined_summary_path: Path | None = None
    format_retry_summary_path: Path | None = None
    if retry_source_run is not None:
        retry_by_key = {(str(row.get('bench')), str(row.get('sid'))): row for row in rows}
        combined_rows = [retry_by_key.get((str(source.get('bench')), str(source.get('sid'))), source) for source in retry_source_rows]
        source_all_keys = {(str(source.get('bench')), str(source.get('sid'))) for source in retry_source_rows}
        combined_rows.extend((row for row in rows if (str(row.get('bench')), str(row.get('sid'))) not in source_all_keys))
        combined_summary = _summary(combined_rows, selected)
        combined_summary_path = output_dir / 'combined_summary.json'
        _atomic_json(combined_summary_path, combined_summary)
        source_selected = [row for row in retry_source_rows if str(row.get('bench')) in selected]
        combined_selected = [row for row in combined_rows if str(row.get('bench')) in selected]
        source_correct = sum((row.get('correct') is True for row in source_selected))
        combined_correct = sum((row.get('correct') is True for row in combined_selected))
        retry_report = {'schema_version': 'neos-format-retry-summary', 'source_run': str(retry_source_run), 'retry_run': str(output_dir), 'selected_source_format_failures': len(retry_source_keys), 'selected_source_unseen': len(retry_unseen_keys), 'retry_rows_available': len(rows), 'retry_final_answers': sum((row.get('termination_reason') == 'final_answer' for row in rows)), 'retry_format_failures_remaining': sum((_is_format_failure(row) for row in rows)), 'retry_adapted_rows': sum((int(row.get('format_adapted_turns') or 0) > 0 for row in rows)), 'retry_adapter_model_calls': sum((int(row.get('format_recovery_model_calls') or 0) for row in rows)), 'source_correct': source_correct, 'combined_correct': combined_correct, 'correct_delta': combined_correct - source_correct, 'combined_summary_file': str(combined_summary_path.resolve())}
        format_retry_summary_path = output_dir / 'format_retry_summary.json'
        _atomic_json(format_retry_summary_path, retry_report)
    phase_timings_ms['accuracy_summary_write'] = (time.perf_counter() - summary_started) * 1000
    evaluator_wall_ms = (time.perf_counter() - evaluator_started) * 1000
    error_rows_this_invocation = [row for row in completed_rows if row.get('error')]
    skipped_rows_this_invocation = [row for row in completed_rows if row.get('skipped')]
    invocation_record = {'invocation_started_at_utc': invocation_started_utc, 'invocation_finished_at_utc': _utc_now(), 'evaluator_wall_ms': round(evaluator_wall_ms, 3), 'phase_timings_ms': {key: round(float(value), 3) for key, value in sorted(phase_timings_ms.items())}, 'pending_at_start': total_pending_work, 'agent_pending_at_start': len(pending), 'judge_only_pending_at_start': len(judge_only_existing), 'rows_written': len(completed_rows), 'error_sample_attempts': len(error_rows_this_invocation), 'native_context_skips': len(skipped_rows_this_invocation), 'format_failure_rollout_policy': args.format_failure_rollout_policy, 'error_sample_worker_wall_ms': round(sum((float((row.get('evaluation_timing') or {}).get('sample_worker_wall_ms') or 0) for row in error_rows_this_invocation)), 3)}
    all_invocations = [*existing_invocations, invocation_record]
    _atomic_jsonl(invocation_path, all_invocations)
    _atomic_jsonl(persistence_path, [*existing_persistence_events, *persistence_events])
    timing_summary = build_timing_summary(rows, [*existing_persistence_events, *persistence_events], selected=selected, evaluator_wall_ms=evaluator_wall_ms, phase_timings_ms=phase_timings_ms, invocations=all_invocations)
    timing_summary_path = output_dir / 'timing_summary.json'
    _atomic_json(timing_summary_path, timing_summary)
    manifest.update({'status': 'completed' if summary['errors'] == 0 else 'completed_with_errors', 'finished_at_utc': _utc_now(), 'summary_file': str((output_dir / 'summary.json').resolve()), 'combined_summary_file': str(combined_summary_path.resolve()) if combined_summary_path is not None else None, 'format_retry_summary_file': str(format_retry_summary_path.resolve()) if format_retry_summary_path is not None else None, 'unified_repeat_summary_file': str(unified_repeat_summary_path.resolve()) if unified_repeat_summary_path is not None else None, 'timing_summary_file': str(timing_summary_path.resolve()), 'persistence_timing_file': str(persistence_path.resolve()), 'durable_trajectory_publisher': trajectory_publisher.stats(), 'evaluation_invocations_file': str(invocation_path.resolve()), 'evaluator_wall_ms': round(evaluator_wall_ms, 3), 'phase_timings_ms': {key: round(value, 3) for key, value in sorted(phase_timings_ms.items())}})
    _atomic_json(output_dir / 'run_manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
