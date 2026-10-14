from __future__ import annotations
import asyncio
import concurrent.futures as futures
import json
import logging
import os
import socket
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from neos.workflow.credit_transport import CreditControl, CreditDeadlineError, TRANSPORT_VERSION, guarded_credit_call
logger = logging.getLogger(__name__)
_manager = None
_manager_lock = threading.Lock()

def enabled() -> bool:
    return all((os.getenv(name, '0').lower() in {'1', 'true', 'yes', 'on'} for name in ('NEOS_TURN_CREDIT_ENABLED', 'NEOS_TURN_CREDIT_PREFETCH')))

def credit_inputs(fields: dict) -> dict | None:
    pass
    from neos.workflow.turn_credit import _unwrap_mapping
    reward = _unwrap_mapping(fields.get('reward_extra_info'))
    if not fields.get('turn_credit_eligible') or reward.get('judge_parse_failed') or (not fields.get('turn_credit_capture')):
        return None
    if not all(fields.get(key) for key in ('turn_credit_trace', 'policy_turn_spans', 'reference_trajectory_path')):
        return None
    return {'question': str(fields.get('question') or ''), 'ground_truth': str(fields.get('ground_truth') or ''), 'failed_trace': fields['turn_credit_trace'], 'failed_visuals': fields.get('_turn_credit_visuals') or [], 'policy_spans': fields['policy_turn_spans'], 'reference_path': str(fields['reference_trajectory_path']), 'qid': str(fields.get('qid') or ''), 'rollout_request_id': str(fields.get('rollout_request_id') or '')}

def ready_negative_keys(partition: dict, ready_uids: list[str], base_n: int) -> list[str]:
    pass
    by_uid = defaultdict(dict)
    wanted = set(ready_uids)
    for key, tag in partition.items():
        if tag.get('is_prompt') or tag.get('status') != 'success':
            continue
        parts = key.rsplit('_', 2)
        if len(parts) != 3 or parts[0] not in wanted:
            continue
        uid, session, output = (parts[0], int(parts[1]), int(parts[2]))
        old = by_uid[uid].get(session)
        if old is None or output > old[0]:
            by_uid[uid][session] = (output, key, float(tag.get('reward_score') or 0.0))
    result = []
    for uid in ready_uids:
        values = list(by_uid[uid].values())
        if len(values) < base_n:
            continue
        if not any((v[2] > 0.5 for v in values)) or not any((v[2] <= 0.5 for v in values)):
            continue
        result.extend((v[1] for v in sorted(values, key=lambda v: v[1]) if v[2] <= 0.5))
    return result

def _fetch_fields(key: str) -> dict:
    import transfer_queue as tq
    from neos.workflow.turn_credit import _unwrap_mapping
    data = tq.kv_batch_get(keys=[key], partition_id='train', select_fields=['extra_fields'])
    values = data['extra_fields']
    values = values.tolist() if hasattr(values, 'tolist') else list(values)
    return _unwrap_mapping(values[0])

def request_key(key, step, kwargs):
    return json.dumps([os.getenv('RL_JOB_ID', ''), int(step), str(key)], separators=(',', ':'))

class CreditPrefetchManager:

    def __init__(self, *, judge=None, fetch_fields=None, workers=None, max_pending=None, audit_path: Path | None=None):
        from neos.workflow.turn_credit import assign_failed_trajectory_credit
        self.judge = judge or assign_failed_trajectory_credit
        self.fetch_fields = fetch_fields or _fetch_fields
        self.workers = max(1, int(workers or os.getenv('NEOS_TURN_CREDIT_MAX_CONCURRENCY', '64')))
        max_pending = max_pending or int(os.getenv('NEOS_TURN_CREDIT_PREFETCH_MAX_PENDING', '128'))
        self.lock = threading.RLock()
        self.audit_lock = threading.Lock()
        self.api_pool = futures.ThreadPoolExecutor(self.workers, thread_name_prefix='credit-prefetch-api')
        self.load_pool = futures.ThreadPoolExecutor(2, thread_name_prefix='credit-prefetch-load')
        self.prefetch_slots = threading.BoundedSemaphore(max(1, max_pending))
        self.load_slots = threading.BoundedSemaphore(max(1, max_pending))
        self.entries: dict[str, dict] = {}
        self.offered: set[tuple[int, str]] = set()
        self.selection: dict[int, set[str]] = {}
        self.counters = Counter()
        self.current_step = 0
        self.closed = False
        if audit_path is None and os.getenv('NEOS_TURN_CREDIT_AUDIT_ROOT'):
            root = Path(os.environ['NEOS_TURN_CREDIT_AUDIT_ROOT']).parent / 'credit_prefetch_audits'
            audit_path = root / f'prefetch-{socket.gethostname()}-pid{os.getpid()}.jsonl'
        self.audit_path = audit_path
        self._emit('manager_started', workers=self.workers, max_pending=max_pending)

    def _emit(self, event: str, **values) -> None:
        if self.audit_path is None:
            return
        row = {'schema_version': 'neos-credit-prefetch', 'event': event, 'created_at_unix': time.time(), 'job_id': os.getenv('RL_JOB_ID', ''), **values}
        try:
            with self.audit_lock:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        except OSError:
            logger.exception('credit prefetch audit write failed')

    def offer(self, keys: list[str], step: int) -> None:
        with self.lock:
            if self.closed or step in self.selection:
                return
            self.current_step = step
            for key in keys:
                ticket = (step, key)
                if ticket in self.offered:
                    continue
                if not self.load_slots.acquire(blocking=False):
                    self.counters['load_queue_full'] += 1
                    break
                self.offered.add(ticket)
                task = self.load_pool.submit(self._load, key, step)
                task.add_done_callback(lambda _task: self.load_slots.release())

    def _load(self, key: str, step: int) -> None:
        retry_later = False
        try:
            with self.lock:
                if self.closed or (step in self.selection and key not in self.selection[step]):
                    self.counters['cancelled_loads'] += 1
                    return
            fields = self.fetch_fields(key)
            kwargs = credit_inputs(fields)
            if kwargs is None:
                with self.lock:
                    self.counters['ineligible_prefetch_rows'] += 1
                return
            entry = self.submit(key, step, kwargs, source='prefetch')
            retry_later = entry is None
        except Exception as exc:
            with self.lock:
                self.counters['prefetch_read_errors'] += 1
            self._emit('load_failed', step=step, trajectory_key=key, error=f'{type(exc).__name__}: {exc}'[:1000])
            retry_later = True
        finally:
            if retry_later:
                with self.lock:
                    self.offered.discard((step, key))

    def submit(self, key: str, step: int, kwargs: dict, *, source: str) -> dict | None:
        digest = request_key(key, step, kwargs)
        with self.lock:
            old = self.entries.get(digest)
            if old is not None and (not old['future'].cancelled()):
                return old
            if self.closed:
                raise RuntimeError('credit prefetch manager is closed')
            speculative = source == 'prefetch'
            if speculative and step in self.selection and (key not in self.selection[step]):
                return None
            if speculative and (not self.prefetch_slots.acquire(blocking=False)):
                self.counters['api_queue_full'] += 1
                return None
            entry = {'cache_key': digest, 'trajectory_key': key, 'step': step, 'source': source, 'qid': kwargs.get('qid'), 'rollout_request_id': kwargs.get('rollout_request_id'), 'submitted_at': time.time(), 'started_at': None, 'done_at': None, 'cancelled_before_api': False, 'consumed': False, 'future': None, 'control': CreditControl(), 'timed_out': False}
            self.entries[digest] = entry
            task = self.api_pool.submit(self._execute, entry, kwargs)
            entry['future'] = task
            task.add_done_callback(lambda task: self._done(entry, task, speculative))
            return entry

    def _execute(self, entry: dict, kwargs: dict) -> dict | BaseException:
        with self.lock:
            step, key = (entry['step'], entry['trajectory_key'])
            if entry['source'] == 'prefetch' and step in self.selection and (key not in self.selection[step]):
                entry['cancelled_before_api'] = True
                return RuntimeError('unused speculative credit cancelled before API call')
            entry['started_at'] = time.time()
            entry['control'].start()
        identity = {k: entry[k] for k in ('cache_key', 'trajectory_key', 'step', 'source', 'qid', 'rollout_request_id')}
        self._emit('judge_started', **identity)
        try:
            return asyncio.run(guarded_credit_call(self.judge, kwargs, entry['control']))
        except Exception as exc:
            error_type = CreditDeadlineError if isinstance(exc, TimeoutError) else RuntimeError
            return error_type(f'{type(exc).__name__}: {exc}')

    def _done(self, entry: dict, task: futures.Future, speculative: bool) -> None:
        if speculative:
            self.prefetch_slots.release()
        with self.lock:
            entry['done_at'] = time.time()
            if task.cancelled():
                entry['cancelled_before_api'] = True
                result = None
            else:
                result = task.result()
            selected = entry['trajectory_key'] in self.selection.get(entry['step'], set())
            timed_out = entry['timed_out'] or entry['control'].cancelled.is_set()
        identity = {k: entry[k] for k in ('cache_key', 'trajectory_key', 'step', 'source', 'qid', 'rollout_request_id')}
        self._emit('judge_finished', **identity, selected_at_completion=selected, selection_known=entry['step'] in self.selection, cancelled_before_api=entry['cancelled_before_api'], applied=bool(result.get('applied')) if isinstance(result, dict) and (not timed_out) else False, credit_audit_path=result.get('audit_path') if isinstance(result, dict) else None, error=str(result)[:1500] if isinstance(result, BaseException) else None, timed_out=timed_out, discarded_late_result=bool(timed_out and isinstance(result, dict)), last_provider=entry['control'].last_response if timed_out else None)

    def close_step(self, step: int, selected_keys: list[str]) -> None:
        with self.lock:
            self.current_step = step
            self.selection[step] = set(selected_keys)
            for entry in list(self.entries.values()):
                if entry['step'] == step and entry['trajectory_key'] not in self.selection[step]:
                    entry['future'].cancel()
        self._emit('selection_closed', step=step, selected_trajectory_keys=selected_keys)

    def resolve(self, jobs: list[tuple[int, dict]], batch_keys: list[str]):
        started = time.monotonic()
        results = []
        pending = []
        hits = Counter()
        for row_index, kwargs in jobs:
            entry = self.submit(batch_keys[row_index], self.current_step, kwargs, source='selected')
            assert entry is not None
            task = entry['future']
            with self.lock:
                entry['consumed'] = True
            if entry['source'] == 'prefetch':
                hits['cache_ready_hits' if task.done() else 'cache_inflight_hits'] += 1
            else:
                hits['on_demand_calls'] += 1
            pending.append((row_index, task, entry))
        wait_started = time.monotonic()
        batch_seconds = float(os.getenv('NEOS_TURN_CREDIT_BATCH_WAIT_TIMEOUT', '1800'))
        if batch_seconds <= 0:
            raise ValueError('credit batch wait timeout must be positive')
        unresolved = {row_index: (task, entry) for row_index, task, entry in pending}
        resolved = {}
        timed_out_rows = 0
        while unresolved:
            futures.wait([task for task, _ in unresolved.values()], timeout=0.2, return_when=futures.FIRST_COMPLETED)
            for row_index, (task, entry) in list(unresolved.items()):
                if task.done():
                    try:
                        resolved[row_index] = task.result()
                    except BaseException as exc:
                        resolved[row_index] = RuntimeError(str(exc))
                    if isinstance(resolved[row_index], TimeoutError):
                        timed_out_rows += 1
                    unresolved.pop(row_index)
                    continue
                control = entry['control']
                if control.started is not None and control.remaining() <= 0 or time.monotonic() - wait_started >= batch_seconds:
                    reason = 'credit trajectory deadline exceeded' if control.started is not None and control.remaining() <= 0 else 'selected credit batch wait deadline exceeded'
                    entry['timed_out'] = True
                    control.cancel(reason)
                    task.cancel()
                    resolved[row_index] = CreditDeadlineError(reason)
                    timed_out_rows += 1
                    unresolved.pop(row_index)
                    self._emit('selected_timeout_fail_open', step=self.current_step, trajectory_key=entry['trajectory_key'], reason=reason)
        results = [(row_index, resolved[row_index]) for row_index, _, _ in pending]
        metrics = self.metrics(self.current_step)
        metrics.update({f'turn_credit/prefetch/{k}': float(v) for k, v in hits.items()})
        metrics['turn_credit/prefetch/tail_wait_seconds'] = time.monotonic() - wait_started
        metrics['turn_credit/prefetch/resolve_seconds'] = time.monotonic() - started
        metrics['turn_credit/prefetch/timeout_fail_open_rows'] = float(timed_out_rows)
        logger.info('credit_prefetch_resolved step=%s metrics=%s', self.current_step, metrics)
        with self.lock:
            consumed = [e['cache_key'] for e in self.entries.values() if e['step'] == self.current_step and e['consumed']]
        self._emit('selected_resolved', step=self.current_step, metrics=metrics, consumed_cache_keys=consumed)
        return (results, metrics)

    def metrics(self, step: int) -> dict[str, float]:
        with self.lock:
            entries = [e for e in self.entries.values() if e['step'] == step]
            values = {'offered': sum((s == step for s, _k in self.offered)), 'scheduled': len(entries), 'prefetch_scheduled': sum((e['source'] == 'prefetch' for e in entries)), 'api_started': sum((e['started_at'] is not None for e in entries)), 'cancelled_before_api': sum((e['cancelled_before_api'] for e in entries)), 'unused_api_started': sum((e['started_at'] is not None and (not e['consumed']) for e in entries)), 'timeout_cancelled': sum((e['timed_out'] or e['control'].cancelled.is_set() for e in entries)), 'pending': sum((not e['future'].done() for e in entries))}
        return {f'turn_credit/prefetch/{k}': float(v) for k, v in values.items()}

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True
            tasks = []
            for entry in self.entries.values():
                if not entry['future'].done():
                    entry['control'].cancel('credit manager shutdown')
                    entry['future'].cancel()
                    tasks.append(entry['future'])
        if tasks:
            futures.wait(tasks, timeout=15)
        self.load_pool.shutdown(wait=False, cancel_futures=True)
        self.api_pool.shutdown(wait=False, cancel_futures=True)
        self._emit('manager_finished', steps={str(s): self.metrics(s) for s in self.selection}, counters=dict(self.counters))

def get_manager() -> CreditPrefetchManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = CreditPrefetchManager()
        return _manager

def offer_ready(partition: dict, ready_uids: list[str], base_n: int, step: int) -> None:
    if enabled() and ready_uids:
        keys = ready_negative_keys(partition, ready_uids, base_n)
        if keys:
            get_manager().offer(keys, step)

def close_step(step: int, selected_keys: list[str]) -> None:
    if enabled():
        get_manager().close_step(step, selected_keys)

def shutdown() -> None:
    global _manager
    with _manager_lock:
        manager, _manager = (_manager, None)
    if manager is not None:
        manager.shutdown()
