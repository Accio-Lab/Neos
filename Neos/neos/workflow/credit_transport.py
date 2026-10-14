
import asyncio
import contextvars
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

TRANSPORT_VERSION = 'isolated-http-stream-deadline'
_control = contextvars.ContextVar('neos_credit_control', default=None)


class CreditDeadlineError(TimeoutError):
    pass


class CreditControl:
    def __init__(self, total_seconds=None):
        self.total_seconds = float(total_seconds if total_seconds is not None else
                                   os.getenv('NEOS_TURN_CREDIT_TOTAL_TIMEOUT', '1200'))
        if self.total_seconds <= 0: raise ValueError('credit total timeout must be positive')
        self.started = None
        self.cancelled = threading.Event()
        self.reason = None
        self.lock = threading.RLock()
        self.processes = set()
        self.last_response = None
        self.guarded = False

    def start(self):
        with self.lock:
            if self.started is None: self.started = time.monotonic()

    def remaining(self):
        return self.total_seconds if self.started is None else self.started + self.total_seconds - time.monotonic()

    def check(self):
        if self.cancelled.is_set() or self.remaining() <= 0:
            raise CreditDeadlineError(self.reason or 'credit trajectory deadline exceeded')

    def bind(self, process):
        with self.lock:
            self.processes.add(process)
            if self.cancelled.is_set(): self._kill(process)

    @staticmethod
    def _kill(process):
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass

    def cancel(self, reason='credit cancelled'):
        with self.lock:
            self.reason = self.reason or reason
            self.cancelled.set()
            for process in self.processes: self._kill(process)


def current_control():
    return _control.get()


async def guarded_credit_call(call, kwargs, control=None):
    control = control or current_control() or CreditControl()
    if control.guarded:
        control.check()
        return await call(**kwargs)
    control.start();control.check();control.guarded=True
    token=_control.set(control)
    loop=asyncio.get_running_loop()
    
    timer=loop.call_later(max(0, control.remaining()), control.cancel, 'credit trajectory deadline exceeded')
    try:
        result=await asyncio.wait_for(call(**kwargs), timeout=max(.001,control.remaining()))
        control.check()
        return result
    except asyncio.CancelledError:
        control.cancel('credit coroutine cancelled');raise
    except TimeoutError as exc:
        control.cancel('credit trajectory deadline exceeded')
        raise CreditDeadlineError(control.reason) from exc
    finally:
        timer.cancel();control.guarded=False;_control.reset(token)


def run_credit_worker(request, *, timeout=None, worker_command=None):
    control=current_control() or CreditControl()
    control.start();control.check()
    limit=float(timeout if timeout is not None else os.getenv('NEOS_TURN_CREDIT_REQUEST_TIMEOUT','600'))
    if limit <= 0: raise ValueError('credit request timeout must be positive')
    limit=min(limit,control.remaining())
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NEOS_CREDIT_PARENT_PID':str(os.getpid()),
         'NEOS_CREDIT_CHILD_TIMEOUT_S':str(limit)}
    command=worker_command or [sys.executable, str(Path(__file__).with_name('credit_http_worker.py'))]
    started=time.monotonic()
    process=subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, env=env)
    control.bind(process)
    deadline_hit=False;cleanup_forced=False
    try:
        try:
            output,errors=process.communicate(json.dumps(request,ensure_ascii=False).encode(), timeout=limit+5)
        except subprocess.TimeoutExpired as exc:
            deadline_hit=True;CreditControl._kill(process)
            try: output,errors=process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                output=exc.output or b'';errors=exc.stderr or b''
                
                
        result=None;content=[];reasoning=[];usage={};response_id=None;finish=None;protocol_error=None
        for line in output.splitlines():
            try: frame=json.loads(line)
            except ValueError: continue
            if frame.get('event')=='delta':
                content.append(frame.get('content',''));reasoning.append(frame.get('reasoning_content',''))
                usage=frame.get('usage') or usage;response_id=frame.get('response_id') or response_id
                finish=frame.get('finish_reason') or finish
            elif frame.get('event')=='result': result=frame['result']
            elif frame.get('event')=='deadline': deadline_hit=True
            elif frame.get('event')=='cleanup_forced': cleanup_forced=True
            elif frame.get('event')=='worker_error': protocol_error=frame.get('error')
        if result is None:
            error=(control.reason if control.cancelled.is_set() else
                   'credit HTTP request deadline exceeded' if deadline_hit or process.returncode==124 else
                   protocol_error or f'credit HTTP worker exited without result (exit={process.returncode})')
            result=dict(content=''.join(content),reasoning_content=''.join(reasoning),usage=usage,
                        response_id=response_id,finish_reason=finish,transport_error=error,
                        usage_complete=bool(usage))
        result.update(worker_pid=process.pid,worker_reaped=process.poll() is not None,
                      cleanup_forced=cleanup_forced,transport_version=TRANSPORT_VERSION,
                      supervisor_elapsed_seconds=round(time.monotonic()-started,3),
                      timeout_kind=('trajectory_cancelled' if control.cancelled.is_set() else
                                    'request_deadline' if deadline_hit or process.returncode==124 else None))
        with control.lock: control.last_response=result
        if control.cancelled.is_set(): result['transport_error']=control.reason
        return result
    finally:
        CreditControl._kill(process)
        if process.poll() is None:
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
        with control.lock: control.processes.discard(process)
