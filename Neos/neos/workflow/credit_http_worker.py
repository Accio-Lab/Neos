
import ctypes
import json
import os
import signal
import sys
import threading
import time

if __package__:
    from .credit_model_routing import resolve_request_model, append_route_audit
else:
    from credit_model_routing import resolve_request_model, append_route_audit


def emit_json(event):
    print(json.dumps(event, ensure_ascii=False, separators=(',', ':')), flush=True)


def consume_and_close(client, kwargs, emit=emit_json, close_guard_seconds=3, model_routing=None, tenant_routing=None):
    stream = None
    content = []; reasoning = []; pending_content = []; pending_reasoning = []
    usage = {}; response_id = None; response_model = None; finish_reason = None; error = None
    route = model_routing or {'configured_model': kwargs.get('model'), 'request_model': kwargs.get('model'),
                              'route_applied': False, 'route_version': None}
    started = last_emit = time.monotonic()
    size = 0
    guard = None
    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            payload = chunk.model_dump(exclude_none=False) if hasattr(chunk, 'model_dump') else chunk
            if payload.get('success') is False:
                raise RuntimeError('qwen credit judge returned success=false: '+str(payload.get('message') or payload)[:4000])
            response_id = payload.get('id') or response_id
            response_model = payload.get('model') or response_model
            if payload.get('usage'): usage = payload['usage']
            for choice in (payload.get('choices') or [])[:1]:
                finish_reason = choice.get('finish_reason') or finish_reason
                delta = choice.get('delta') or {}
                c = str(delta.get('content') or ''); r = str(delta.get('reasoning_content') or '')
                if c: content.append(c); pending_content.append(c)
                if r: reasoning.append(r); pending_reasoning.append(r)
                size += len(c) + len(r)
            if size > 4_000_000:
                raise RuntimeError('credit response exceeded the bounded output buffer')
            if time.monotonic() - last_emit >= 1:
                emit(dict(event='delta', content=''.join(pending_content), reasoning_content=''.join(pending_reasoning),
                          usage=usage, response_id=response_id, finish_reason=finish_reason))
                pending_content.clear(); pending_reasoning.clear(); last_emit=time.monotonic()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:4000]
    finally:
        result = dict(content=''.join(content).strip(), reasoning_content=''.join(reasoning).strip(),
                      usage=usage, response_id=response_id, finish_reason=finish_reason,
                      transport_error=error, elapsed_seconds=round(time.monotonic()-started, 3),
                      usage_complete=bool(usage), configured_model=route.get('configured_model'),
                      request_model=kwargs.get('model'), response_model=response_model,
                      actual_model=response_model or kwargs.get('model'),
                      model_routing={k:v for k,v in route.items() if k!='audit_root'},
                      tenant_routing=dict(tenant_routing or {}))
        
        
        emit(dict(event='result', result=result))
        if close_guard_seconds is not None:
            def force_close():
                emit(dict(event='cleanup_forced'))
                os._exit(0)
            guard=threading.Timer(close_guard_seconds, force_close);guard.daemon=True;guard.start()
        try:
            
            
            append_route_audit(route, 'request_finished', response_id=response_id,
                               response_model=response_model, actual_model=result['actual_model'],
                               finish_reason=finish_reason, usage=usage,
                               transport_error=error, elapsed_seconds=result['elapsed_seconds'])
            if stream is not None and callable(getattr(stream, 'close', None)): stream.close()
        finally:
            try:
                if callable(getattr(client, 'close', None)): client.close()
            finally:
                if guard is not None: guard.cancel()
    return result


def main():
    
    
    expected_parent = int(os.environ.get('NEOS_CREDIT_PARENT_PID', '0'))
    if sys.platform == 'linux' and expected_parent:
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != expected_parent: return
    seconds = float(os.environ.get('NEOS_CREDIT_CHILD_TIMEOUT_S', '600'))
    def expire():
        emit_json(dict(event='deadline'))
        os._exit(124)
    timer=threading.Timer(seconds, expire);timer.daemon=True;timer.start()
    try:
        request=json.load(sys.stdin)
        kwargs, routing=resolve_request_model(request['kwargs'])
        audit_error=append_route_audit(routing, 'request_started')
        if audit_error:routing['routing_audit_start_error']=audit_error
        import httpx
        import openai
        auth=dict(request['auth'])
        tenant_routing={'credential_source':'explicit_configuration'}
        client=openai.OpenAI(api_key=auth['api_key'], base_url=auth['base_url'],
            default_headers=auth['headers'], max_retries=0,
            timeout=httpx.Timeout(connect=30, read=300, write=120, pool=30))
        consume_and_close(client, kwargs, close_guard_seconds=3, model_routing=routing,
                          tenant_routing=tenant_routing)
    except Exception as exc:
        emit_json(dict(event='worker_error', error=f'{type(exc).__name__}: {exc}'[:4000]))
    finally:
        timer.cancel()


if __name__ == '__main__': main()
