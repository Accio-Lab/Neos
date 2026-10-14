
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import requests
from neos.paths import PROJECT_ROOT, project_path
from neos.configuration import required_setting, output_path


def commands(args):
    groups=[g.split(',') for g in args.gpu_groups.split(';')]
    flat=[gpu for group in groups for gpu in group]
    if not groups or any(not x.isdigit() for x in flat) or len(set(flat))!=len(flat):
        raise ValueError('GPU groups must contain distinct numeric device IDs')
    result=[]
    for i,group in enumerate(groups):
        argv=[sys.executable,'-m','sglang.launch_server','--model-path',str(project_path(args.model)),
            '--served-model-name',args.name,'--host','127.0.0.1','--port',str(args.backend_port+i),
            '--tp',str(len(group)),'--context-length',str(args.context_length),
            '--max-running-requests',str(args.max_running_requests),
            '--mem-fraction-static',str(args.memory_fraction)]
        if args.kind=='policy':argv+=['--reasoning-parser','qwen3','--tool-call-parser','qwen3_coder']
        result.append((','.join(group),argv))
    return result


def stop_owned(children):
    processes=[]
    for child in children:
        try:
            p=psutil.Process(child.pid);processes+=p.children(recursive=True)+[p]
        except psutil.NoSuchProcess:continue
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    _,alive=psutil.wait_procs(list(set(processes)),timeout=20)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass
    for child in children:
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['policy','mei'],required=True)
    p.add_argument('--model');p.add_argument('--name')
    p.add_argument('--gpu-groups',help='For example 0,1;2,3;4,5;6,7')
    p.add_argument('--context-length',type=int)
    p.add_argument('--max-running-requests',type=int,default=16)
    p.add_argument('--memory-fraction',type=float)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int)
    p.add_argument('--backend-port',type=int);p.add_argument('--log-dir')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();mei=a.kind=='mei'
    a.model=a.model or required_setting('NEOS_MEI_MODEL' if mei else 'NEOS_SERVE_POLICY_MODEL')
    a.log_dir=a.log_dir or str(output_path('services'))
    a.name=a.name or ('neos-mei' if mei else os.getenv('NEOS_SERVE_POLICY_NAME','neos-policy'))
    a.gpu_groups=a.gpu_groups or ('0;1;2;3;4;5;6;7' if mei else '0,1;2,3;4,5;6,7')
    a.context_length=a.context_length or (32768 if mei else 262144)
    a.memory_fraction=a.memory_fraction or (.78 if mei else .60)
    a.port=a.port or (48080 if mei else 62080)
    a.backend_port=a.backend_port or (48000 if mei else 62000)
    if a.max_running_requests<1 or not 0<a.memory_fraction<1:raise ValueError('Invalid memory/concurrency setting')
    specs=commands(a)
    proxy=[sys.executable,'-m','neos.serving.proxy','--host',a.host,'--port',str(a.port),
           '--timeout','600','--max-body-mib','128']
    for i in range(len(specs)):proxy+=['--backend',f'http://127.0.0.1:{a.backend_port+i}']
    if a.dry_run:
        print(json.dumps({'replicas':specs,'proxy':proxy},indent=2));return
    if not (project_path(a.model)/'config.json').is_file():raise FileNotFoundError(project_path(a.model))
    selected={int(x) for group,_ in specs for x in group.split(',')}
    uuids={line.split(',')[1].strip():int(line.split(',')[0]) for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
    if not selected.issubset(set(uuids.values())):raise ValueError('Selected GPU does not exist')
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(uuids.get(line.split(',')[0].strip()) in selected for line in used.splitlines()):
        raise RuntimeError('Selected GPUs already have a workload; nothing was stopped')
    logs=project_path(a.log_dir)/a.name;logs.mkdir(parents=True,exist_ok=True)
    children=[];streams=[]
    def spawn(cmd,path,env=None):
        stream=path.open('ab');streams.append(stream)
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        children.append(child)
    def stop(sig,frame):raise SystemExit(128+sig)
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    session=requests.Session();session.trust_env=False
    try:
        for i,(devices,argv) in enumerate(specs):
            spawn(argv,logs/f'backend-{i}.log',dict(os.environ,CUDA_VISIBLE_DEVICES=devices))
        pending=set(range(len(specs)));deadline=time.monotonic()+1200
        while pending:
            if any(c.poll() is not None for c in children):raise RuntimeError('A model replica failed during startup')
            for i in list(pending):
                try:
                    data=session.get(f'http://127.0.0.1:{a.backend_port+i}/v1/models',timeout=2).json()
                    if a.name in {m['id'] for m in data['data']}:pending.remove(i)
                except (requests.RequestException,ValueError,KeyError):pass
            if time.monotonic()>deadline:raise TimeoutError('Model replicas did not become ready')
            if pending:time.sleep(2)
        spawn(proxy,logs/'proxy.log')
        print(json.dumps({'ready':True,'model':a.name,'endpoint':f'http://{a.host}:{a.port}/v1','logs':str(logs)}),flush=True)
        while all(c.poll() is None for c in children):time.sleep(2)
        raise RuntimeError('A serving process exited; see service logs')
    finally:
        stop_owned(children);session.close()
        for stream in streams:stream.close()


if __name__=='__main__':main()
