#!/usr/bin/env python3

from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import psutil
import requests

from neos.paths import PROJECT_ROOT
REPO=PROJECT_ROOT
sys.path.insert(0,str(REPO))
from neos.workflow.coevolution import atomic_json, read_json

class MEINode:
    def __init__(self, config):
        self.config=config;self.root=Path(config['root'])
        self.children=[]
        self.version=int(config.get('initial_mei_version',0))
        self.after=int(config.get('initial_mei_after_step',0))
        if self.version<0 or self.after<0:
            raise ValueError('Initial MEI version/step must be nonnegative')
        self.model_path=Path(config['initial_mei'])
        self.model_name=self.name(self.version)
        self.train_tp=int(config.get('mei_tensor_parallel_size',2))
        if self.train_tp not in {1,2,4,8}:
            raise ValueError('MEI training TP must divide the dedicated eight GPUs')
        self.train_dp=8//self.train_tp
        if int(config.get('mei_data_parallel_size',self.train_dp))!=self.train_dp:
            raise ValueError('MEI training TP*DP must equal eight (CP=PP=1)')
        self.train_mb=int(config.get('mei_micro_batch_size',1))
        gbs=int(config.get('mei_global_batch',16))
        fallback=[int(value) for value in config.get('mei_micro_batch_fallbacks',[])]
        if (self.train_mb<1 or gbs<1 or gbs%(self.train_dp*self.train_mb)
            or any(m<1 or m>=self.train_mb or gbs%(self.train_dp*m) for m in fallback)):
            raise ValueError('MEI global batch must be divisible by DP * micro-batch; fallback batches must be smaller')
        self.telemetry_stop=threading.Event();self.telemetry_thread=None
        self.addresses=[f'http://127.0.0.1:{config.get("backend_base_port",48000)+i}' for i in range(8)]
        self.owns_gpus=False
        self.api=requests.Session();self.api.trust_env=False
        self.restore_completed_update()

    def restore_completed_update(self):
        pass
        for path in sorted((self.root/'job_results').glob('after-step-*.json')):
            result=read_json(path,{})
            if result.get('status') not in {'trained','skipped_insufficient_rows'}:
                continue
            step=int(path.stem.rsplit('-',1)[-1])
            if result['status']=='trained':
                checkpoint=Path(result['checkpoint'])
                if not (checkpoint/'model.safetensors.index.json').is_file():
                    raise RuntimeError(f'Committed MEI checkpoint is unavailable: {checkpoint}')
                self.version=int(result['version']);self.model_path=checkpoint
                self.model_name=self.name(self.version)
            self.after=step

    def pending_jobs(self):
        return [p for p in sorted((self.root/'jobs').glob('after-step-*.json'))
                if int(read_json(p)['after_policy_step'])>self.after]

    def name(self,v):
        return self.config.get('mei_name_prefix','coev-mei')+f'-h{v:03d}'

    def publish(self,status,**extra):
        value={'status':status,'version':self.version,
          'model':'local/'+self.model_name,'model_path':str(self.model_path),
          'after_policy_step':self.after,**extra}
        
        
        if getattr(self, '_published_state', None) == value:
            return
        atomic_json(self.root/'serving.json',{**value, 'at_unix':time.time()})
        self._published_state=value

    def log(self,event,**fields):
        print(json.dumps({'event':event,'at_unix':time.time(),**fields}),flush=True)

    def start_gpu_telemetry(self):
        if not self.config.get('mei_gpu_telemetry',False):return
        path=self.root/'telemetry'/f'mei-{socket.gethostname()}-pid{os.getpid()}.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        def observe():
            while not self.telemetry_stop.is_set():
                row={'at_unix':time.time(),'host':socket.gethostname(),
                     'mei_state':getattr(self,'_published_state',{}),
                     'policy_state':read_json(self.root/'policy.json',{})}
                try:
                    output=subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader,nounits'],text=True,timeout=5)
                    row['gpus']=[dict(zip(['index','utilization_pct','memory_mib','power_w'],
                        [float(v.strip()) for v in line.split(',')]))
                        for line in output.splitlines() if line.strip()]
                except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
                try:
                    with path.open('a') as f:f.write(json.dumps(row)+'\n')
                except OSError as exc:self.log('mei_gpu_telemetry_write_error',error=str(exc))
                self.telemetry_stop.wait(5)
        self.telemetry_thread=threading.Thread(target=observe,name='mei-gpu-telemetry',daemon=True)
        self.telemetry_thread.start()


    def spawn(self, argv, logfile, env=None):
        logfile.parent.mkdir(parents=True,exist_ok=True)
        with logfile.open('a') as output:
            p=subprocess.Popen(argv,env={**os.environ,**(env or {})},stdout=output,
                               stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(p)
        return p

    def stop_children(self):
        processes=[]
        for child in self.children:
            with contextlib.suppress(psutil.NoSuchProcess):
                p=psutil.Process(child.pid)
                processes.extend(p.children(recursive=True));processes.append(p)
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
        _,alive=psutil.wait_procs(list(set(processes)),timeout=25)
        for p in alive:
            with contextlib.suppress(psutil.NoSuchProcess):p.kill()
        for child in self.children:
            with contextlib.suppress(subprocess.TimeoutExpired):child.wait(timeout=5)
        self.children=[]
        
        for _ in range(15):
            out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            if not any(x.strip().isdigit() for x in out.splitlines()):return
            time.sleep(1)
        raise RuntimeError('GPU context remains after stopping owned MEI/SFT processes')



    def serve(self):
        self.publish('loading')
        version_dir=self.root/'mei_versions'/f'h{self.version:03d}'
        for gpu in range(8):
            port=self.config.get('backend_base_port',48000)+gpu
            argv=['bash','-c',f'source {REPO}/environment/activate.sh\nexec "$@"','coev-serve',
                  sys.executable,'-m','sglang.launch_server',
                  '--model-path',str(self.model_path),'--served-model-name',self.model_name,
                  '--host','127.0.0.1','--port',str(port),'--tp','1',
                  '--context-length','32768','--max-running-requests','16',
                  '--cuda-graph-max-bs','16','--mem-fraction-static','0.78']
            self.spawn(argv,version_dir/f'server-{gpu}.log',{'CUDA_VISIBLE_DEVICES':str(gpu),
                'TRITON_CACHE_DIR':str(Path('/tmp')/self.root.name/'mei-triton'/str(gpu)),
                'TORCHINDUCTOR_CACHE_DIR':str(Path('/tmp')/self.root.name/'mei-inductor'/str(gpu))})
        deadline=time.monotonic()+1200
        pending=set(self.addresses)
        while pending:
            if any(p.poll() is not None for p in self.children):raise RuntimeError('MEI server failed during load')
            for address in list(pending):
                try:
                    r=self.api.get(address+'/health',timeout=2)
                    if r.status_code==200:
                        data=self.api.get(address+'/v1/models',timeout=2).json()
                        if self.model_name in {x.get('id') for x in data.get('data',[])}:pending.remove(address)
                except requests.RequestException:pass
            if time.monotonic()>deadline:raise TimeoutError('MEI startup timeout')
            if pending:time.sleep(2)
        self.log('mei_backends_ready',backends=8)
        proxy=[sys.executable,str(REPO/'neos/serving/proxy.py'),
               '--host','0.0.0.0','--port',str(self.config.get('service_port',48080)),
               '--timeout','600','--max-body-mib','128']
        for address in self.addresses:proxy+=['--backend',address]
        self.spawn(proxy,version_dir/'proxy.log')
        for _ in range(30):
            try:
                r=self.api.get(f'http://127.0.0.1:{self.config.get("service_port",48080)}/health',timeout=20)
                if r.ok and r.json().get('healthy_backends')==8:break
            except requests.RequestException:pass
            time.sleep(1)
        else:raise RuntimeError('MEI proxy did not become ready')
        self.publish('waiting_policy')
        self.log('mei_version_deployed',version=self.version,model=self.model_name,after_step=self.after)

    def train(self,job):
        epochs=float(job.get('epochs',1))
        if not epochs.is_integer() or epochs<1:
            raise ValueError('MEI SFT requires a positive integer epoch count')
        epochs=int(epochs)
        self.publish('training',update_after_step=job['after_policy_step'])
        self.stop_children()
        data=job['data'];step=int(job['after_policy_step'])
        rows=int(data['train_rows']);gbs=int(self.config.get('mei_global_batch',16))
        if rows<gbs:
            self.after=step
            atomic_json(self.root/'job_results'/f'after-step-{step:06d}.json',
                        {'status':'skipped_insufficient_rows','rows':rows,'required':gbs})
            self.serve();return
        output=self.root/'mei_training'/f'after-step-{step}'/'weights'
        log=self.root/'mei_training'/f'after-step-{step}'/'train.log'
        args=[sys.executable,'-m','torch.distributed.run',
              '--nnodes=1','--nproc-per-node=8','--master-addr=127.0.0.1','--master-port=29771',
              '--module','swift.cli._megatron.sft',
              '--model',str(self.model_path),'--model_type','qwen3_5',
              '--external_plugins',str(REPO/'plugins/qwen35_neos_native_tools.py'),
              '--dataset',data['train_file'],'--output_dir',str(output),
              '--tuner_type','full','--freeze_llm','false','--freeze_vit','true','--freeze_aligner','true',
              '--torch_dtype','bfloat16','--bf16','true','--template','qwen3_5','--agent_template','qwen3_5',
              '--max_length','32768','--max_pixels','786432','--truncation_strategy','delete',
              '--padding_free','false','--packing','false','--group_by_length','true',
              '--split_dataset_ratio','0','--strict','true','--loss_scale','default','--enable_thinking','false',
              '--tensor_model_parallel_size',str(self.train_tp),'--context_parallel_size','1','--pipeline_model_parallel_size','1',
              '--expert_model_parallel_size','1','--expert_tensor_parallel_size','1','--sequence_parallel','true',
              '--micro_batch_size',str(self.train_mb),'--global_batch_size',str(gbs),
              '--num_train_epochs',str(epochs),
              '--lr',str(job['learning_rate']),'--min_lr',str(job['learning_rate']/10),
              '--lr_decay_style','cosine','--lr_warmup_fraction','0.05',
              '--weight_decay','0.1','--clip_grad','1.0',
              '--attention_backend','flash','--cross_entropy_loss_fusion','true',
              '--gradient_accumulation_fusion','false','--use_distributed_optimizer','true',
              '--optimizer_cpu_offload','false',
              '--recompute_granularity','full','--recompute_method','uniform','--recompute_num_layers','1',
              '--finetune','true','--dataset_num_proc',str(self.config.get('mei_dataset_num_proc',8)),
              '--dataloader_num_workers',str(self.config.get('mei_dataloader_num_workers',2)),
              '--train_dataloader_shuffle','true','--logging_steps','1','--report_to','tensorboard',
              '--save_strategy','steps','--save_steps','1000000000','--save_total_limit','2',
              '--save_safetensors','true','--no_save_optim','true','--no_save_rng','true',
              '--manual_gc','true','--manual_gc_steps',str(self.config.get('mei_manual_gc_steps',1)),
              '--nccl_comm_warmup','true']
        env={'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','NCCL_IB_DISABLE':'1','NCCL_SOCKET_IFNAME':'lo','GLOO_SOCKET_IFNAME':'lo',
             'NNODES':'1','NPROC_PER_NODE':'8','NODE_RANK':'0','IMAGE_MAX_TOKEN_NUM':'1728',
             'USE_MCORE_GDN':'1','CUDA_DEVICE_MAX_CONNECTIONS':'1','TOKENIZERS_PARALLELISM':'false',
             'HF_DATASETS_CACHE':str(Path('/tmp')/self.root.name/f'mei-datasets-{step}'),
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
        batches=list(dict.fromkeys([self.train_mb,*[int(v) for v in self.config.get('mei_micro_batch_fallbacks',[])
                                                   if int(v)<self.train_mb]]))
        for attempt,micro in enumerate(batches,1):
            args[args.index('--micro_batch_size')+1]=str(micro)
            offset=log.stat().st_size if log.exists() else 0
            self.log('mei_sft_started',step=step,rows=rows,lr=job['learning_rate'],source=str(self.model_path),
                     tp=self.train_tp,dp=self.train_dp,cp=1,pp=1,max_length=32768,global_batch=gbs,
                     micro_batch=micro,gradient_accumulation=gbs//(self.train_dp*micro),attempt=attempt)
            child=self.spawn(['bash','-c',f'source {REPO}/environment/activate_sft.sh\nexec "$@"','coev-sft',*args],log,env)
            result=child.wait()
            self.stop_children()
            if not result:
                self.train_mb=micro
                break
            with log.open('rb') as f:
                f.seek(offset);failed_output=f.read().decode(errors='replace')
            is_oom=any(marker in failed_output for marker in ('CUDA out of memory','torch.OutOfMemoryError','CUDA error: out of memory'))
            if not is_oom or attempt==len(batches):
                raise RuntimeError(f'MEI SFT failed after policy step {step}, exit {result}; see {log}')
            self.log('mei_sft_oom_retry',step=step,failed_micro_batch=micro,next_micro_batch=batches[attempt],
                     restart_from_source=str(self.model_path),global_batch=gbs)
        checkpoints=[p for p in output.glob('*/checkpoint-*') if (p/'model.safetensors.index.json').is_file()]
        if not checkpoints:raise RuntimeError('MEI SFT did not save final HF weights')
        checkpoint=max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))
        index=read_json(checkpoint/'model.safetensors.index.json')['weight_map']
        if any(not (checkpoint/x).is_file() for x in set(index.values())):raise RuntimeError('Missing new MEI shard')
        self.model_path=checkpoint;self.version+=1;self.model_name=self.name(self.version);self.after=step
        atomic_json(self.root/'job_results'/f'after-step-{step:06d}.json',
                    {'status':'trained','version':self.version,'checkpoint':str(checkpoint),'rows':rows,
                     'learning_rate':job['learning_rate'],'train_tp':self.train_tp,'train_dp':self.train_dp,
                     'micro_batch':self.train_mb,'global_batch':gbs})
        self.serve()



    def run(self):
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'],text=True)
        if any(line.strip().isdigit() for line in out.splitlines()):
            raise RuntimeError('The dedicated MEI GPUs already have a workload; nothing was stopped')
        self.start_gpu_telemetry()
        pending=self.pending_jobs()
        if pending:self.train(read_json(pending[0]))
        else:self.serve()
        while True:
            policy=read_json(self.root/'policy.json',{})
            status=self.config.get('policy_terminal_status')
            if policy.get('phase') in {'failed','complete'}:return
            if status and Path(status).is_file():return
            pending=self.pending_jobs()
            if pending:
                self.train(read_json(pending[0]));continue
            if policy.get('phase')=='request_rollout' and self.after>=int(policy.get('required_mei_after_step',0)):
                self.publish('ready',policy_step=int(policy['step']))
            if any(p.poll() is not None for p in self.children):
                raise RuntimeError('A MEI service process exited')
            time.sleep(1)

    def close(self):
        self.telemetry_stop.set()
        if self.telemetry_thread:self.telemetry_thread.join(timeout=7)
        with contextlib.suppress(Exception):self.stop_children()

def main():
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=read_json(a.config)
    if config['mei_host'] not in subprocess.check_output(['hostname','-I'],text=True).split():
        raise SystemExit('Run the MEI controller only on its dedicated configured node')
    os.environ['NEOS_COEV_ROOT']=config['root']
    Path(config['root']).mkdir(parents=True,exist_ok=True)
    lock=(Path(config['root'])/'mei-controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=MEINode(config)
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:controller.run()
    except BaseException as error:
        controller.publish('failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:controller.close()
if __name__=='__main__':main()
