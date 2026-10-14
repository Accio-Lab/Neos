
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from neos.paths import PROJECT_ROOT, project_path
from neos.configuration import required_setting


def hf_source(checkpoint):
    for candidate in [checkpoint,checkpoint/'actor/model/huggingface',checkpoint/'model/huggingface']:
        if (candidate/'model.safetensors.index.json').is_file() or (candidate/'model.safetensors').is_file():return candidate
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',nargs='?');p.add_argument('output',nargs='?');p.add_argument('--base-model')
    p.add_argument('--target',choices=['policy-9b','policy-35b','mei'])
    p.add_argument('--tp',type=int);p.add_argument('--cp',type=int)
    p.add_argument('--pp',type=int);p.add_argument('--ep',type=int)
    p.add_argument('--etp',type=int);p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    defaults={'tp':4,'cp':2,'pp':1,'ep':1,'etp':1}
    if a.target:
        roles={
            'policy-9b':('NEOS_SFT_CKPT_9B','NEOS_POLICY_9B_MODEL','NEOS_BASE_MODEL_9B','policy_9b_sft'),
            'policy-35b':('NEOS_SFT_CKPT_35B','NEOS_POLICY_35B_MODEL','NEOS_BASE_MODEL_35B','policy_35b_sft'),
            'mei':('NEOS_SFT_CKPT_MEI','NEOS_MEI_MODEL','NEOS_BASE_MODEL_9B','mei_sft'),
        }
        source,destination,base,recipe=roles[a.target]
        a.checkpoint=a.checkpoint or required_setting(source)
        a.output=a.output or required_setting(destination)
        a.base_model=a.base_model or required_setting(base)
        defaults.update(json.loads((PROJECT_ROOT/'configs/training'/f'{recipe}.json').read_text()))
    if not a.checkpoint or not a.output:
        p.error('Choose --target with paths in config.local.json, or supply checkpoint and output')
    for key in ('tp','cp','pp','ep','etp'):
        if getattr(a,key) is None:setattr(a,key,defaults[key])
    checkpoint,output=project_path(a.checkpoint),project_path(a.output)
    source=hf_source(checkpoint)
    command=[sys.executable,'-m','torch.distributed.run',f'--nproc-per-node={a.gpus}',
        '--nnodes=1','--master-addr=127.0.0.1',f'--master-port={os.getenv("MASTER_PORT","29677")}',
        '--module','swift.cli._megatron.export','--mcore_model',str(checkpoint),
        '--output_dir',str(output),'--to_hf','true','--torch_dtype','bfloat16',
        '--tensor_model_parallel_size',str(a.tp),'--context_parallel_size',str(a.cp),
        '--pipeline_model_parallel_size',str(a.pp),'--expert_model_parallel_size',str(a.ep),
        '--expert_tensor_parallel_size',str(a.etp),'--sequence_parallel',str(a.tp>1).lower(),
        '--test_convert_precision','false','--external_plugins',str(PROJECT_ROOT/'plugins/qwen35_neos_native_tools.py')]
    if a.base_model:command+=['--model',str(project_path(a.base_model))]
    if a.dry_run:print(json.dumps({'hf_source':str(source) if source else None,'export_command':command},indent=2));return
    if output.exists():raise FileExistsError(output)
    if not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
    if source:
        output.mkdir(parents=True)
        for file in source.iterdir():
            if not file.is_file():continue
            if file.suffix=='.safetensors':
                try:os.link(file,output/file.name)
                except OSError:shutil.copy2(file,output/file.name)
            elif file.suffix in {'.json','.jinja','.txt'}:shutil.copy2(file,output/file.name)
    else:
        if min(a.tp,a.cp,a.pp,a.ep,a.etp,a.gpus)<1 or a.gpus%(a.tp*a.cp*a.pp) or a.gpus%(a.ep*a.etp*a.pp):
            raise ValueError('Invalid original checkpoint topology')
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True,timeout=10)
        if any(line.strip().isdigit() for line in active.splitlines()):
            raise RuntimeError('The conversion GPUs already have a workload; nothing was stopped')
        env=dict(os.environ,NPROC_PER_NODE=str(a.gpus))
        subprocess.run(command,env=env,check=True)
    if a.base_model:
        for name in ['chat_template.jinja','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json',
                     'preprocessor_config.json','processor_config.json','video_preprocessor_config.json']:
            base=project_path(a.base_model)/name
            if not (output/name).exists() and base.is_file():shutil.copy2(base,output/name)
    if not (output/'preprocessor_config.json').exists() and (output/'processor_config.json').exists():
        config=json.loads((output/'processor_config.json').read_text())
        if isinstance(config.get('image_processor'),dict):
            (output/'preprocessor_config.json').write_text(json.dumps(config['image_processor'],indent=2)+'\n')
    index=output/'model.safetensors.index.json'
    if index.is_file():
        if any(not (output/name).is_file() for name in set(json.loads(index.read_text())['weight_map'].values())):
            raise RuntimeError('Incomplete exported HF shards')
    elif not (output/'model.safetensors').is_file():raise RuntimeError('Export did not produce HF weights')
    print(json.dumps({'checkpoint':str(checkpoint),'model':str(output),'format':'huggingface'}))


if __name__=='__main__':main()
