import json
from pathlib import Path

import pytest
from neos.paths import PROJECT_ROOT
from neos.training.sft import build_command
from neos.training.coevolution import resolve_config


def recipe(name):return json.loads((PROJECT_ROOT/'configs/training'/f'{name}.json').read_text())


@pytest.fixture(autouse=True)
def configured_paths(monkeypatch, tmp_path):
    values = {
        'NEOS_BASE_MODEL_9B':'Qwen3.5-9B', 'NEOS_BASE_MODEL_35B':'Qwen3.5-35B-A3B',
        'NEOS_POLICY_9B_MODEL':'policy-9b', 'NEOS_POLICY_35B_MODEL':'qwen35-35b-a3b-sft',
        'NEOS_MEI_MODEL':'mei', 'NEOS_SFT_DATA':'policy.jsonl',
        'NEOS_MEI_TRAIN_DATA':'mei.jsonl', 'NEOS_MEI_VALIDATION_DATA':'val.jsonl',
        'NEOS_RL_DATA_DIR':'rl-data', 'NEOS_OUTPUT_ROOT':'output',
    }
    for key,value in values.items():monkeypatch.setenv(key,str(tmp_path/value))


@pytest.mark.parametrize('name,length,batch,tp,cp,lr',[
    ('policy_9b_sft',131072,8,4,2,1e-5),
    ('policy_35b_sft',131072,8,2,4,1e-5),
    ('mei_sft',32768,64,2,1,3e-5),
])
def test_full_sft_contract(name,length,batch,tp,cp,lr):
    config=recipe(name);args=build_command(config)
    value=lambda key:args[args.index('--'+key)+1]
    assert value('tuner_type')=='full' and value('num_train_epochs')=='2'
    assert int(value('max_length'))==length and int(value('global_batch_size'))==batch
    assert int(value('tensor_model_parallel_size'))==tp and int(value('context_parallel_size'))==cp
    assert float(value('lr'))==lr
    assert value('no_save_optim')==value('no_save_rng')=='true'
    assert '--train_iters' not in args


@pytest.mark.parametrize('name,batch,steps,tp,cp,ep',[
    ('coev_9b',64,40,4,2,1),('coev_35b',32,82,4,4,16)])
def test_coevolution_contract(name,batch,steps,tp,cp,ep):
    config=resolve_config(recipe(name),policy_hosts=[],mei_host='',run_id='unit',dry_run=True)
    assert (config['policy_batch_size'],config['rollout_n'],config['policy_steps'])==(batch,8,steps)
    assert config['policy_parallelism']=={'tp':tp,'cp':cp,'pp':1,'ep':ep,'etp':1}
    assert config['mei_learning_rate']==1e-6
    assert config['mei_update_k']==config['policy_save_freq']==10
    assert config['answer_judge_model']=='qwen3.7-plus'
    assert config['credit_judge_model']=='qwen3.7-plus'
    assert config['forced_final_correct_reward']==.6 and config['no_validation']
    assert 'resume_checkpoint' not in config
    assert not any('keepalive' in key or 'idle_guard' in key for key in config)


def test_35b_defaults_to_qwen35():
    sft = recipe('policy_35b_sft')
    assert sft['model'] == ''
    assert sft['path_settings']['model'] == 'NEOS_BASE_MODEL_35B'
    args = build_command(sft)
    assert Path(args[args.index('--model')+1]).name == 'Qwen3.5-35B-A3B'
    assert sft['model_type'] == 'qwen3_5_moe'
    assert sft['plugin'] == 'plugins/qwen35_frozen_router_full_1node.py'
    assert (PROJECT_ROOT / sft['plugin']).is_file()
    backend = PROJECT_ROOT / 'plugins/qwen35_neos_full_1node.py'
    assert backend.is_file()
    assert 'qwen36' not in backend.read_text()
    config = resolve_config(recipe('coev_35b'), policy_hosts=[], mei_host='',
                            run_id='unit-qwen35', dry_run=True)
    assert Path(config['initial_policy']).name == 'qwen35-35b-a3b-sft'


def test_coevolution_uses_one_default_name_on_all_hosts(monkeypatch):
    monkeypatch.setenv('NEOS_RUN_ID', '')
    configs = [resolve_config(recipe('coev_35b'), policy_hosts=[], mei_host='', dry_run=True)
               for _ in range(5)]
    assert {config['job_id'] for config in configs} == {'neos-coevolution'}
    assert len({config['root'] for config in configs}) == 1
    monkeypatch.setenv('NEOS_RUN_ID', 'named-experiment')
    config = resolve_config(recipe('coev_35b'), policy_hosts=[], mei_host='', dry_run=True)
    assert config['job_id'] == 'named-experiment'
