import json
import os
from pathlib import Path

import pytest

from neos.configuration import PATH_KEYS, ROOT, configured_environment, evaluation_output_path, resolve_recipe_paths
from neos.eval.datasets import load_fvqa
from neos.paths import project_path


def use_config(monkeypatch,tmp_path,data):
    for key in json.loads((ROOT/'config.example.json').read_text()):monkeypatch.delenv(key,raising=False)
    path=tmp_path/'settings.json';path.write_text(json.dumps(data))
    monkeypatch.setenv('NEOS_CONFIG',str(path))


def test_shipped_paths_and_provider_credentials_are_empty():
    data=json.loads((ROOT/'config.example.json').read_text())
    for key in PATH_KEYS:assert data[key]==''
    for key,value in data.items():
        if key.endswith(('_ENDPOINT','_API_KEY','_API_VERSION','_PUBLIC_BASE_URL','_UPLOAD_URL','_ACCESS_KEY_ID','_ACCESS_KEY_SECRET','_SESSION_TOKEN')):
            assert value==''
    for path in (ROOT/'configs/training').glob('*.json'):
        recipe=json.loads(path.read_text())
        for field in recipe.get('path_settings',{}):assert recipe[field]==''


def test_single_file_resolves_paths_and_headers(monkeypatch,tmp_path):
    use_config(monkeypatch,tmp_path,{'NEOS_VENV':'.venv','NEOS_POLICY_35B_MODEL':'models/own-model',
        'NEOS_SERVE_POLICY_MODEL':'${NEOS_POLICY_35B_MODEL}',
        'NEOS_DASHSCOPE_HEADERS':{'organization':'example'},'NEOS_POLICY_HOSTS':['p0','p1']})
    env=configured_environment()
    assert env['NEOS_VENV']==str(ROOT/'.venv')
    assert env['NEOS_SERVE_POLICY_MODEL']==env['NEOS_POLICY_35B_MODEL']
    assert json.loads(env['NEOS_DASHSCOPE_HEADERS'])=={'organization':'example'}
    assert env['NEOS_POLICY_HOSTS']=='p0,p1'


def test_generated_mei_endpoint_is_not_overwritten(monkeypatch,tmp_path):
    use_config(monkeypatch,tmp_path,{'NEOS_LOCAL_MEI_ENDPOINT':'https://configured.example.test/v1'})
    monkeypatch.setenv('NEOS_LOCAL_MEI_ENDPOINT','http://runtime.example.test:48080/v1')
    assert configured_environment()['NEOS_LOCAL_MEI_ENDPOINT']=='http://runtime.example.test:48080/v1'


def test_mei_weights_and_served_model_are_independent(monkeypatch,tmp_path):
    use_config(monkeypatch,tmp_path,{'NEOS_MEI_MODEL':'models/mei',
        'NEOS_MEI_MODEL_SPEC':'local/neos-mei'})
    env=configured_environment()
    assert env['NEOS_MEI_MODEL']==str(ROOT/'models/mei')
    assert env['NEOS_MEI_MODEL_SPEC']=='local/neos-mei'


def test_empty_path_fails_before_launch(monkeypatch):
    monkeypatch.delenv('NEOS_BASE_MODEL_35B',raising=False)
    with pytest.raises(ValueError,match='NEOS_BASE_MODEL_35B'):
        resolve_recipe_paths({'model':'','path_settings':{'model':'NEOS_BASE_MODEL_35B'}})
    with pytest.raises(ValueError):project_path('')


def test_benchmark_root_comes_from_configuration(monkeypatch,tmp_path):
    folder=tmp_path/'visiondr300/fvqa';folder.mkdir(parents=True)
    (folder/'manifest.jsonl').write_text(json.dumps({'sid':'q1','question':'Question','answers':['Answer'],'image_paths':[]})+'\n')
    monkeypatch.setenv('NEOS_EVAL_DATA_ROOT',str(tmp_path))
    rows=load_fvqa()
    assert len(rows)==1 and rows[0]['answers']==['Answer']


def test_unknown_and_cyclic_configuration_are_rejected(monkeypatch,tmp_path):
    use_config(monkeypatch,tmp_path,{'HOME':'not-allowed'})
    with pytest.raises(ValueError,match='Unknown'):configured_environment()
    use_config(monkeypatch,tmp_path,{'NEOS_VENV':'${NEOS_VENV}'})
    with pytest.raises(ValueError,match='Cyclic'):configured_environment()


def test_evaluation_names_are_optional_and_unique(monkeypatch, tmp_path):
    monkeypatch.setenv('NEOS_OUTPUT_ROOT', str(tmp_path))
    first = evaluation_output_path(benchmark='vdr', repeat_count=4)
    second = evaluation_output_path(benchmark='vdr', repeat_count=4, run_id='')
    assert first != second
    for path in (first, second):
        assert path.parent == tmp_path / 'evaluations'
        assert path.name.startswith('vdr-repeat4-')
        assert path.is_dir()


def test_evaluation_explicit_name_and_output_remain_resumable(monkeypatch, tmp_path):
    monkeypatch.setenv('NEOS_OUTPUT_ROOT', str(tmp_path))
    named = evaluation_output_path(benchmark='vdr', repeat_count=4, run_id='previous')
    assert named == tmp_path / 'evaluations' / 'previous'
    assert named == evaluation_output_path(benchmark='vdr', repeat_count=4, run_id='previous')
    monkeypatch.delenv('NEOS_OUTPUT_ROOT')
    assert named == evaluation_output_path(benchmark='vdr', repeat_count=4, out=str(named))


def test_evaluation_explicit_name_stays_inside_output_root(monkeypatch, tmp_path):
    monkeypatch.setenv('NEOS_OUTPUT_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='Run ID'):
        evaluation_output_path(benchmark='vdr', repeat_count=4, run_id='../outside')
