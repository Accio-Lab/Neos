import json
import sys

import pandas as pd
from PIL import Image

from neos.eval.durable_jsonl import DurableJSONLPublisher
from neos.eval.trace_artifacts import EvaluationArtifactStore
from neos.training import data, mei_data
from neos.workflow.credit_prefetch import credit_inputs
from neos.workflow.mei_trace import MEITraceStore
from neos.workflow.image_codec import context_image_data_uri
from neos.workflow.turn_credit import _load_reference_bundle


def test_artifact_ids_publish_and_resolve(tmp_path):
    store = EvaluationArtifactStore(tmp_path/'shared', local_root=tmp_path/'local', defer_shared=True)
    image = Image.new('RGB', (32, 32), 'red')
    first = store.save_image(image, image_ref='img_1', kind='fetched')
    second = store.save_image(image, image_ref='img_2', kind='fetched')
    assert first['artifact_id'] != second['artifact_id']
    image_path = store.resolve_uri(first['uri'])
    Image.new('RGB', image.size, 'green').save(image_path)
    result = store.publish_row_artifacts({'images':[first, second]})
    assert result['published_or_present'] == 2
    with Image.open(store.resolve_uri(first['uri'])) as loaded:
        assert loaded.getpixel((0, 0)) == (0, 128, 0)


def test_mei_trace_assets_compile_to_sft(tmp_path):
    store = MEITraceStore(tmp_path/'traces', local_root=tmp_path/'local')
    image = Image.new('RGB', (48, 48), 'red')
    target = {'rational':'visible','candidates':[]}
    captured = store.record(namespace='neos_image_mei', prompt_version='test', model='local/neos-mei',
        fallback_models=[], messages=[{'role':'user','content':[
            {'type':'text','text':'Inspect this image'},
            {'type':'image_url','image_url':{'url':context_image_data_uri(image)}}]}],
        max_completion_tokens=100, raw_output=json.dumps(target), parsed_output=target,
        timing=None, descriptor={})
    store.close()
    record = json.loads((tmp_path/'traces/records.jsonl').read_text())
    assert record['trace_id'] == captured['trace_id']
    assert record['assets'][0]['asset_id']
    messages, images = mei_data.convert_messages(record, target)
    assert len(images) == 1 and '<image>' in messages[0]['content']
    assert json.loads(messages[-1]['content']) == target


def test_reference_loading_uses_the_supplied_path(tmp_path):
    path=tmp_path/'reference.json'
    path.write_text(json.dumps({'qid':'q1','status':'complete','final_answer':'A',
        'messages':[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}],
        'image_registry':[]}))
    row=data.build_row({'qid':'q1','question':'Q','answer':'A','images':[],
                       'reference_trajectory_path':str(path)},0,tmp_path)
    assert row['extra_info']['turn_credit_eligible']
    assert _load_reference_bundle(str(path))['final_answer']=='A'
    fields={**row['extra_info'],'turn_credit_capture':True,'turn_credit_trace':[{'role':'assistant'}],
            'policy_turn_spans':[{'span_id':'T1.A'}]}
    assert credit_inputs(fields)['reference_path']==str(path)


def test_prepare_rl_data_accepts_custom_row_count(tmp_path, monkeypatch):
    source=tmp_path/'questions.jsonl';output=tmp_path/'prepared'
    source.write_text(''.join(json.dumps({'qid':f'q{i}','question':'Q','answer':'A'})+'\n' for i in range(3)))
    monkeypatch.setattr(sys,'argv',['prepare','--input',str(source),'--output',str(output)])
    data.main()
    assert len(pd.read_parquet(output/'train.parquet'))==3
    assert json.loads((output/'manifest.json').read_text())['rows']==3


def test_publisher_resume_uses_saved_row_ids(tmp_path):
    target=tmp_path/'results.jsonl';spool=tmp_path/'spool'
    row={'bench':'test','sid':'sample','evaluation_repeat':1}
    first=DurableJSONLPublisher(target,local_root=spool,batch_rows=1)
    assert first._path_for_key(('test','sample',1)) == first._path_for_key(('test','sample',1))
    first.enqueue(row);first.close(5)
    second=DurableJSONLPublisher(target,local_root=spool,existing_rows=[row],batch_rows=1)
    assert second.enqueue(row)['already_published']
    second.close(5)
    assert len(target.read_text().splitlines())==1
