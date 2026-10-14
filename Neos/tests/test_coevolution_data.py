import json
from pathlib import Path
from unittest.mock import patch

from neos.workflow import coevolution as c


def fields():
    return {'coev_mei_model':'local/mei-h000','coev_mei_version':0,'coev_policy_step':1,
        'rollout_request_id':'r1','qid':'q1',
        'policy_turn_spans':[{'span_id':f'T{i}.A','turn':i,'accepted':True,'action_kind':'tool_call'} for i in range(1,4)],
        'agent_trace':[{'role':'tool','turn':i,'name':'WebVisit','metadata':{'mei_trace_id':f't{i}'}} for i in range(1,4)]}


def test_only_correct_rollouts_or_positive_failed_turns_are_selected():
    assert {r['trace_id'] for r in c.selected_mei_links(fields(),correct=True)}=={'t1','t2','t3'}
    credit={'applied':True,'turns':[{'span_id':'T1.A','status':'productive'},
        {'span_id':'T2.A','status':'causal_error'},{'span_id':'T3.A','status':'recovery'}]}
    assert {r['trace_id'] for r in c.selected_mei_links(fields(),correct=False,credit=credit)}=={'t1','t3'}
    assert not c.selected_mei_links(fields(),correct=False,credit={'applied':False})


def test_exact_io_deduplicates_and_missing_trace_does_not_block(tmp_path):
    with patch.dict('os.environ',{'NEOS_COEV_ROOT':str(tmp_path)}):
        c.observe_answer(fields(),correct=True,judge_called=True,judge_parse_failed=False)
        traces=tmp_path/'mei_traces';traces.mkdir()
        target={'evidence':'source','summary':'answer','relevant_images':[]}
        records=[{'trace_id':f't{i}','namespace':'neos_web_mei','configured_model':'local/mei-h000',
            'actual_model':'local/mei-h000','input':{'messages':[{'role':'user','content':'exact input'}]},
            'output':{'parsed_json':target,'raw_text':json.dumps(target)}} for i in [1,2]]
        (traces/'records.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in records))
        output=tmp_path/'data'
        report=c.build_mei_batch(tmp_path,first_step=1,last_step=1,mei_model='local/mei-h000',output=output)
        assert report['train_rows']==1 and report['duplicate_current_window_count']==1
        assert report['dropped_missing_trace_count']==1
        row=json.loads((output/'train.jsonl').read_text())
        assert json.loads(row['messages'][-1]['content'])==target
        assert 'reasoning_content' not in row['messages'][-1]


def test_prior_successful_updates_are_not_trained_twice(tmp_path):
    target={'summary':'answer'}
    record={'input':{'messages':[{'role':'user','content':'same'}]}}
    identity=c.mei_io_identity(record,target)
    c.atomic_json(tmp_path/'job_results/after-step-000010.json',{'status':'trained'})
    data=tmp_path/'mei_training/after-step-10/data';data.mkdir(parents=True)
    (data/'selection.jsonl').write_text(json.dumps({'accepted':True,'io_identity':identity})+'\n')
    assert identity in c.previously_trained_mei_io(tmp_path,11)
