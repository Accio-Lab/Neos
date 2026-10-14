import copy
import torch
from neos.workflow import turn_credit as tc


def spans():
    return [dict(span_id=f'T{i}.A1',start=6*i,end=6*i+6,
        component_spans={'cot':{'start':6*i,'end':6*i+4},'act':{'start':6*i+4,'end':6*i+6}})
        for i in range(5)]


def test_explicit_productive_is_not_overwritten_by_downstream(monkeypatch):
    monkeypatch.setenv('NEOS_TURN_CREDIT_CAUSAL_COMPONENTS','0')
    payload={'schema_version':tc.credit_schema_version(),
        'events':[{'span_id':'T0.A1','type':'causal_error','severity':1,'confidence':1}],
        'productive_turns':[{'span_id':'T1.A1','confidence':1,'rationale':'valid new evidence'}],
        'confidence':1}
    original=copy.deepcopy(payload)
    judged=tc._validate_credit_payload(payload,spans())
    assert payload==original
    assert [r['status'] for r in judged['turns']]==['causal_error','productive',
        'downstream_consequence','downstream_consequence','downstream_consequence']
    fields={'policy_turn_spans':spans(),'reward_extra_info':{'acc':0,
        'turn_credit':dict(judged,eligible=True,applied=True)}}
    before=-torch.ones((1,30));mask=torch.ones((1,30))
    after,_=tc.reshape_failed_turn_advantages(before,mask,[fields])
    assert (after[0,6:12]>0).all() and after.min()>=-4
    assert torch.allclose(after[0,12:18],torch.full((6,),-.5*.65))


def test_correct_or_failed_judge_does_not_change_grpo():
    before=torch.ones((1,30));mask=torch.ones((1,30))
    for reward in [{'acc':1,'turn_credit':{'eligible':True,'applied':True}},
                   {'acc':0,'turn_credit':{'eligible':True,'applied':False}}]:
        after,_=tc.reshape_failed_turn_advantages(before,mask,[{'policy_turn_spans':spans(),'reward_extra_info':reward}])
        assert torch.equal(before,after)
