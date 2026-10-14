import asyncio
from unittest.mock import patch
from neos.eval import answer_judge
from neos.eval.gateway import LLMGateway, _is_dashscope_model
from neos.workflow import rl_reward


class Judge:
    model='unit'
    def __init__(self):self.messages=[]
    def chat(self,messages,**kwargs):self.messages.append(messages);return 'correct: [yes]\nreasoning: match'
    def take_last_call(self):return None


def test_official_prompt_and_complete_answer():
    judge=Judge();answer='x'*10000+' correct answer'
    result=answer_judge.judge_final_answer(judge,'question','correct answer',answer)
    assert result['correct'] and judge.messages[0]==answer_judge.judge_messages('question','correct answer',answer)


def test_rl_uses_the_same_official_judge():
    judge=Judge()
    with patch.object(rl_reward,'LLMGateway',return_value=judge),patch.dict('os.environ',{'NEOS_COEV_ROOT':''}):
        result=asyncio.run(rl_reward.compute_score('unit','','A',extra_info={'question':'Q','final_answer':'A'}))
    assert result['score']==1.0
    assert judge.messages[0]==answer_judge.judge_messages('Q','A','A')


def test_forced_correct_discount_preserves_correctness():
    with patch.dict('os.environ',{'NEOS_FORCED_FINAL_CORRECT_REWARD':'0.6'}):
        assert rl_reward.answer_reward(True,{'forced_finalization':True})['score']==.6
        assert rl_reward.answer_reward(True,{'forced_finalization':True})['acc']==1.
        assert rl_reward.answer_reward(False,{})['score']==0.


def test_judge_defaults_and_model_overrides(monkeypatch):
    monkeypatch.delenv('NEOS_JUDGE_MODEL',raising=False)
    monkeypatch.delenv('NEOS_TURN_CREDIT_MODEL',raising=False)
    result=asyncio.run(rl_reward.compute_score('unit','',''))
    assert result['judge_model']==result['turn_credit']['model']=='qwen3.7-plus'
    assert _is_dashscope_model(result['judge_model'])
    model='qwen-custom'
    monkeypatch.setenv('NEOS_JUDGE_MODEL',model)
    monkeypatch.setenv('NEOS_TURN_CREDIT_MODEL',model)
    result=asyncio.run(rl_reward.compute_score('unit','',''))
    assert result['judge_model']==result['turn_credit']['model']==model
    assert LLMGateway(model=model).model==model and _is_dashscope_model(model)
