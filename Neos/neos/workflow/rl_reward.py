from __future__ import annotations
import asyncio
import os
from typing import Any
from neos.eval.answer_judge import judge_final_answer
from neos.eval.gateway import LLMGateway
from neos.workflow.turn_credit import CREDIT_SCHEMA_VERSION, DEFAULT_CREDIT_MODEL

def _value(extra_info: Any, key: str, default=None):
    return extra_info.get(key, default) if hasattr(extra_info, 'get') else default

def answer_reward(correct: bool, extra_info: Any) -> dict[str, Any]:
    pass
    forced_correct_reward = float(os.getenv('NEOS_FORCED_FINAL_CORRECT_REWARD', '1.0'))
    if not 0.5 < forced_correct_reward <= 1.0:
        raise ValueError('forced-final correct reward must be in (0.5, 1.0]')
    forced = bool(_value(extra_info, 'forced_finalization', False))
    discounted = bool(correct and forced)
    return {'score': (forced_correct_reward if discounted else 1.0) if correct else 0.0, 'acc': float(correct), 'base_answer_reward': float(correct), 'forced_finalization': forced, 'forced_final_correct': discounted, 'forced_final_correct_reward': forced_correct_reward, 'reward_policy_version': 'answer-accuracy-max-turn-discount'}

async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_) -> dict[str, Any]:
    pass
    del data_source
    question = str(_value(extra_info, 'question', '') or '').strip()
    final_answer = str(_value(extra_info, 'final_answer', '') or '').strip()
    correct_answer = str(ground_truth or '').strip()
    judge_model = os.getenv('NEOS_JUDGE_MODEL', 'qwen3.7-plus')
    credit_model = os.getenv('NEOS_TURN_CREDIT_MODEL', DEFAULT_CREDIT_MODEL)
    credit_base = {'schema_version': CREDIT_SCHEMA_VERSION, 'eligible': bool(_value(extra_info, 'turn_credit_eligible', False)), 'applied': False, 'model': credit_model}
    if not question or not correct_answer:
        return {**answer_reward(False, extra_info), 'judge_model': judge_model, 'judge_called': False, 'judge_parse_failed': False, 'turn_credit': {**credit_base, 'reason': 'missing_question_or_ground_truth'}}
    if final_answer:
        judgement = await asyncio.to_thread(judge_final_answer, LLMGateway(model=judge_model), question, correct_answer, final_answer)
        correct = bool(judgement.get('correct'))
        judge_parse_failed = bool(judgement.get('judge_parse_failed'))
        judge_called = True
    else:
        judgement = {'judge_attempts': []}
        correct = False
        judge_parse_failed = False
        judge_called = False
    if correct:
        turn_credit = {**credit_base, 'applied': False, 'reason': 'answer_correct_ordinary_grpo'}
    elif judge_parse_failed:
        turn_credit = {**credit_base, 'applied': False, 'reason': 'answer_judge_parse_failed'}
    else:
        turn_credit = {**credit_base, 'applied': False, 'reason': 'pending_complete_group_outcome'}
    if os.getenv('NEOS_COEV_ROOT'):
        from neos.workflow.coevolution import observe_answer
        observe_answer(extra_info, correct=correct, judge_called=judge_called, judge_parse_failed=judge_parse_failed)
    return {**answer_reward(correct, extra_info), 'judge_model': judge_model, 'judge_called': judge_called, 'judge_parse_failed': judge_parse_failed, 'judge_attempt_count': len(judgement.get('judge_attempts') or []), 'turn_credit': turn_credit}
