from __future__ import annotations
import logging
import re
import time
from neos.eval.gateway import FatalLLMGatewayError, RetryExhaustedLLMGatewayError
JUDGE_PROMPT_VERSION = 'vision-deepresearch-official-answer-v1'
JUDGE_SOURCE_COMMIT = '1f13ea7261de0ec8e9961d3ef2c9c33fcd6d27b1'
JUDGE_SOURCE_URL = f'https://github.com/Osilly/Vision-DeepResearch/blob/{JUDGE_SOURCE_COMMIT}/rllm/eval/eval_runner.py'
JUDGE_SYSTEM_PROMPT = 'You are a helpful assistant.'
JUDGE_PROMPT = 'You are a impartial judge evaluating whether a deep research report contains the correct answer.\n\n[Question]\n{question}\n\n[Correct Answer]\n{reference_answer}\n\n[Deep Research Report]\n{assistant_answer}\n\nTask: Determine if the deep research report contains the correct answer anywhere in its content.\n\nInstructions:\n1. Read through the entire research report carefully\n2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n3. Check if the information in the report is consistent with the correct answer\n4. The answer does NOT need to be in a specific format or labeled as "final answer"\n5. Provide your reasoning\n6. Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\nOutput format:\ncorrect: [yes/no]\nreasoning: [your explanation]\n'
CORRECT_RE = re.compile('correct:\\s*\\[?\\s*(yes|no)\\s*\\]?', re.IGNORECASE)
logger = logging.getLogger(__name__)

def render_judge_prompt(question: str, correct_answer: str, response: str) -> str:
    return JUDGE_PROMPT.format(question=question, reference_answer=correct_answer, assistant_answer=response or '')

def judge_messages(question: str, correct_answer: str, response: str) -> list[dict]:
    return [{'role': 'system', 'content': JUDGE_SYSTEM_PROMPT}, {'role': 'user', 'content': render_judge_prompt(question, correct_answer, response)}]

def parse_correct(text: str) -> bool | None:
    match = CORRECT_RE.search(text or '')
    return match.group(1).lower() == 'yes' if match else None

def judge_correct(judge, question: str, correct_answer: str, response: str, retries: int=3) -> dict:
    pass
    messages = judge_messages(question, correct_answer, response)
    attempts = []
    last_response = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            output = judge.chat(messages, max_completion_tokens=2048)
            last_response = output
            record = {'attempt': attempt, 'response': output, 'latency_ms': round((time.perf_counter() - started) * 1000, 3)}
            take = getattr(judge, 'take_last_call', None)
            call = take() if take else None
            if call is not None:
                record['gateway_call'] = call
            attempts.append(record)
            correct = parse_correct(output)
            if correct is not None:
                return {'judgement': output, 'correct': correct, 'judge_attempts': attempts, 'judge_parse_failed': False}
            logger.warning('judge unparsable: %.120s', output)
        except (FatalLLMGatewayError, RetryExhaustedLLMGatewayError):
            raise
        except Exception as error:
            record = {'attempt': attempt, 'error': f'{type(error).__name__}: {error}', 'latency_ms': round((time.perf_counter() - started) * 1000, 3)}
            take = getattr(judge, 'take_last_call', None)
            call = take() if take else None
            if call is not None:
                record['gateway_call'] = call
            attempts.append(record)
            logger.warning('judge call failed: %s', error)
    return {'judgement': last_response, 'correct': False, 'judge_attempts': attempts, 'judge_parse_failed': True}

def judge_final_answer(judge, question: str, correct_answer: str, final_answer: str | None) -> dict:
    answer = (final_answer or '').strip()
    if not answer:
        return {'judgement': 'correct: no\nreasoning: No explicit final answer.', 'correct': False, 'judge_attempts': [], 'judge_parse_failed': False, 'automatic': True}
    return judge_correct(judge, question, correct_answer, answer)
