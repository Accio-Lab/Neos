import argparse
import json
import os
from pathlib import Path
import re
from neos.workflow.protocol import NEOS_SYSTEM_PROMPT, NEOS_PROTOCOL_VERSION, build_user_prompt
from neos.workflow.image_codec import CONTEXT_IMAGE_MAX_PIXELS
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_TOOL_NAMES
from neos.configuration import required_setting
from neos.paths import project_path

def build_row(record, index, base):
    question = str(record.get('question') or '').strip()
    answer = str(record.get('answer') or record.get('ground_truth') or '').strip()
    qid = str(record.get('qid') or record.get('source_sample_id') or '').strip()
    if not question or not answer or (not qid):
        raise ValueError(f'Row{index}: question, answer and stable qid are required')
    images = []
    for value in record.get('images') or []:
        path = Path(value.get('image') if isinstance(value, dict) else value).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append({'image': str(path.resolve()), 'max_pixels': CONTEXT_IMAGE_MAX_PIXELS, 'min_pixels': 3136})
    reference = record.get('reference_trajectory_path') or ''
    if reference:
        packed = re.fullmatch('(.+\\.jsonl)#byte=(\\d+),(\\d+)', reference)
        path = Path(packed[1] if packed else reference).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if packed:
            reference = f'{path.resolve()}#byte={packed[2]},{packed[3]}'
        else:
            reference = str(path.resolve())
    eligible = bool(reference)
    extra = {'qid': qid, 'source_row': index, 'source_sample_id': record.get('source_sample_id'), 'question': question, 'answer': answer, 'answer_aliases': record.get('answer_aliases', []), 'initial_image_count': len(images), 'need_tools_kwargs': False, 'protocol_version': NEOS_PROTOCOL_VERSION, 'tool_schema_version': DEFAULT_NEOS_SCHEMA_VERSION, 'tool_selection': sorted(NEOS_TOOL_NAMES), 'turn_credit_eligible': eligible, 'reference_trajectory_path': reference, 'reference_policy_model': record.get('reference_policy_model', '')}
    return {'data_source': record.get('data_source', 'neos_qa'), 'agent_name': 'neos_agent', 'prompt': [{'role': 'system', 'content': NEOS_SYSTEM_PROMPT}, {'role': 'user', 'content': '<image>\n' * len(images) + build_user_prompt(question, len(images))}], 'images': images, 'ability': 'multimodal_long_horizon_search', 'reward_model': {'style': 'llm_judge_binary_plus_failed_turn_credit', 'ground_truth': answer}, 'extra_info': extra}

def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    rows = []
    seen = set()
    a.input = project_path(a.input or required_setting('NEOS_RL_INPUT_JSONL'))
    a.output = project_path(a.output or required_setting('NEOS_RL_DATA_DIR'))
    with a.input.open() as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = build_row(json.loads(line), index, a.input.resolve().parent)
            qid = row['extra_info']['qid']
            if qid in seen:
                raise ValueError(f'Duplicate QID: {qid}')
            seen.add(qid)
            rows.append(row)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'train.parquet').exists():
        raise FileExistsError(a.output / 'train.parquet')
    frame = pd.DataFrame(rows)
    frame.to_parquet(a.output / 'train.parquet', index=False)
    frame.iloc[:0].to_parquet(a.output / 'val.parquet', index=False)
    (a.output / 'manifest.json').write_text(json.dumps({'rows': len(rows), 'validation_rows': 0, 'reference_eligible': sum((r['extra_info']['turn_credit_eligible'] for r in rows)), 'protocol_version': NEOS_PROTOCOL_VERSION}, indent=2) + '\n')
if __name__ == '__main__':
    main()
