
from __future__ import annotations
from neos.paths import project_path
from neos.configuration import required_setting
import json
import os
from pathlib import Path

EXPECTED = {'browsecomp':1266,'browsecomp_zh':289,'hle_text':2158,'deepsearchqa':900}


def load_text_search(bench: str, root: Path | None = None, *, check_count: bool = True) -> list[dict]:
    if bench not in EXPECTED:
        raise ValueError(f'Unsupported text benchmark: {bench}')
    root = project_path(root or required_setting('NEOS_TEXT_SEARCH_DATA_ROOT'))
    samples=[]
    with (root/f'{bench}.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get('benchmark') != bench or row.get('images') or row.get('file_name'):
                raise ValueError('Text-input benchmark modality/identity mismatch')
            if not row.get('question') or not str(row.get('answer') or '').strip():
                raise ValueError('Missing question/ground truth')
            metadata=row.get('metadata') or {}
            samples.append({'sid':str(row['id']),'bench':bench,'question':row['question'],
                'answers':[row['answer']], 'image_paths':[], 'reference_missing':False,
                'area':metadata.get('category') or metadata.get('problem_category') or metadata.get('topic'),
                'category':metadata.get('category') or metadata.get('problem_category'),
                
                
                'answer_type':metadata.get('answer_type'), 'input_variant':'original_question'})
    if len({r['sid'] for r in samples}) != len(samples):
        raise ValueError(f'{bench}: duplicate IDs')
    if check_count and len(samples)!=EXPECTED[bench]:
        raise ValueError(f'{bench}: expected {EXPECTED[bench]} rows, got {len(samples)}')
    return samples


def load_browsecomp(): return load_text_search('browsecomp')
def load_browsecomp_zh(): return load_text_search('browsecomp_zh')
def load_hle_text(): return load_text_search('hle_text')
def load_deepsearchqa(): return load_text_search('deepsearchqa')

TEXT_SEARCH_LOADERS = {'browsecomp':load_browsecomp,'browsecomp_zh':load_browsecomp_zh,
                       'hle_text':load_hle_text,'deepsearchqa':load_deepsearchqa}
