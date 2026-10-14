from __future__ import annotations
from neos.identifiers import new_id
import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable
from PIL import Image
from neos.workflow.image_codec import codec_config, context_image_jpeg_bytes
from neos.workflow.protocol import NEOS_PROTOCOL_VERSION, NEOS_SYSTEM_PROMPT, build_user_prompt, parse_neos_assistant_message
from neos.workflow.schemas import DEFAULT_NEOS_SCHEMA_VERSION, NEOS_SCHEMA_VERSIONS, all_openai_tool_schemas
from neos.eval.trace_artifacts import EvaluationArtifactStore
from neos.workflow.visual_results import output_refs, reused_refs_from_observation, verify_execution_output_refs
CONVERTER_VERSION = 'neos-trace-to-openai-native-sft-context-jpeg'

def _inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    found = sorted(path.rglob('trajectories.jsonl'))
    if not found:
        found = sorted(path.rglob('*.jsonl'))
    return found

def _artifact_path(artifact: dict[str, Any], store: EvaluationArtifactStore) -> Path:
    local = artifact.get('local_path')
    if local and Path(local).is_file():
        return Path(local).resolve()
    uri = str(artifact.get('uri') or '')
    if uri.startswith('artifact://images/'):
        path = store.resolve_uri(uri)
        if path.is_file():
            return path
    raise FileNotFoundError(f'cannot resolve trajectory image artifact: {artifact}')

class _ContextImageStore:
    pass

    def __init__(self, root: Path):
        self.root = root

    def materialize(self, source: Path) -> Path:
        identity = new_id()
        target = self.root / identity[:2] / f'{identity}.jpg'
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            payload = context_image_jpeg_bytes(opened)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp')
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 384)
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
        return target

def convert_row(row: dict[str, Any], *, artifact_store: EvaluationArtifactStore, correct_only: bool, image_transform: Callable[[Path], Path] | None=None, schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> dict[str, Any]:
    if row.get('error'):
        raise ValueError('trajectory row has an error')
    if correct_only and row.get('correct') is not True:
        raise ValueError('trajectory was not accepted by the judge')
    question = str(row.get('question') or '').strip()
    if not question:
        raise ValueError('question is empty')
    input_artifacts = list(row.get('input_images') or [])

    def resolve(artifact: dict[str, Any]) -> str:
        path = _artifact_path(artifact, artifact_store)
        if image_transform is not None:
            path = image_transform(path)
        return str(path)
    images = [resolve(artifact) for artifact in input_artifacts]
    user_prefix = '<image>\n' * len(input_artifacts)
    messages = [{'role': 'system', 'content': NEOS_SYSTEM_PROMPT}, {'role': 'user', 'content': user_prefix + build_user_prompt(question, len(input_artifacts))}]
    final_seen = False
    pending_call: tuple[str, str] | None = None
    for event in row.get('trajectory') or []:
        role = event.get('role')
        if role == 'assistant':
            if pending_call is not None:
                raise ValueError('assistant turn appears before pending tool observation')
            message = event.get('message')
            if not isinstance(message, dict):
                message = {'role': 'assistant', 'content': event.get('content') or '', 'reasoning_content': event.get('reasoning_content') or '', 'tool_calls': event.get('tool_calls') or []}
            parsed = parse_neos_assistant_message(message, tool_schema_version=schema_version)
            if not parsed.valid:
                raise ValueError(f'invalid assistant action: {parsed.error}')
            if final_seen:
                raise ValueError('assistant content appears after final answer')
            normalized = dict(parsed.message or {})
            messages.append(normalized)
            if parsed.action_kind == 'tool_call':
                call = parsed.tool_call or {}
                pending_call = (str(call.get('id') or ''), str(call.get('name') or ''))
            final_seen = parsed.action_kind == 'answer'
        elif role == 'tool':
            if final_seen:
                raise ValueError('tool observation appears after final answer')
            if pending_call is None:
                raise ValueError('tool observation has no preceding native call')
            call_id, name = pending_call
            if event.get('tool_call_id') != call_id or event.get('name') != name:
                raise ValueError('tool observation ID/name does not match native call')
            artifacts = list(event.get('images') or [])
            tool_images = [resolve(artifact) for artifact in artifacts]
            reused = reused_refs_from_observation(str(event.get('content') or ''))
            if reused:
                expected = output_refs(len(images), len(tool_images))
                metadata = dict(event.get('metadata') or {})
                metadata.setdefault('output_refs', expected)
                metadata.setdefault('reused_refs', reused)
                verify_execution_output_refs(name, metadata, expected, registry_size=len(images))
            images.extend(tool_images)
            messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name, 'content': '<image>\n' * len(tool_images) + str(event.get('content') or '')})
            pending_call = None
    if pending_call is not None:
        raise ValueError('trajectory ends with an unresolved native tool call')
    if not final_seen:
        raise ValueError('trajectory has no valid final answer')
    return {'messages': messages, 'images': images, 'tools': all_openai_tool_schemas(schema_version)}

def iter_rows(paths: Iterable[Path]):
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield (path, line_number, json.loads(line))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--correct-only', action='store_true')
    parser.add_argument('--artifact-root', type=Path, default=None)
    parser.add_argument('--schema-version', choices=NEOS_SCHEMA_VERSIONS, default=DEFAULT_NEOS_SCHEMA_VERSION)
    args = parser.parse_args()
    paths = _inputs(args.input.resolve())
    if not paths:
        raise FileNotFoundError(f'no trajectory JSONL under {args.input}')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    os.chmod(output, 448)
    store = EvaluationArtifactStore(args.artifact_root)
    context_store = _ContextImageStore(output / 'context_images')
    counts: Counter[str] = Counter()
    train_path = output / 'train.jsonl'
    dropped_path = output / 'dropped.jsonl'
    with train_path.open('w', encoding='utf-8') as train, dropped_path.open('w', encoding='utf-8') as dropped:
        for source, line_number, row in iter_rows(paths):
            counts['input'] += 1
            try:
                converted = convert_row(row, artifact_store=store, correct_only=args.correct_only, image_transform=context_store.materialize, schema_version=args.schema_version)
            except Exception as exc:
                counts[f'drop:{type(exc).__name__}'] += 1
                dropped.write(json.dumps({'source': str(source), 'line': line_number, 'sid': row.get('sid'), 'reason': str(exc)}, ensure_ascii=False) + '\n')
                continue
            rendered = json.dumps(converted, ensure_ascii=False, separators=(',', ':')) + '\n'
            train.write(rendered)
            counts['kept'] += 1
    os.chmod(train_path, 384)
    os.chmod(dropped_path, 384)
    manifest = {'schema_version': 'neos-sft-manifest', 'converter_version': CONVERTER_VERSION, 'protocol_version': NEOS_PROTOCOL_VERSION, 'tool_schema_version': args.schema_version, 'source_files': [str(path) for path in paths], 'correct_only': args.correct_only, 'rows': dict(sorted(counts.items())), 'context_image_codec': codec_config(), 'contract_hashes': {}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.chmod(output / 'manifest.json', 384)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
