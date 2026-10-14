
from __future__ import annotations

import json
import re
from typing import Any, Iterable


VISUAL_RESULT_SCHEMA_VERSION = "neos-visual-result-output-ref"
VISUAL_TOOLS = frozenset({"FetchImage", "CropImage"})
_SUCCESS_STATUS = {"FetchImage": "loaded", "CropImage": "cropped"}
_IMAGE_REF_RE = re.compile(r"^img_[1-9][0-9]*$")
_REGISTER_UPDATE_ANNOTATION_RE = re.compile(
    r"\n\nVisual register update: successful outputs were appended "
    r"in request order as img_[1-9][0-9]*(?:, img_[1-9][0-9]*)*\. "
    r"These pixels are now available to the policy\.\s*\Z"
)


def output_refs(registry_size: int, count: int) -> list[str]:
    pass
    if registry_size < 0 or count < 0:
        raise ValueError("registry_size and count must be non-negative")
    return [f"img_{registry_size + index}" for index in range(1, count + 1)]


def _validated_refs(refs: Iterable[str]) -> list[str]:
    normalized = [str(ref) for ref in refs]
    if any(_IMAGE_REF_RE.fullmatch(ref) is None for ref in normalized):
        raise ValueError(f"invalid visual output refs: {normalized!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate visual output refs: {normalized!r}")
    return normalized


def attach_output_refs(
    tool_name: str,
    records: list[dict[str, Any]],
    refs: Iterable[str],
) -> list[dict[str, Any]]:
    pass
    if tool_name not in VISUAL_TOOLS:
        if list(refs):
            raise ValueError(f"{tool_name} cannot materialize visual outputs")
        return records
    normalized_refs = _validated_refs(refs)
    success_status = _SUCCESS_STATUS[tool_name]
    success_records = [
        record for record in records if record.get("status") == success_status
    ]
    if len(success_records) != len(normalized_refs):
        raise ValueError(
            f"{tool_name} returned {len(success_records)} successful records "
            f"but {len(normalized_refs)} image refs"
        )
    ref_iter = iter(normalized_refs)
    upgraded: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        if record.get("status") == success_status:
            expected = next(ref_iter)
            existing = record.get("output_ref")
            if existing is not None and existing != expected:
                raise ValueError(
                    f"{tool_name} output_ref mismatch: {existing!r} != {expected!r}"
                )
            record["output_ref"] = expected
        elif record.get('status') == 'reused':
            if _IMAGE_REF_RE.fullmatch(str(record.get('output_ref') or '')) is None:
                raise ValueError(f'{tool_name} reused record requires a valid existing output_ref')
        elif "output_ref" in record:
            raise ValueError(
                f"{tool_name} non-success record unexpectedly has output_ref"
            )
        upgraded.append(record)
    return upgraded


def render_visual_result(
    tool_name: str,
    records: list[dict[str, Any]],
    refs: Iterable[str],
) -> str:
    pass
    upgraded = attach_output_refs(tool_name, records, refs)
    return json.dumps(upgraded, ensure_ascii=False, indent=2)


def strip_register_update_annotation(observation: str) -> str:
    pass
    return _REGISTER_UPDATE_ANNOTATION_RE.sub("", str(observation)).rstrip()


def migrate_visual_observation(
    tool_name: str,
    observation: str,
    refs: Iterable[str],
) -> str:
    pass
    if tool_name not in VISUAL_TOOLS:
        return observation
    annotation_free = strip_register_update_annotation(observation)
    try:
        records = json.loads(annotation_free)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"cannot parse saved {tool_name} observation as JSON: {exc}"
        ) from exc
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"saved {tool_name} observation is not a record list")
    return render_visual_result(tool_name, records, refs)


def verify_execution_output_refs(
    tool_name: str,
    metadata: dict[str, Any],
    expected_refs: Iterable[str],
    *,
    registry_size: int | None = None,
) -> None:
    pass
    expected = _validated_refs(expected_refs)
    actual = metadata.get("output_refs")
    if tool_name in VISUAL_TOOLS:
        if actual is None and not expected and not metadata.get('reused_refs'):
            return
        if not isinstance(actual, list) or [str(ref) for ref in actual] != expected:
            raise RuntimeError(
                f"{tool_name} runtime output_refs {actual!r} do not match "
                f"agent refs {expected!r}"
            )
    elif expected:
        raise RuntimeError(f"non-visual tool {tool_name} returned image pixels")
    reused = _validated_refs(metadata.get('reused_refs') or [])
    if reused:
        if tool_name not in VISUAL_TOOLS or registry_size is None:
            raise RuntimeError('Reused visual refs require a visual tool and the prior registry size')
        known = set(output_refs(0, registry_size)) | set(expected)
        if not set(reused).issubset(known):
            raise RuntimeError(f'Reused refs are not already visible or created by this call: {reused}')


def reused_refs_from_observation(observation: str) -> list[str]:
    pass
    try:
        records = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        return []
    refs = [str(r.get('output_ref') or '') for r in records
            if isinstance(r, dict) and r.get('status') == 'reused']
    return _validated_refs(dict.fromkeys(refs))
