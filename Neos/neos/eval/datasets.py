
from __future__ import annotations
import json
import os
from pathlib import Path
from neos.paths import project_path
from neos.configuration import required_setting

DATASET_FOLDERS = {
    'mmsearch': 'mmsearch', 'browsecomp_vl': 'browsecomp_vl',
    'mm_browsecomp': 'mm_browsecomp', 'mmsearch_plus': 'mmsearch_plus',
    'simplevqa': 'visiondr300/simplevqa', 'fvqa': 'visiondr300/fvqa',
    'livevqa': 'visiondr300/livevqa_mc', 'infoseek': 'infoseek',
    'vdr': 'vdr', 'browsecomp_v3': 'browsecomp_v3', 'visbrowse': 'visbrowse',
}

def configured_root(root, bench):
    return project_path(root) if root is not None else project_path(required_setting('NEOS_EVAL_DATA_ROOT')) / DATASET_FOLDERS[bench]

def load_mmsearch(root: Path | None = None) -> list[dict]:
    root = configured_root(root, 'mmsearch')
    if (root / 'manifest.jsonl').is_file():
        return load_normalized_manifest(root, 'mmsearch')
    samples = []
    for line in (root / "end2end.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        alts = d.get("alternative_gt_answers") or []
        if isinstance(alts, str):
            try:
                alts = json.loads(alts)
            except json.JSONDecodeError:
                alts = []
        answers = [d["gt_answer"]] + [a for a in alts if a and a != d["gt_answer"]]
        img = root / (d.get("query_image") or "")
        samples.append({
            "sid": d["sample_id"], "bench": "mmsearch", "question": d["query"],
            "answers": answers, "area": d.get("area"), "subfield": d.get("subfield"),
            "image_path": img if d.get("query_image") and img.exists() else None,
        })
    return samples


def load_bcvl(root: Path | None = None) -> list[dict]:
    root = configured_root(root, 'browsecomp_vl')
    if (root / 'manifest.jsonl').is_file():
        return load_normalized_manifest(root, 'browsecomp_vl')
    samples = []
    for lv in ("1", "2"):
        for i, line in enumerate((root / f"bc_vl_level{lv}.jsonl").read_text().splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            img = root / d["image_path"].replace("browsecomp-vl/", "")
            samples.append({
                "sid": f"level{lv}_{i+1}",
                "bench": "browsecomp_vl",
                "question": d["question"],
                "answers": d["answers"], "level": lv, "domain": d.get("domain"),
                "image_path": img if img.exists() else None,
            })
    return samples


def load_mmbrowsecomp(root: Path | None = None) -> list[dict]:
    root = configured_root(root, 'mm_browsecomp')
    if (root / 'manifest.jsonl').is_file():
        return load_normalized_manifest(root, 'mm_browsecomp')
    pass
    data_path = root / "data" / "MMBrowseComp_400_decrypted.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"{data_path} is missing. Clone the official MM-BrowseComp repo and "
            "run its src/decrypt.py on data/MMBrowseComp_400.jsonl first."
        )
    samples = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        image_paths = []
        image_source_urls = []
        for image_url in data.get("images") or []:
            
            image_name = Path(str(image_url).split("?", 1)[0]).name
            image_path = root / "MMBC_images" / image_name
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"MM-BrowseComp id={data.get('id')} image missing: {image_path}"
                )
            image_paths.append(image_path)
            
            
            
            image_source_urls.append(str(image_url))
        samples.append({
            "sid": f"id_{data['id']}",
            "bench": "mm_browsecomp",
            "question": data["question"],
            "answers": [data["answer"]],
            "image_paths": image_paths,
            "image_source_urls": image_source_urls,
            "area": data.get("category"),
            "category": data.get("category"),
            "subtask": data.get("subtask"),
            "level": str(data["level"]) if data.get("level") is not None else None,
            "source": data.get("source"),
            "checklist": data.get("checklist"),
        })
    return samples


def load_normalized_manifest(root: Path, bench: str) -> list[dict]:
    pass
    root = configured_root(root, bench)
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} is missing; provide a normalized manifest with sid, question, answers and local image_paths")
    samples = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        image_paths = []
        for relative in data.get("image_paths") or []:
            image_path = root / relative
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{bench} sid={data.get('sid')} image missing: {image_path}"
                )
            image_paths.append(image_path)
        answers = data.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        reference_missing = bool(data.get("reference_missing"))
        if not answers and not reference_missing:
            raise ValueError(
                f"{bench} sid={data.get('sid')} has no answer and is not "
                "marked reference_missing"
            )
        samples.append({
            **data,
            "sid": str(data["sid"]),
            "bench": bench,
            "answers": [str(answer) for answer in answers if str(answer).strip()],
            "image_paths": image_paths,
            "area": data.get("category") or data.get("area"),
            "reference_missing": reference_missing,
        })
    return samples


def load_mmsearch_plus(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "mmsearch_plus")


def load_simplevqa(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "simplevqa")


def load_livevqa(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "livevqa")


def load_fvqa(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "fvqa")


def load_infoseek(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "infoseek")


def load_vdr(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "vdr")


def load_browsecomp_v3(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "browsecomp_v3")


def load_visbrowse(root: Path | None = None) -> list[dict]:
    return load_normalized_manifest(root, "visbrowse")
