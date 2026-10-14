#!/usr/bin/env python3
"""
Pre-validate all cached HuggingFace models to provide detailed feedback.

This script runs once during CI initialization (in prepare_runner.sh) to:
1. Scan snapshots in ~/.cache/huggingface/hub/ (with time/quantity limits)
2. Validate completeness (config/tokenizer/weights)
3. Output detailed failure reasons for debugging

NOTE: This script no longer writes shared validation markers. Each test run
independently validates its cache using per-run markers to avoid cross-runner
cache state pollution.
"""

import glob
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.model_loader.ci_weight_validation import (  
    _validate_diffusion_model,
    validate_cache_with_detailed_reason,
)


MAX_VALIDATION_TIME_SECONDS = 300  


def find_all_hf_snapshots():
    """
    Find all HuggingFace snapshots in cache.

    Returns:
        List of (model_name, snapshot_dir) tuples, sorted by mtime (newest first)
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")

    if not os.path.isdir(hub_dir):
        print(f"HF hub directory not found: {hub_dir}")
        return []

    snapshots = []

    
    for model_dir in glob.glob(os.path.join(hub_dir, "models--*")):
        
        dir_name = os.path.basename(model_dir)
        if not dir_name.startswith("models--"):
            continue

        
        
        parts = dir_name.split("--")
        if len(parts) < 3 or parts[0] != "models":
            
            continue
        
        
        model_name = parts[1] + "/" + "-".join(parts[2:])

        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        
        for snapshot_hash_dir in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot_hash_dir)
            if os.path.isdir(snapshot_path):
                try:
                    mtime = os.path.getmtime(snapshot_path)
                    snapshots.append((model_name, snapshot_path, mtime))
                except OSError:
                    continue

    
    snapshots.sort(key=lambda x: x[2], reverse=True)

    
    return [(name, path) for name, path, _ in snapshots]


def is_transformers_text_model(snapshot_dir):
    """
    Check if a snapshot is a transformers text model.

    Only excludes (returns False) for models with STRONG evidence of being
    diffusers/generation pipelines. Uses conservative heuristics to avoid
    false negatives on multimodal LLMs with tokenizers.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        True if this looks like a transformers text model, False otherwise (N/A)
    """
    
    diffusers_markers = [
        "model_index.json",  
        "scheduler",  
    ]
    if any(
        os.path.exists(os.path.join(snapshot_dir, marker))
        for marker in diffusers_markers
    ):
        return False

    config_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.exists(config_path):
        
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        
        model_type = config.get("_class_name") or config.get("model_type")
        if model_type:
            model_type_lower = str(model_type).lower()
            
            if any(
                keyword in model_type_lower
                for keyword in [
                    "diffusion",
                    "unet",
                    "vae",
                    "controlnet",
                    "stable-diffusion",
                    "latent-diffusion",
                ]
            ):
                return False

        
        architectures = config.get("architectures", [])
        if architectures:
            arch_str = " ".join(architectures).lower()
            
            
            for keyword in [
                "diffusion",
                "unet2d",
                "unet3d",
                "vaedecoder",  
                "vaeencoder",
                "controlnet",
                "autoencoder",
                "ditmodel",  
                "pixart",  
            ]:
                if keyword in arch_str:
                    return False

        
        
        model_name = config.get("_name_or_path", "").lower()

        if any(
            keyword in model_name
            for keyword in [
                "image-edit-",  
                "-image-editing",
                "dit-",  
                "pixart-",  
            ]
        ):
            
            has_tokenizer = any(
                os.path.exists(os.path.join(snapshot_dir, fname))
                for fname in ["tokenizer.json", "tokenizer.model", "tiktoken.model"]
            )
            if not has_tokenizer:
                
                return False

        
        
        
        return True

    except (json.JSONDecodeError, OSError, KeyError):
        
        return True


def scan_weight_files(snapshot_dir):
    """
    Scan for weight files in a snapshot.

    Returns:
        List of weight file paths, or empty list if scan fails
    """
    weight_files = []

    
    index_patterns = ["*.safetensors.index.json", "pytorch_model.bin.index.json"]
    index_files = []
    for pattern in index_patterns:
        index_files.extend(glob.glob(os.path.join(snapshot_dir, pattern)))

    
    for index_file in index_files:
        if index_file.endswith(".safetensors.index.json"):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                for weight_file in set(weight_map.values()):
                    weight_path = os.path.join(snapshot_dir, weight_file)
                    if os.path.exists(weight_path):
                        weight_files.append(weight_path)
            except Exception as e:
                print(
                    f"  Warning: Failed to parse index {os.path.basename(index_file)}: {e}"
                )

    
    if not weight_files:
        matched = glob.glob(
            os.path.join(snapshot_dir, "**/*.safetensors"), recursive=True
        )
        MAX_WEIGHT_FILES = 1000
        if len(matched) > MAX_WEIGHT_FILES:
            print(
                f"  Warning: Too many safetensors files ({len(matched)} > {MAX_WEIGHT_FILES})"
            )
            return []

        for f in matched:
            if os.path.exists(f):  
                weight_files.append(f)

    return weight_files


def validate_snapshot(model_name, snapshot_dir, weight_files, validated_cache):
    """
    Validate a snapshot and return detailed status.

    Uses in-process cache to avoid duplicate validation within the same run.

    Args:
        model_name: Model identifier
        snapshot_dir: Path to snapshot directory
        weight_files: List of weight files to validate
        validated_cache: Dict to track already-validated snapshots in this run

    Returns:
        Tuple of (result, reason):
        - (True, None) if validation passed
        - (False, reason_str) if validation failed
        - (None, None) if skipped (already validated in this run)
    """
    
    if snapshot_dir in validated_cache:
        return None, None  

    try:
        
        is_complete, reason = validate_cache_with_detailed_reason(
            snapshot_dir=snapshot_dir,
            weight_files=weight_files,
            model_name_or_path=model_name,
        )

        
        validated_cache[snapshot_dir] = (is_complete, reason)

        return is_complete, reason

    except Exception as e:
        error_msg = f"Validation raised exception: {e}"
        return False, error_msg


def main():
    start_time = time.time()

    print("=" * 70)
    print("CI_OFFLINE: Pre-validating cached HuggingFace models")
    print("=" * 70)
    print(f"Max time: {MAX_VALIDATION_TIME_SECONDS}s")
    print()

    print("Scanning HuggingFace cache for models...")
    snapshots = find_all_hf_snapshots()

    if not snapshots:
        print("No cached models found, skipping validation")
        print("=" * 70)
        return

    print(f"Found {len(snapshots)} snapshot(s) in cache")
    print()

    validated_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0

    
    validated_cache = {}

    for model_name, snapshot_dir in snapshots:
        
        elapsed = time.time() - start_time
        if elapsed > MAX_VALIDATION_TIME_SECONDS:
            print()
            print(
                f"Time limit reached ({elapsed:.1f}s > {MAX_VALIDATION_TIME_SECONDS}s)"
            )
            print(
                f"Stopping validation, {len(snapshots) - processed_count} snapshots remaining"
            )
            break

        snapshot_hash = os.path.basename(snapshot_dir)
        print(
            f"[{processed_count + 1}/{len(snapshots)}] {model_name} ({snapshot_hash[:8]}...)"
        )
        processed_count += 1

        
        model_index_path = os.path.join(snapshot_dir, "model_index.json")
        is_diffusion_model = os.path.exists(model_index_path)

        if is_diffusion_model:
            
            try:
                is_valid, reason = _validate_diffusion_model(snapshot_dir)

                if is_valid:
                    print("  PASS (diffusion) - Cache complete & valid")
                    validated_count += 1
                else:
                    print(f"  FAIL (diffusion) - {reason}")
                    failed_count += 1

            except Exception as e:
                print(f"  FAIL (diffusion) - Validation raised exception: {e}")
                failed_count += 1

            continue

        
        
        if not is_transformers_text_model(snapshot_dir):
            
            print(
                "  SKIP (unknown type) - Not a diffusers pipeline or transformers model"
            )
            skipped_count += 1
            continue

        
        weight_files = scan_weight_files(snapshot_dir)

        if not weight_files:
            print("  SKIP (no weights) - empty or incomplete download")
            skipped_count += 1
            continue

        
        try:
            result, reason = validate_snapshot(
                model_name, snapshot_dir, weight_files, validated_cache
            )

            if result is True:
                print("  PASS - Cache complete & valid")
                validated_count += 1
            elif result is False:
                
                if reason:
                    print(f"  FAIL (incomplete) - {reason}")
                else:
                    print("  FAIL (incomplete) - cache validation failed")
                failed_count += 1
            else:  
                print("  SKIP (already validated in this run)")
                skipped_count += 1

        except Exception as e:
            print(f"  FAIL (error) - Validation raised exception: {e}")
            failed_count += 1

    elapsed_total = time.time() - start_time

    print()
    print("=" * 70)
    print(f"Validation summary (completed in {elapsed_total:.1f}s):")
    print(f"  PASS (complete & valid):      {validated_count}")
    print(f"  FAIL (incomplete/corrupted):  {failed_count}")
    print(f"  SKIP (no weights/duplicate):  {skipped_count}")
    print(f"  Total processed:              {processed_count}/{len(snapshots)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
