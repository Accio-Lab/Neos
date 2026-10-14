"""
CI-specific weight validation and cache cleanup utilities.

This module contains validation and cleanup logic that is ONLY used in CI environments.
These functions handle:
- Validating safetensors files for corruption
- Checking for missing shards in sharded models
- Cleaning up corrupted files (selective or full cache deletion)
- Automatic retry logic for corrupted downloads
- Validating config/tokenizer files completeness to enable offline mode

For regular users, weight_utils.py provides simple download functionality without
the overhead of validation and automatic cleanup. The CI-specific behavior is
gated by is_in_ci() checks in weight_utils.py.
"""

import glob as glob_module
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from typing import List, Optional, Tuple

import safetensors

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)





VALIDATION_MARKER_VERSION = "5"


def _remote_file_exists(
    repo_id: str, filename: str, revision: Optional[str], allow_remote_check: bool
) -> Optional[bool]:
    """
    Check if a file exists on Hugging Face Hub for a specific revision.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        filename: File name to check (e.g., "hf_quant_config.json")
        revision: Git revision (commit hash, branch, or tag). None means default branch.
        allow_remote_check: Whether remote checks are allowed (e.g., CI validation phase)

    Returns:
        True if file exists on hub, False if it doesn't exist, None if we cannot determine
        (network error or remote check not allowed - be conservative and assume incomplete)
    """
    if not allow_remote_check:
        logger.debug(
            "Remote check disabled for %s/%s, returning None (unknown)",
            repo_id,
            filename,
        )
        return None

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        exists = api.file_exists(repo_id=repo_id, filename=filename, revision=revision)
        logger.debug(
            "Remote file check: %s/%s (revision=%s) exists=%s",
            repo_id,
            filename,
            revision or "default",
            exists,
        )
        return exists
    except Exception as e:
        
        
        logger.debug(
            "Failed to check remote file existence for %s/%s (revision=%s): %s. "
            "Will treat as optional.",
            repo_id,
            filename,
            revision or "default",
            e,
        )
        return None


def _get_validation_marker_path(snapshot_dir: str) -> Optional[str]:
    """
    Get the path to validation marker file for a snapshot.

    Marker is stored in /tmp to avoid permission issues with HF cache directory.
    Marker key is sha256(snapshot_dir) to avoid any collisions regardless of
    model_name_or_path format.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to marker file or None if snapshot_dir is invalid
    """
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    
    
    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")

    
    
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]

    
    return f"/tmp/sglang_hf_validation_{dir_hash}.json"


def _get_per_run_marker_dir() -> str:
    """
    Get the directory for per-run validation markers.

    These markers are specific to the current CI run and are not shared across
    runners. They are stored in a temporary directory that is cleaned up after
    the run completes.

    Returns:
        Path to per-run marker directory
    """
    
    base_dir = os.environ.get("RUNNER_TEMP", os.environ.get("TMPDIR", "/tmp"))
    marker_dir = os.path.join(base_dir, "sglang_ci_offline_markers")
    os.makedirs(marker_dir, exist_ok=True)
    return marker_dir


def _get_per_run_marker_path(snapshot_dir: str) -> Optional[str]:
    """
    Get the path to per-run validation marker file for a snapshot.

    Per-run markers are specific to the current CI run and are not shared
    across runners. This prevents cross-runner cache state pollution.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to per-run marker file or None if snapshot_dir is invalid
    """
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]

    marker_dir = _get_per_run_marker_dir()
    return os.path.join(marker_dir, f"{dir_hash}.json")


def _read_per_run_marker(snapshot_dir: str) -> Optional[dict]:
    """
    Read per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict if exists and valid, None otherwise
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if not marker_path or not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)

        
        if not isinstance(marker, dict):
            return None

        required_keys = ["timestamp", "model_id", "snapshot_hash", "validation_passed"]
        if not all(k in marker for k in required_keys):
            return None

        if marker.get("validation_passed") is not True:
            return None

        return marker

    except Exception as e:
        logger.debug("Failed to read per-run marker from %s: %s", marker_path, e)
        return None


def _write_per_run_marker(
    snapshot_dir: str, model_id: str, required_files: Optional[list] = None
) -> None:
    """
    Write per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
        model_id: Model identifier
        required_files: List of required files that were validated
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if not marker_path:
        logger.debug("Cannot write per-run marker: invalid snapshot_dir")
        return

    from datetime import datetime

    snapshot_hash = os.path.basename(snapshot_dir)

    marker = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "snapshot_hash": snapshot_hash,
        "validation_passed": True,
        "required_files": required_files or [],
    }

    try:
        marker_dir = os.path.dirname(marker_path)
        os.makedirs(marker_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_path = f.name
            json.dump(marker, f, indent=2)

        os.replace(temp_path, marker_path)
        logger.debug("Wrote per-run marker to %s", marker_path)
    except Exception as e:
        logger.warning("Failed to write per-run marker to %s: %s", marker_path, e)
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _remove_per_run_marker(snapshot_dir: str) -> None:
    """
    Remove per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if marker_path and os.path.exists(marker_path):
        try:
            os.remove(marker_path)
            logger.debug("Removed per-run marker: %s", marker_path)
        except Exception as e:
            logger.warning("Failed to remove per-run marker %s: %s", marker_path, e)


def _read_validation_marker(snapshot_dir: str) -> Optional[dict]:
    """
    Read validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict with keys: version, validated_at, validation_passed
        None if marker doesn't exist or is invalid or validation_passed is not True
    """
    marker_path = _get_validation_marker_path(snapshot_dir)
    if not marker_path:
        return None

    if not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)

        
        if not isinstance(marker, dict):
            return None

        required_keys = ["version", "validated_at", "validation_passed"]
        if not all(key in marker for key in required_keys):
            return None

        
        if marker["version"] != VALIDATION_MARKER_VERSION:
            logger.debug(
                "Validation marker version mismatch: %s != %s, will re-validate",
                marker["version"],
                VALIDATION_MARKER_VERSION,
            )
            return None

        
        
        
        if marker.get("validation_passed") is not True:
            logger.debug(
                "Validation marker has validation_passed=%s, treating as invalid",
                marker.get("validation_passed"),
            )
            return None

        return marker
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to read validation marker at %s: %s", marker_path, e)
        return None


def _write_validation_marker(snapshot_dir: str, passed: bool) -> None:
    """
    Write validation marker for a snapshot (atomic write).

    IMPORTANT: We only cache successful validations. Failed validations are NOT
    cached to allow retry after files are downloaded.

    Args:
        snapshot_dir: Path to snapshot directory
        passed: Whether validation passed
    """
    if not passed:
        
        return

    marker_path = _get_validation_marker_path(snapshot_dir)
    if not marker_path:
        logger.debug("Cannot write marker: invalid snapshot_dir")
        return

    from datetime import datetime

    marker = {
        "version": VALIDATION_MARKER_VERSION,
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "validation_passed": passed,
    }

    try:
        
        marker_dir = os.path.dirname(marker_path)
        os.makedirs(marker_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_path = f.name
            json.dump(marker, f, indent=2)

        
        os.replace(temp_path, marker_path)
        logger.debug("Wrote validation marker to %s (passed=%s)", marker_path, passed)
    except Exception as e:
        logger.warning("Failed to write validation marker to %s: %s", marker_path, e)
        
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _validate_json_file(file_path: str, file_name: str) -> bool:
    """
    Validate that a JSON file exists, is non-empty, and can be parsed.

    Args:
        file_path: Path to the JSON file
        file_name: Name of the file (for logging)

    Returns:
        True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.debug("CI cache validation: %s not found at %s", file_name, file_path)
        return False

    if not os.path.isfile(file_path):
        logger.warning(
            "CI cache validation: %s is not a file: %s", file_name, file_path
        )
        return False

    
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning("CI cache validation: %s is empty: %s", file_name, file_path)
            return False
    except OSError as e:
        logger.warning("CI cache validation: Cannot get size of %s: %s", file_name, e)
        return False

    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        logger.warning(
            "CI cache validation: %s is not valid JSON: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False
    except Exception as e:
        logger.warning(
            "CI cache validation: Failed to read %s: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False


def _validate_config_and_tokenizer_files(
    snapshot_dir: str,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_remote_check: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that critical config and tokenizer files exist and are valid.

    This checks for:
    - config.json (required)
    - tokenizer_config.json (required)
    - generation_config.json (optional but validated if present)
    - hf_quant_config.json (conditionally required based on Hub) - for FP4/FP8/ModelOpt
    - quantize_config.json / quant_config.json (optional but validated if present) - for AWQ/GPTQ
    - params.json (optional but validated if present) - for Mistral native format
    - preprocessor_config.json (optional but validated if present) - for vision models
    - trust_remote_code dynamic modules (required if auto_map present in config.json)
    - At least one tokenizer file: tokenizer.json, tokenizer.model, or tiktoken.model

    Args:
        snapshot_dir: Path to the model snapshot directory
        model_id: Model repository ID (e.g., "meta-llama/Llama-2-7b-hf"), used for remote checks
        revision: Git revision (commit hash), used for remote checks
        allow_remote_check: Whether to check Hub for file existence to determine requirements

    Returns:
        Tuple of (is_valid, missing_files)
        - is_valid: True if all required files are present and valid
        - missing_files: List of missing or invalid file names
    """
    missing_files = []

    
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for file_name in required_files:
        file_path = os.path.join(snapshot_dir, file_name)
        if not _validate_json_file(file_path, file_name):
            missing_files.append(file_name)

    
    generation_config_path = os.path.join(snapshot_dir, "generation_config.json")
    if os.path.exists(generation_config_path):
        if not _validate_json_file(generation_config_path, "generation_config.json"):
            missing_files.append("generation_config.json (exists but invalid)")

    
    
    
    hf_quant_config_path = os.path.join(snapshot_dir, "hf_quant_config.json")
    local_hf_quant_exists = os.path.exists(hf_quant_config_path)

    
    
    
    remote_hf_quant_exists = None
    is_hf_repo = (
        model_id is not None
        and "/" in model_id
        and not os.path.isabs(model_id)
        and not model_id.startswith("/")
    )
    if is_hf_repo and allow_remote_check:
        remote_hf_quant_exists = _remote_file_exists(
            repo_id=model_id,
            filename="hf_quant_config.json",
            revision=revision,
            allow_remote_check=allow_remote_check,
        )

    
    if remote_hf_quant_exists is True:
        
        if not local_hf_quant_exists:
            missing_files.append(
                f"hf_quant_config.json (required: exists on Hub for revision {revision or 'default'} but missing locally)"
            )
            log_info_on_rank0(
                logger,
                f"Hub has hf_quant_config.json for {model_id} revision {revision or 'default'} "
                f"but local snapshot missing it. Cache incomplete, will not write marker.",
            )
        elif not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
            missing_files.append("hf_quant_config.json (exists but invalid)")
    elif remote_hf_quant_exists is False:
        
        
        if local_hf_quant_exists:
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")
    else:
        
        
        if local_hf_quant_exists:
            
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")
        
        logger.debug(
            "Cannot verify hf_quant_config.json on Hub for %s (revision=%s), "
            "treating as optional since remote status unknown",
            model_id or "unknown",
            revision or "default",
        )

    
    
    
    for quant_config_name in ["quantize_config.json", "quant_config.json"]:
        quant_config_path = os.path.join(snapshot_dir, quant_config_name)
        if os.path.exists(quant_config_path):
            if not _validate_json_file(quant_config_path, quant_config_name):
                missing_files.append(f"{quant_config_name} (exists but invalid)")
            break  

    
    
    
    params_json_path = os.path.join(snapshot_dir, "params.json")
    if os.path.exists(params_json_path):
        if not _validate_json_file(params_json_path, "params.json"):
            missing_files.append("params.json (exists but invalid)")

    
    
    
    preprocessor_config_path = os.path.join(snapshot_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_config_path):
        if not _validate_json_file(
            preprocessor_config_path, "preprocessor_config.json"
        ):
            missing_files.append("preprocessor_config.json (exists but invalid)")

    
    
    
    config_path = os.path.join(snapshot_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            auto_map = config.get("auto_map", {})
            if auto_map and isinstance(auto_map, dict):
                
                
                
                custom_files = set()
                for key, value in auto_map.items():
                    if isinstance(value, str) and "." in value:
                        
                        module_name = value.split(".")[0]
                        custom_files.add(f"{module_name}.py")

                
                
                
                
                
                for custom_file in custom_files:
                    custom_file_path = os.path.join(snapshot_dir, custom_file)
                    if not os.path.exists(custom_file_path):
                        missing_files.append(
                            f"{custom_file} (required for trust_remote_code)"
                        )
                        logger.debug(
                            f"Custom module file not in snapshot: {custom_file} for {snapshot_dir}"
                        )
                    elif not os.path.isfile(custom_file_path):
                        missing_files.append(f"{custom_file} (exists but not a file)")
        except (json.JSONDecodeError, OSError, KeyError) as e:
            
            logger.debug("Failed to check auto_map in config.json: %s", e)

    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
    ]

    tokenizer_found = False
    for tokenizer_file in tokenizer_files:
        tokenizer_path = os.path.join(snapshot_dir, tokenizer_file)
        if os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path):
            
            if tokenizer_file == "tokenizer.json":
                if _validate_json_file(tokenizer_path, tokenizer_file):
                    tokenizer_found = True
                    break
            else:
                
                try:
                    if os.path.getsize(tokenizer_path) > 0:
                        tokenizer_found = True
                        break
                except OSError:
                    pass

    if not tokenizer_found:
        missing_files.append("tokenizer file")

    is_valid = len(missing_files) == 0
    return is_valid, missing_files


def ci_validate_cache_and_enable_offline_if_complete(
    snapshot_dir: str,
    weight_files: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate local cache completeness (config/tokenizer/weights) and determine
    if offline mode can be safely enabled.

    This function uses a snapshot-level marker to cache validation results,
    so the heavy validation is done at most once per snapshot per runner.

    This function checks:
    1. Validation marker (if exists and version matches, skip re-validation)
    2. Config and tokenizer files (config.json, tokenizer_config.json, etc.)
    3. Weight files (safetensors shards, index files, corruption check)

    If all are present and valid, it returns True to signal that offline
    mode can be safely enabled.

    IMPORTANT: This should be called BEFORE any HF operations, and if it
    returns True, the caller should set HF_HUB_OFFLINE=1 for the server
    subprocess env ONLY (not global environment).

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate (must be non-empty)
        model_name_or_path: Model identifier for logging

    Returns:
        True if cache is complete and offline mode can be enabled, False otherwise
    """
    
    if not weight_files:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: No weight files provided, skip offline, keep online allowed - {model_name_or_path}",
        )
        return False

    
    
    marker = _read_validation_marker(snapshot_dir)
    if marker is not None:
        marker_path = _get_validation_marker_path(snapshot_dir)
        marker_name = os.path.basename(marker_path) if marker_path else "unknown"
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Marker hit (marker={marker_name}), skip re-validation, offline mode will be enabled - {model_name_or_path}",
        )
        return True

    
    

    
    
    revision = os.path.basename(snapshot_dir)

    
    
    import huggingface_hub.constants

    allow_remote_check = not huggingface_hub.constants.HF_HUB_OFFLINE

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: No marker found, performing full validation "
        f"(snapshot={revision}, allow_remote_check={allow_remote_check}) - {model_name_or_path}",
    )

    
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Missing config/tokenizer files {missing_config_files}, skip offline, keep online allowed - {model_name_or_path}",
        )
        
        return False

    
    
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)
    if not weights_valid:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Weight validation failed ({error_msg}), skip offline, keep online allowed - {model_name_or_path}",
        )
        
        return False

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: Cache validation PASSED, offline mode will be enabled - {model_name_or_path}",
    )

    
    
    _write_validation_marker(snapshot_dir, passed=True)
    return True


def _infer_component_type(component_name: str, component_info: list) -> str:
    """
    Infer component type from component name and info.

    Args:
        component_name: Name of the component (e.g., "scheduler", "tokenizer")
        component_info: Component info from model_index.json (e.g., ["diffusers", "SchedulerClass"])

    Returns:
        Component type string for validation rules
    """
    
    name_lower = component_name.lower()

    
    if "scheduler" in name_lower:
        return "scheduler"
    elif "tokenizer" in name_lower:
        return "tokenizer"
    elif "image_processor" in name_lower:
        return "image_processor"
    elif "feature_extractor" in name_lower:
        return "feature_extractor"
    elif "processor" in name_lower:
        return "processor"
    else:
        
        return "model"


def _check_component_config(
    component_dir: str, component_type: str
) -> Tuple[bool, List[str]]:
    """
    Check if component has required config files based on type.

    Args:
        component_dir: Path to component directory
        component_type: Type of component (scheduler, tokenizer, processor, model, etc.)

    Returns:
        Tuple of (has_valid_config, list_of_candidates_tried)
    """
    if component_type == "scheduler":
        
        candidates = ["scheduler_config.json", "config.json"]
        for candidate in candidates:
            candidate_path = os.path.join(component_dir, candidate)
            if _validate_json_file(candidate_path, candidate):
                return True, candidates
        return False, candidates

    elif component_type == "tokenizer":
        
        
        
        
        
        candidates = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json+merges.txt",
        ]

        
        tokenizer_json_path = os.path.join(component_dir, "tokenizer.json")
        if _validate_json_file(tokenizer_json_path, "tokenizer.json"):
            return True, candidates

        
        tokenizer_model_path = os.path.join(component_dir, "tokenizer.model")
        if os.path.exists(tokenizer_model_path) and os.path.isfile(
            tokenizer_model_path
        ):
            try:
                if os.path.getsize(tokenizer_model_path) > 0:
                    return True, candidates
            except OSError:
                pass

        
        vocab_path = os.path.join(component_dir, "vocab.json")
        merges_path = os.path.join(component_dir, "merges.txt")
        if _validate_json_file(vocab_path, "vocab.json") and os.path.exists(
            merges_path
        ):
            return True, candidates

        return False, candidates

    elif component_type in ["processor", "feature_extractor", "image_processor"]:
        
        candidates = ["preprocessor_config.json", "config.json"]
        for candidate in candidates:
            candidate_path = os.path.join(component_dir, candidate)
            if _validate_json_file(candidate_path, candidate):
                return True, candidates
        return False, candidates

    else:
        
        candidates = ["config.json"]
        config_path = os.path.join(component_dir, "config.json")
        if _validate_json_file(config_path, "config.json"):
            return True, candidates
        return False, candidates


def _check_component_weights(component_dir: str) -> bool:
    """
    Check if component directory has weight files.

    Args:
        component_dir: Path to component directory

    Returns:
        True if weight files found, False otherwise
    """
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]

    for pattern in weight_patterns:
        weight_files = glob_module.glob(os.path.join(component_dir, pattern))
        if weight_files:
            return True

    return False


def _format_component_list(components: List[str], max_show: int = 5) -> str:
    """
    Format component list with truncation.

    Args:
        components: List of component names
        max_show: Maximum number to show before truncating

    Returns:
        Formatted string like "comp1, comp2, comp3" or "comp1, comp2, +3 more"
    """
    if len(components) <= max_show:
        return ", ".join(components)
    else:
        shown = components[:max_show]
        remaining = len(components) - max_show
        return f"{', '.join(shown)}, +{remaining} more"


def _validate_diffusion_model(
    snapshot_dir: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate diffusion model (diffusers pipeline) cache completeness.

    This validation is based on model_index.json as the single source of truth.
    Error reporting uses coarse-grained error codes unless verbose mode is enabled.

    Error codes:
    - DIFFUSERS_INVALID_INDEX: model_index.json missing or corrupted
    - DIFFUSERS_INVALID_COMPONENTS: model_index.json has no valid components
    - DIFFUSERS_MISSING_COMPONENT: component directory or config missing
    - DIFFUSERS_MISSING_WEIGHTS: component weights missing

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if validation passed
        - (False, error_code_with_components) if validation failed
    """
    
    verbose = os.environ.get("SGLANG_CI_VALIDATE_VERBOSE") == "1"

    
    model_index_path = os.path.join(snapshot_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json not found"

    
    try:
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if verbose:
            return False, f"DIFFUSERS_INVALID_INDEX: model_index.json parse error - {e}"
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json corrupted"

    
    components = {
        k: v
        for k, v in model_index.items()
        if not k.startswith("_") and isinstance(v, list)
    }

    if not components:
        return False, "DIFFUSERS_INVALID_COMPONENTS: no valid components defined"

    
    missing_dirs = []
    missing_configs = []
    missing_configs_verbose = []
    missing_weights = []

    
    for component_name, component_info in components.items():
        component_dir = os.path.join(snapshot_dir, component_name)

        
        if not os.path.isdir(component_dir):
            missing_dirs.append(component_name)
            continue

        
        component_type = _infer_component_type(component_name, component_info)

        
        has_valid_config, config_candidates = _check_component_config(
            component_dir, component_type
        )

        if not has_valid_config:
            missing_configs.append(component_name)
            if verbose:
                candidates_str = ", ".join(config_candidates)
                missing_configs_verbose.append(
                    f"{component_name} (tried: {candidates_str})"
                )
            continue

        
        
        needs_weights = component_type not in [
            "scheduler",
            "tokenizer",
            "processor",
            "feature_extractor",
            "image_processor",
        ]

        if needs_weights:
            has_weights = _check_component_weights(component_dir)
            if not has_weights:
                missing_weights.append(component_name)

    
    if missing_dirs or missing_configs or missing_weights:
        errors = []

        if missing_dirs:
            dir_str = _format_component_list(missing_dirs)
            if verbose:
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (dirs): {dir_str}")
            else:
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(dir): {dir_str}")

        if missing_configs:
            if verbose:
                config_str = "; ".join(missing_configs_verbose)
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (configs): {config_str}")
            else:
                config_str = _format_component_list(missing_configs)
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(cfg): {config_str}")

        if missing_weights:
            weight_str = _format_component_list(missing_weights)
            errors.append(f"DIFFUSERS_MISSING_WEIGHTS: {weight_str}")

        return False, " | ".join(errors)

    return True, None


def validate_cache_with_detailed_reason(
    snapshot_dir: str, weight_files: List[str], model_name_or_path: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate cache and return detailed reason for failure.

    This function performs validation without relying on shared validation markers.
    Used by prevalidate_cached_models.py to provide detailed feedback.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate
        model_name_or_path: Model identifier for logging

    Returns:
        Tuple of (success, reason):
        - (True, None) if validation passed
        - (False, reason_str) if validation failed with specific reason
    """
    
    if not weight_files:
        return False, "No weight files provided"

    
    revision = os.path.basename(snapshot_dir)

    
    allow_remote_check = os.environ.get("HF_HUB_OFFLINE") != "1"

    
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:
        missing_files_str = ", ".join(missing_config_files)
        return False, f"Missing config/tokenizer files: {missing_files_str}"

    
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)
    if not weights_valid:
        return False, f"Weight validation failed: {error_msg}"

    
    return True, None


def validate_cache_lightweight(
    snapshot_dir: str, requires_hf_quant_config: bool = False
) -> bool:
    """
    Lightweight runtime validation for cache completeness.

    This is used during test runs to ensure the current runner's cache
    is complete before enabling offline mode. Much faster than full validation
    as it only checks file existence, not corruption.

    Args:
        snapshot_dir: Path to the model snapshot directory
        requires_hf_quant_config: If True, hf_quant_config.json must exist
                                  (required for modelopt quantization)

    Returns:
        True if cache is complete, False otherwise
    """
    
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for fname in required_files:
        if not os.path.exists(os.path.join(snapshot_dir, fname)):
            return False

    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
    ]

    has_tokenizer = any(
        os.path.exists(os.path.join(snapshot_dir, fname)) for fname in tokenizer_files
    )
    if not has_tokenizer:
        return False

    
    
    
    config_path = os.path.join(snapshot_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            auto_map = config.get("auto_map", {})
            if auto_map and isinstance(auto_map, dict):
                
                
                
                custom_files = set()
                for key, value in auto_map.items():
                    if isinstance(value, str) and "." in value:
                        
                        module_name = value.split(".")[0]
                        custom_files.add(f"{module_name}.py")

                
                for custom_file in custom_files:
                    custom_file_path = os.path.join(snapshot_dir, custom_file)
                    if not os.path.exists(custom_file_path):
                        logger.debug(
                            "Custom module file not in snapshot: %s for %s",
                            custom_file,
                            snapshot_dir,
                        )
                        return False
                    elif not os.path.isfile(custom_file_path):
                        logger.debug(
                            "Custom module path exists but not a file: %s",
                            custom_file_path,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:
            
            logger.debug("Failed to check auto_map in config.json: %s", e)

    
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    has_index = os.path.exists(index_path)

    if has_index:
        
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if weight_map:
                
                required_shards = set(weight_map.values())
                for shard_name in required_shards:
                    shard_path = os.path.join(snapshot_dir, shard_name)
                    if not os.path.exists(shard_path):
                        logger.debug(
                            "Index validation failed: missing shard %s in %s",
                            shard_name,
                            snapshot_dir,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug("Failed to validate index file %s: %s", index_path, e)
            return False
    else:
        
        safetensors_files = glob_module.glob(
            os.path.join(snapshot_dir, "*.safetensors")
        )
        if not safetensors_files:
            return False

        
        
        shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.safetensors$")
        shard_groups = {}

        for f in safetensors_files:
            base_name = os.path.basename(f)
            match = shard_pattern.match(base_name)
            if match:
                prefix = match.group(1)
                shard_id = int(match.group(2))
                total_shards = int(match.group(3))
                group_key = f"{prefix}-of-{total_shards}"

                if group_key not in shard_groups:
                    shard_groups[group_key] = {
                        "total": total_shards,
                        "found_shards": set(),
                    }
                shard_groups[group_key]["found_shards"].add(shard_id)

        
        for group_key, group_info in shard_groups.items():
            total_shards = group_info["total"]
            found_shards = group_info["found_shards"]
            expected_shards = set(range(1, total_shards + 1))
            missing_shards = expected_shards - found_shards

            if missing_shards:
                logger.debug(
                    "Shard validation failed: missing shards %s in %s for %s",
                    sorted(missing_shards),
                    group_key,
                    snapshot_dir,
                )
                return False

    
    if requires_hf_quant_config:
        hf_quant_path = os.path.join(snapshot_dir, "hf_quant_config.json")
        if not os.path.exists(hf_quant_path):
            return False

    return True


def _validate_safetensors_file(file_path: str) -> bool:
    """
    Validate that a safetensors file is readable and not corrupted.

    Args:
        file_path: Path to the safetensors file

    Returns:
        True if the file is valid, False if corrupted
    """
    try:
        
        
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            
            _ = list(f.keys())
        return True
    except Exception as e:
        logger.warning(
            "Corrupted safetensors file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False


def _validate_pytorch_bin_file(file_path: str) -> bool:
    """
    Validate that a PyTorch .bin file is readable and not corrupted.

    This catches corruption issues like truncated downloads or invalid archives
    that would cause errors like:
    "RuntimeError: PytorchStreamReader failed reading file data/X: invalid header
    or archive is corrupted"

    Args:
        file_path: Path to the .bin file

    Returns:
        True if the file is valid, False if corrupted
    """
    try:
        import torch

        
        
        torch.load(file_path, map_location="cpu", weights_only=True, mmap=False)
        return True
    except Exception as e:
        logger.warning(
            "Corrupted PyTorch bin file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Check if all files listed in safetensors index files actually exist on disk.

    This catches cases where the snapshot directory exists but files are missing
    (e.g., due to incomplete downloads or corrupted cache).

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (all_exist, error_message)
    """
    
    index_files = [
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")
    ]

    if not index_files:
        
        return True, None

    for index_file in index_files:
        index_path = os.path.join(snapshot_dir, index_file)

        
        if os.path.islink(index_path) and not os.path.exists(index_path):
            
            try:
                blob_path = os.path.realpath(index_path)
                os.remove(index_path)
                logger.warning(
                    "Removed broken index symlink: %s (blob missing)", index_file
                )
                
                if os.path.exists(blob_path):
                    os.remove(blob_path)
            except Exception as e:
                logger.error("Failed to remove broken symlink %s: %s", index_file, e)
            return (
                False,
                f"Broken index file symlink: {index_file} (cleaned up, will re-download)",
            )

        try:
            with open(index_path) as f:
                index_data = json.load(f)

            weight_map = index_data.get("weight_map", {})
            if not weight_map:
                continue

            
            required_files = set(weight_map.values())
            missing_files = []

            for file_name in required_files:
                file_path = os.path.join(snapshot_dir, file_name)
                
                if not os.path.exists(file_path):
                    missing_files.append(file_name)

            if missing_files:
                return (
                    False,
                    f"Missing {len(missing_files)} file(s) from index {index_file}: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",
                )

        except FileNotFoundError as e:
            
            logger.warning("Failed to read index file %s: %s", index_file, e)
            return (
                False,
                f"Index file {index_file} unreadable (will re-download)",
            )
        except Exception as e:
            logger.warning("Failed to read index file %s: %s", index_file, e)
            continue

    return True, None


def _validate_sharded_model(
    snapshot_dir: str, weight_files: List[str]
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate that all model shards are present and not corrupted.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths

    Returns:
        Tuple of (is_valid, error_message, corrupted_files)
        - corrupted_files: List of file paths that are corrupted (for selective cleanup)
    """
    
    
    index_check_valid, index_error = _check_index_files_exist(snapshot_dir)
    if not index_check_valid:
        return False, index_error, []

    
    shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.(safetensors|bin)")

    
    shard_groups = {}
    for f in weight_files:
        base_name = os.path.basename(f)
        match = shard_pattern.match(base_name)
        if match:
            prefix = match.group(1)
            total_shards_str = match.group(3)
            suffix = match.group(4)

            group_key = f"{prefix}-of-{total_shards_str}.{suffix}"
            if group_key not in shard_groups:
                shard_groups[group_key] = {
                    "prefix": prefix,
                    "total": int(total_shards_str),
                    "suffix": suffix,
                    "found_shards": [],
                    "files": [],
                }

            shard_id = int(match.group(2))
            shard_groups[group_key]["found_shards"].append(shard_id)
            shard_groups[group_key]["files"].append(f)

    
    corrupted_files = []

    
    for group_key, group_info in shard_groups.items():
        total_shards = group_info["total"]
        found_shards = set(group_info["found_shards"])
        expected_shards = set(range(1, total_shards + 1))

        
        missing_shards = expected_shards - found_shards
        if missing_shards:
            return (
                False,
                f"Missing shards in {group_key}: {sorted(missing_shards)}",
                [],
            )

        
        if group_info["suffix"] == "safetensors":
            for f in group_info["files"]:
                if not _validate_safetensors_file(f):
                    corrupted_files.append(f)
        elif group_info["suffix"] == "bin":
            for f in group_info["files"]:
                if not _validate_pytorch_bin_file(f):
                    corrupted_files.append(f)

        
        if group_info["suffix"] == "safetensors":
            index_file = os.path.join(
                snapshot_dir, f"{group_info['prefix']}.safetensors.index.json"
            )
            if not os.path.exists(index_file):
                return (
                    False,
                    f"Missing index file: {os.path.basename(index_file)}",
                    [],
                )

    if corrupted_files:
        return (
            False,
            f"Corrupted shard files: {[os.path.basename(f) for f in corrupted_files]}",
            corrupted_files,
        )

    return True, None, []


def _cleanup_corrupted_files_selective(
    model_name_or_path: str, corrupted_files: List[str]
) -> int:
    """
    Selectively remove corrupted files and their blobs to force re-download.

    This is more efficient than removing the entire model cache as it only
    re-downloads corrupted files rather than the entire model.

    Args:
        model_name_or_path: Model identifier
        corrupted_files: List of corrupted file paths (symlinks in snapshot)

    Returns:
        Number of files successfully cleaned up
    """
    cleaned_count = 0

    for file_path in corrupted_files:
        try:
            
            if os.path.islink(file_path):
                blob_path = os.path.realpath(file_path)

                
                os.remove(file_path)
                logger.info(
                    "Removed corrupted symlink: %s", os.path.basename(file_path)
                )

                
                if os.path.exists(blob_path):
                    os.remove(blob_path)
                    logger.info(
                        "Removed corrupted blob: %s", os.path.basename(blob_path)
                    )

                cleaned_count += 1
            elif os.path.exists(file_path):
                
                os.remove(file_path)
                logger.info("Removed corrupted file: %s", os.path.basename(file_path))
                cleaned_count += 1

        except Exception as e:
            logger.error(
                "Failed to remove corrupted file %s: %s",
                os.path.basename(file_path),
                e,
            )

    if cleaned_count > 0:
        logger.warning(
            "Removed %d corrupted file(s) for %s. "
            "These will be re-downloaded on next load.",
            cleaned_count,
            model_name_or_path,
        )

    return cleaned_count


def _cleanup_corrupted_model_cache(
    model_name_or_path: str, snapshot_dir: str, reason: str
) -> None:
    """
    Remove entire corrupted model cache directory to force a clean re-download.

    This is used when we cannot selectively clean (e.g., missing shards, incomplete
    downloads with unknown affected files).

    Args:
        model_name_or_path: Model identifier
        snapshot_dir: Path to the snapshot directory
        reason: Reason for cleanup
    """
    
    repo_folder = os.path.abspath(os.path.join(snapshot_dir, "..", ".."))

    try:
        logger.warning(
            "Removing entire cache for %s at %s. Reason: %s",
            model_name_or_path,
            repo_folder,
            reason,
        )
        shutil.rmtree(repo_folder)
        logger.info("Successfully removed corrupted cache directory")
    except Exception as e:
        logger.error(
            "Failed to remove corrupted cache directory %s: %s. "
            "Manual cleanup may be required.",
            repo_folder,
            e,
        )


def ci_validate_and_cleanup_local_snapshot(
    model_name_or_path: str,
    found_local_snapshot_dir: str,
    local_weight_files: List[str],
) -> bool:
    """
    CI-specific validation and cleanup for local model snapshots.

    This function validates the local snapshot and performs automatic cleanup
    if corruption or missing files are detected. This behavior is only appropriate
    for CI environments where we want automatic recovery.

    Args:
        model_name_or_path: Model identifier for logging
        found_local_snapshot_dir: Path to the local snapshot directory
        local_weight_files: List of weight file paths found in the snapshot

    Returns:
        True if the snapshot is valid and can be used, False if it was invalid
        and cleanup was performed (caller should re-download)
    """
    
    repo_folder = os.path.abspath(os.path.join(found_local_snapshot_dir, "..", ".."))
    blobs_dir = os.path.join(repo_folder, "blobs")

    
    incomplete_files = []
    if os.path.isdir(blobs_dir):
        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))

    if incomplete_files:
        log_info_on_rank0(
            logger,
            f"Found {len(incomplete_files)} .incomplete files in {blobs_dir} for "
            f"{model_name_or_path}. Will clean up and re-download.",
        )
        _cleanup_corrupted_model_cache(
            model_name_or_path,
            found_local_snapshot_dir,
            f"Incomplete download detected ({len(incomplete_files)} incomplete files)",
        )
        return False

    
    if local_weight_files:
        is_valid, error_msg, corrupted_files = _validate_sharded_model(
            found_local_snapshot_dir, local_weight_files
        )
        if not is_valid:
            if corrupted_files:
                
                log_info_on_rank0(
                    logger,
                    f"Found {len(corrupted_files)} corrupted file(s) for "
                    f"{model_name_or_path}: {error_msg}. "
                    "Will selectively clean and re-download only these files.",
                )
                _cleanup_corrupted_files_selective(model_name_or_path, corrupted_files)
                return False
            else:
                
                
                
                log_info_on_rank0(
                    logger,
                    f"Validation failed for {model_name_or_path}: {error_msg}. "
                    "Will attempt to download missing files.",
                )
                return False

        
        for f in local_weight_files:
            base_name = os.path.basename(f)
            
            
            if base_name in [
                "model.safetensors",
                "pytorch_model.safetensors",
                "adapter_model.safetensors",
            ]:
                if not _validate_safetensors_file(f):
                    log_info_on_rank0(
                        logger,
                        f"Corrupted model file {base_name} for {model_name_or_path}. "
                        "Will selectively clean and re-download this file.",
                    )
                    
                    _cleanup_corrupted_files_selective(model_name_or_path, [f])
                    return False
            
            elif base_name in [
                "pytorch_model.bin",
                "model.bin",
                "adapter_model.bin",
            ]:
                if not _validate_pytorch_bin_file(f):
                    log_info_on_rank0(
                        logger,
                        f"Corrupted model file {base_name} for {model_name_or_path}. "
                        "Will selectively clean and re-download this file.",
                    )
                    
                    _cleanup_corrupted_files_selective(model_name_or_path, [f])
                    return False

    return True


def _validate_weights_after_download(
    hf_folder: str,
    allow_patterns: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate downloaded weight files to catch corruption early.

    This function validates safetensors files after download to catch
    corruption issues (truncated downloads, network errors, etc.) before
    model loading fails with cryptic errors. If corruption is found,
    the corrupted files are automatically cleaned up.

    Args:
        hf_folder: Path to the downloaded model folder
        allow_patterns: Patterns used to match weight files
        model_name_or_path: Model identifier for error messages

    Returns:
        True if all files are valid, False if corrupted files were found and cleaned up
    """
    
    weight_files: List[str] = []
    for pattern in allow_patterns:
        weight_files.extend(glob_module.glob(os.path.join(hf_folder, pattern)))

    if not weight_files:
        return True  

    
    corrupted_files = []
    for f in weight_files:
        if f.endswith(".safetensors") and os.path.exists(f):
            if not _validate_safetensors_file(f):
                corrupted_files.append(os.path.basename(f))
        elif f.endswith(".bin") and os.path.exists(f):
            if not _validate_pytorch_bin_file(f):
                corrupted_files.append(os.path.basename(f))

    if corrupted_files:
        
        _cleanup_corrupted_files_selective(
            model_name_or_path,
            [os.path.join(hf_folder, f) for f in corrupted_files],
        )
        log_info_on_rank0(
            logger,
            f"Downloaded model files are corrupted for {model_name_or_path}: "
            f"{corrupted_files}. The corrupted files have been removed. "
            "Will retry download.",
        )
        return False

    return True


def _get_lock_file_path(
    model_name_or_path: str, cache_dir: Optional[str] = None
) -> str:
    """
    Generate a unique lock file path for download coordination.

    In CI environments where multiple containers share an NFS-mounted HF cache,
    the lock file is placed on the shared cache directory so ALL containers
    coordinate on the same lock. This prevents cross-container .incomplete
    file race conditions.

    Falls back to /dev/shm (container-local) for non-CI or when the cache
    dir is not accessible.

    Args:
        model_name_or_path: Model identifier
        cache_dir: HF cache directory (None to use default)

    Returns:
        Path to the lock file
    """
    key_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest()[:16]

    
    
    
    try:
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE
        if os.path.isdir(effective_cache_dir):
            lock_dir = os.path.join(effective_cache_dir, ".sglang_locks")
            os.makedirs(lock_dir, exist_ok=True)
            return os.path.join(lock_dir, f"download_{key_hash}.lock")
    except Exception:
        pass

    
    if os.path.isdir("/dev/shm"):
        return f"/dev/shm/sglang_download_lock_{key_hash}"
    return f"/tmp/sglang_download_lock_{key_hash}"


def _cleanup_incomplete_blobs(model_name_or_path: str, cache_dir: Optional[str]) -> int:
    """
    Remove stale .incomplete files from the model's blobs directory.

    This is lighter than _cleanup_corrupted_model_cache (which deletes the
    entire cache). We only remove .incomplete files so snapshot_download
    starts fresh on retry, preserving any successfully downloaded blobs.

    Args:
        model_name_or_path: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        cache_dir: HF cache directory (None to use default)

    Returns:
        Number of .incomplete files removed
    """
    try:
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE
        repo_folder_name = huggingface_hub.constants.REPO_ID_SEPARATOR.join(
            ["models", *model_name_or_path.split("/")]
        )
        blobs_dir = os.path.join(effective_cache_dir, repo_folder_name, "blobs")

        if not os.path.isdir(blobs_dir):
            return 0

        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))
        removed = 0
        for f in incomplete_files:
            try:
                os.remove(f)
                removed += 1
                logger.debug("Removed incomplete blob: %s", os.path.basename(f))
            except OSError as e:
                logger.debug(
                    "Failed to remove incomplete blob %s: %s", os.path.basename(f), e
                )

        if removed > 0:
            logger.warning(
                "Cleaned up %d .incomplete blob(s) for %s in %s",
                removed,
                model_name_or_path,
                blobs_dir,
            )
        return removed

    except Exception as e:
        logger.debug("Failed to clean up incomplete blobs: %s", e)
        return 0


def ci_download_with_validation_and_retry(
    model_name_or_path: str,
    allow_patterns: List[str],
    ignore_patterns,
    cache_dir: Optional[str],
    revision: Optional[str],
    max_retries: int = 3,
) -> str:
    """
    CI-specific download with validation and automatic retry on corruption.

    This function handles the download of model weights in CI environments,
    with automatic validation and retry logic for handling corrupted downloads.

    Uses filelock.FileLock on the shared HF cache directory to coordinate
    downloads across all processes AND all containers sharing the same
    NFS-mounted cache. Only one process downloads at a time; others wait
    for the lock then use the cached result.

    Args:
        model_name_or_path: The model name or path
        allow_patterns: The allowed patterns for weight files
        ignore_patterns: The patterns to filter out weight files
        cache_dir: The cache directory to store model weights
        revision: The revision of the model
        max_retries: Maximum number of download retries if corruption is detected

    Returns:
        str: The path to the downloaded model weights

    Raises:
        RuntimeError: If download fails after max_retries attempts
    """
    import filelock
    import huggingface_hub.constants
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    
    
    
    lock_file_path = _get_lock_file_path(model_name_or_path, cache_dir)

    logger.info(
        "[CI Download] Process %d using lock file: %s",
        os.getpid(),
        lock_file_path,
    )

    
    
    
    lock = filelock.FileLock(lock_file_path, timeout=-1, mode=0o666)

    logger.info(
        "[CI Download] Process %d waiting to acquire lock for %s",
        os.getpid(),
        model_name_or_path,
    )

    with lock:
        logger.info(
            "[CI Download] Process %d ACQUIRED lock for %s",
            os.getpid(),
            model_name_or_path,
        )

        
        
        try:
            from sglang.srt.model_loader.weight_utils import (
                _find_local_hf_snapshot_dir_unlocked,
            )

            cached_path = _find_local_hf_snapshot_dir_unlocked(
                model_name_or_path, cache_dir, allow_patterns, revision
            )
            if cached_path is not None:
                logger.info(
                    "[CI Download] Process %d found cached model after "
                    "acquiring lock (downloaded by another container): %s",
                    os.getpid(),
                    cached_path,
                )
                return cached_path
        except Exception as e:
            logger.debug(
                "[CI Download] Re-check for cached model failed (non-fatal): %s", e
            )

        
        
        cleaned = _cleanup_incomplete_blobs(model_name_or_path, cache_dir)
        if cleaned > 0:
            logger.info(
                "[CI Download] Pre-download cleanup: removed %d stale "
                ".incomplete file(s) for %s",
                cleaned,
                model_name_or_path,
            )

        hf_folder = None
        for attempt in range(max_retries):
            try:
                hf_folder = snapshot_download(
                    model_name_or_path,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    cache_dir=cache_dir,
                    tqdm_class=DisabledTqdm,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    
                    
                    
                    
                    max_workers=1,
                )
            except (FileNotFoundError, OSError) as e:
                
                
                
                logger.warning(
                    "[CI Download] Process %d hit download error "
                    "(attempt %d/%d) for %s: %s: %s",
                    os.getpid(),
                    attempt + 1,
                    max_retries,
                    model_name_or_path,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries - 1:
                    
                    
                    backoff = 10 * (2**attempt)
                    logger.info(
                        "[CI Download] Cleaning up .incomplete files and "
                        "retrying in %ds...",
                        backoff,
                    )
                    _cleanup_incomplete_blobs(model_name_or_path, cache_dir)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Download failed for {model_name_or_path} after "
                    f"{max_retries} attempts due to download errors. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e

            
            is_valid = _validate_weights_after_download(
                hf_folder, allow_patterns, model_name_or_path
            )

            if is_valid:
                return hf_folder

            
            if attempt < max_retries - 1:
                log_info_on_rank0(
                    logger,
                    f"Retrying download for {model_name_or_path} "
                    f"(attempt {attempt + 2}/{max_retries})...",
                )
            else:
                raise RuntimeError(
                    f"Downloaded model files are still corrupted for "
                    f"{model_name_or_path} after {max_retries} attempts. "
                    "This may indicate a persistent issue with the model files "
                    "on Hugging Face Hub or network problems."
                )

        
        return hf_folder


def ci_validate_and_clean_hf_cache(model_path: str) -> None:
    """
    Validate and clean corrupted safetensors files in HF cache before loading.

    This function is needed because HFRunner (used in tests) calls transformers'
    from_pretrained() directly, which bypasses SGLang's weight validation.
    Corrupted cached files can cause cryptic errors like "EOF while parsing"
    from safetensors.

    Only runs in CI to avoid overhead for regular users.

    Args:
        model_path: Model identifier (e.g., "meta-llama/Llama-2-7b")
    """
    from sglang.utils import is_in_ci

    if not is_in_ci():
        return

    
    if os.path.isdir(model_path):
        return

    try:
        import huggingface_hub.constants

        
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
        repo_folder = os.path.join(
            cache_dir,
            huggingface_hub.constants.REPO_ID_SEPARATOR.join(
                ["models", *model_path.split("/")]
            ),
        )

        if not os.path.isdir(repo_folder):
            return

        
        snapshots_dir = os.path.join(repo_folder, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return

        
        corrupted_files = []
        for snapshot_hash in os.listdir(snapshots_dir):
            snapshot_dir = os.path.join(snapshots_dir, snapshot_hash)
            if not os.path.isdir(snapshot_dir):
                continue

            
            safetensors_files = glob_module.glob(
                os.path.join(snapshot_dir, "*.safetensors")
            )

            for sf_file in safetensors_files:
                
                if not os.path.exists(sf_file):
                    continue

                if not _validate_safetensors_file(sf_file):
                    corrupted_files.append(sf_file)

            
            bin_files = glob_module.glob(os.path.join(snapshot_dir, "*.bin"))

            for bin_file in bin_files:
                
                if not os.path.exists(bin_file):
                    continue

                if not _validate_pytorch_bin_file(bin_file):
                    corrupted_files.append(bin_file)

        if corrupted_files:
            logger.warning(
                "HFRunner: Found %d corrupted weight file(s) for %s. "
                "Removing to force re-download.",
                len(corrupted_files),
                model_path,
            )
            _cleanup_corrupted_files_selective(model_path, corrupted_files)

    except Exception as e:
        
        logger.debug("HF cache validation failed (non-fatal): %s", e)
