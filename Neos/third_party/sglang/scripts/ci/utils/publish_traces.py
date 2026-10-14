"""
Publish performance traces to GitHub repository
"""

import argparse
import base64
import json
import os
import sys
import time
import warnings
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def is_rate_limit_error(e):
    """Check if an exception is a GitHub rate limit error (not permission error)"""
    if not isinstance(e, HTTPError):
        return False
    if e.code == 429:
        return True
    if e.code == 403:
        
        error_body = getattr(e, "error_body", "")
        if isinstance(error_body, str):
            
            rate_limit_phrases = [
                "rate limit",
                "abuse detection",
                "secondary rate limit",
            ]
            return any(phrase in error_body.lower() for phrase in rate_limit_phrases)
    return False


def is_permission_error(e):
    """Check if an exception is a GitHub permission error"""
    if not isinstance(e, HTTPError) or e.code != 403:
        return False
    error_body = getattr(e, "error_body", "")
    if isinstance(error_body, str):
        permission_phrases = [
            "resource not accessible",
            "must have push access",
            "permission",
            "denied",
        ]
        return any(phrase in error_body.lower() for phrase in permission_phrases)
    return False


def make_github_request(url, token, method="GET", data=None):
    """Make authenticated request to GitHub API"""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if data:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode("utf-8")

    req = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        print(f"GitHub API request failed: {e}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Error response body: {error_body}")
            e.error_body = error_body  
        except Exception:
            e.error_body = ""
        raise
    except Exception as e:
        print(f"GitHub API request failed with a non-HTTP error: {e}")
        raise


def verify_token_permissions(repo_owner, repo_name, token):
    """Verify that the token has necessary permissions for the repository"""
    print("Verifying token permissions...")

    checks = [
        (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}",  
            "Repository access verified",
        ),
        (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents",  
            "Repository contents access verified",
        ),
    ]

    for url, success_message in checks:
        try:
            response = make_github_request(url, token)
            if success_message == "Repository access verified":
                repo_data = json.loads(response)
                print(f"{success_message}: {repo_data['full_name']}")
            else:
                print(success_message)
        except Exception as e:
            if is_rate_limit_error(e):
                warnings.warn(
                    "GitHub API rate limit exceeded during token verification."
                )
                return "rate_limited"
            print(f"Failed to verify permissions for {url}: {e}")
            return False

    return True


def get_branch_sha(repo_owner, repo_name, branch, token):
    """Get SHA of the branch head"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["object"]["sha"]


def get_tree_sha(repo_owner, repo_name, commit_sha, token):
    """Get tree SHA from commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits/{commit_sha}"
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["tree"]["sha"]


def create_blob(repo_owner, repo_name, content, token, max_retries=3):
    """Create a blob with file content"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/blobs"

    
    content_b64 = base64.b64encode(content).decode("utf-8")

    data = {"content": content_b64, "encoding": "base64"}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            return json.loads(response)["sha"]
        except Exception as e:
            
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt  
                print(
                    f"Blob creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def create_blobs(repo_owner, repo_name, files, token):
    """Create blobs for all files and return tree items with blob SHAs"""
    tree_items = []
    for i, (file_path, content) in enumerate(files):
        
        blob_sha = create_blob(repo_owner, repo_name, content, token)
        tree_items.append(
            {
                "path": file_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            }
        )
        
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"Created {i + 1}/{len(files)} blobs...")
    return tree_items


def create_tree(repo_owner, repo_name, base_tree_sha, tree_items, token, max_retries=3):
    """Create a new tree from pre-created blob SHAs"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees"

    data = {"base_tree": base_tree_sha, "tree": tree_items}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            return json.loads(response)["sha"]
        except Exception as e:
            
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Tree creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def create_commit(
    repo_owner, repo_name, tree_sha, parent_sha, message, token, max_retries=3
):
    """Create a new commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits"

    data = {"tree": tree_sha, "parents": [parent_sha], "message": message}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            commit_sha = json.loads(response)["sha"]

            
            verify_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits/{commit_sha}"
            verify_response = make_github_request(verify_url, token)
            verify_data = json.loads(verify_response)
            if verify_data["sha"] != commit_sha:
                raise Exception(
                    f"Commit verification failed: expected {commit_sha}, got {verify_data['sha']}"
                )

            return commit_sha
        except Exception as e:
            
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Commit creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def update_branch_ref(repo_owner, repo_name, branch, commit_sha, token, max_retries=3):
    """Update branch reference to point to new commit"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )

    data = {"sha": commit_sha}

    for attempt in range(max_retries):
        try:
            make_github_request(url, token, method="PATCH", data=data)
            return
        except HTTPError as e:
            
            if is_rate_limit_error(e):
                raise

            
            is_object_not_exist = False
            if hasattr(e, "error_body"):
                try:
                    error_data = json.loads(e.error_body)
                    if "Object does not exist" in error_data.get("message", ""):
                        is_object_not_exist = True
                except Exception:
                    pass

            if is_object_not_exist and attempt < max_retries - 1:
                
                wait_time = 2**attempt
                print(
                    f"Branch update failed with 'Object does not exist' (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s for consistency..."
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Branch update failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def copy_trace_files(source_dir, target_base_path):
    """Copy trace files and return list of files to upload.

    Only uploads traces from TP rank 0 to avoid duplicated data across tensor parallel ranks.
    """
    files_to_upload = []

    if not os.path.exists(source_dir):
        print(f"Warning: Traces directory {source_dir} does not exist")
        return files_to_upload

    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json.gz"):

                
                if "TP-" in file and "TP-0" not in file:
                    continue

                source_file = os.path.join(root, file)
                
                rel_path = os.path.relpath(source_file, source_dir)
                target_path = f"{target_base_path}/{rel_path}"

                
                with open(source_file, "rb") as f:
                    content = f.read()

                files_to_upload.append((target_path, content))

    return files_to_upload


def publish_traces(traces_dir, run_id, run_number):
    """Publish traces from a single directory to GitHub repository in a single commit"""
    target_base_path = f"traces/{run_id}"
    files_to_upload = copy_trace_files(traces_dir, target_base_path)

    if not files_to_upload:
        print("No trace files found to upload")
        return

    print(f"Found {len(files_to_upload)} files to upload")
    publish_traces_from_files(files_to_upload, run_id, run_number)


def publish_traces_from_files(files_to_upload, run_id, run_number):
    """Publish pre-collected trace files to GitHub repository in a single commit"""
    
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    
    repo_owner = "sglang-bot"
    repo_name = "sglang-ci-data"
    branch = "main"

    
    permission_check = verify_token_permissions(repo_owner, repo_name, token)
    if permission_check == "rate_limited":
        warnings.warn(
            "Skipping trace upload due to GitHub API rate limit. "
            "This is expected during high CI activity and does not indicate a test failure."
        )
        return
    elif not permission_check:
        print(
            "Token permission verification failed. Please check the token permissions."
        )
        sys.exit(1)

    max_retries = 5
    retry_delay = 5  

    
    try:
        tree_items = create_blobs(repo_owner, repo_name, files_to_upload, token)
    except Exception as e:
        
        if is_rate_limit_error(e):
            warnings.warn(
                "GitHub API rate limit exceeded during blob creation. Skipping trace upload."
            )
            return
        
        if is_permission_error(e):
            print(
                f"ERROR: Token does not have write permission to {repo_owner}/{repo_name}. "
                "Please update the GH_PAT_FOR_NIGHTLY_CI_DATA secret with a token that has "
                "'contents: write' permission for the repository."
            )
            sys.exit(1)
        print(f"Failed to create blobs: {e}")
        raise

    for attempt in range(max_retries):
        try:
            
            branch_sha = get_branch_sha(repo_owner, repo_name, branch, token)
            print(f"Current branch head: {branch_sha}")

            
            tree_sha = get_tree_sha(repo_owner, repo_name, branch_sha, token)
            print(f"Current tree SHA: {tree_sha}")

            
            new_tree_sha = create_tree(
                repo_owner, repo_name, tree_sha, tree_items, token
            )
            print(f"Created new tree: {new_tree_sha}")

            
            commit_message = f"Nightly traces for run {run_id} at {run_number} ({len(files_to_upload)} files)"
            commit_sha = create_commit(
                repo_owner,
                repo_name,
                new_tree_sha,
                branch_sha,
                commit_message,
                token,
            )
            print(f"Created commit: {commit_sha}")

            
            update_branch_ref(repo_owner, repo_name, branch, commit_sha, token)
            print("Updated branch reference")

            print("Successfully published all traces in a single commit")
            return

        except Exception as e:
            
            is_retryable = False
            error_type = "unknown"

            if hasattr(e, "error_body"):
                if "Update is not a fast forward" in e.error_body:
                    is_retryable = True
                    error_type = "fast-forward conflict"
                elif "Object does not exist" in e.error_body:
                    is_retryable = True
                    error_type = "object consistency"

            
            if isinstance(e, HTTPError) and e.code in [422, 500, 502, 503, 504]:
                is_retryable = True
                error_type = f"HTTP {e.code}"

            
            if is_rate_limit_error(e):
                warnings.warn("GitHub API rate limit exceeded. Skipping trace upload.")
                return

            
            if is_permission_error(e):
                print(
                    f"ERROR: Token does not have write permission to {repo_owner}/{repo_name}. "
                    "Please update the GH_PAT_FOR_NIGHTLY_CI_DATA secret with a token that has "
                    "'contents: write' permission for the repository."
                )
                sys.exit(1)

            if is_retryable and attempt < max_retries - 1:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed ({error_type}). Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Failed to publish traces after {attempt + 1} attempts: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Publish performance traces to GitHub repository"
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        action="append",
        dest="traces_dirs",
        required=True,
        help="Traces directory to publish (can be specified multiple times)",
    )
    args = parser.parse_args()

    
    run_id = os.getenv("GITHUB_RUN_ID", "test")
    run_number = os.getenv("GITHUB_RUN_NUMBER", "12345")

    if not run_id or not run_number:
        print(
            "Error: GITHUB_RUN_ID and GITHUB_RUN_NUMBER environment variables must be set"
        )
        sys.exit(1)

    
    target_base_path = f"traces/{run_id}"
    all_files = []
    for traces_dir in args.traces_dirs:
        print(f"Processing traces from directory: {traces_dir}")
        files = copy_trace_files(traces_dir, target_base_path)
        all_files.extend(files)

    if not all_files:
        print("No trace files found to upload across all directories")
        return

    print(f"Found {len(all_files)} total files to upload")

    
    publish_traces_from_files(all_files, run_id, run_number)


if __name__ == "__main__":
    main()
