"""
SGLang CI Consecutive Failures Analyzer

Monitors GitHub Actions workflows for consecutive test failures and runner issues.
Detects failure streaks, tracks job health, identifies problematic runners, and generates alerts.

Features:
- Analyzes all jobs in PR Test workflow (excluding administrative jobs)
- Tracks consecutive failure streaks for each job
- Monitors runner health and failure rates
- Identifies whether failures are code-related or infrastructure-related
- Generates detailed reports with actionable recommendations

Usage:
    python ci_failures_analysis.py --token <GITHUB_TOKEN> --limit 100
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests


class SGLangFailuresAnalyzer:
    """Analyzes consecutive failures in GitHub Actions workflows."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Failures-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        
        self.excluded_jobs = [
            "check-changes",
            "pr-test-finish",  
            "pr-test-amd-finish",  
            "call-gate",
            "pr-gate",
            "check-all-jobs",
        ]
        self.test_summaries = {}

    def get_recent_runs(
        self,
        limit: int = 500,
        workflow_filter: List[str] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Fetch recent workflow runs from GitHub API using workflow file names.

        Args:
            limit: Number of runs to fetch per workflow
            workflow_filter: List of workflow filenames
            filters: Optional dict of API filters (e.g., {"event": "schedule"}, {"branch": "main"})
        """
        filter_desc = f"workflows: {', '.join(workflow_filter)}"
        if filters:
            filter_desc += f", filters: {filters}"

        print(f"Fetching {limit} runs per workflow ({filter_desc})...")

        all_runs = []

        for workflow_file in workflow_filter:
            print(f"Fetching runs for {workflow_file}...")

            
            url = f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_file}/runs"
            params = {"per_page": min(limit, 100), "status": "completed"}

            
            if filters:
                params.update(filters)

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                runs = data.get("workflow_runs", [])
                print(f"  Found {len(runs)} runs for {workflow_file}")
                all_runs.extend(runs[:limit])

            except requests.exceptions.RequestException as e:
                print(f"Error fetching runs for {workflow_file}: {e}")
                continue

        print(f"Collected {len(all_runs)} total runs")
        return all_runs

    def get_jobs_for_run(self, run_id: int) -> List[Dict]:
        """Get all jobs for a specific workflow run, handling pagination."""
        try:
            all_jobs = []
            url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            params = {"per_page": 100}  

            while url:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                jobs = data.get("jobs", [])
                all_jobs.extend(jobs)

                
                link_header = response.headers.get("Link", "")
                next_url = None
                if link_header:
                    links = link_header.split(", ")
                    for link in links:
                        if 'rel="next"' in link:
                            try:
                                parts = link.split(";")
                                if parts:
                                    next_url = parts[0].strip("<>")
                            except Exception as e:
                                print(f"Error parsing Link header: {link}, error: {e}")
                                next_url = None
                            break
                url = next_url
                params = {}  

            return all_jobs
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs for run {run_id}: {e}")
            return []

    def get_job_logs(self, job_id: int) -> str:
        """Fetch logs for a specific job."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{job_id}/logs"
            response = self.session.get(url, timeout=60, allow_redirects=True)
            if response.status_code == 200:
                return response.text
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error fetching logs for job {job_id}: {e}")
            return ""

    def get_online_runners(self) -> Dict[str, Dict]:
        """
        Fetch all self-hosted runners and their online status from GitHub API.

        Returns:
            Dict mapping runner label sets to their online/total counts.
            E.g., {"8-gpu-h200-runner": {"online": 2, "total": 3, "busy": 1}}
        """
        print("Fetching self-hosted runner status...")
        try:
            
            runner_token = os.environ.get("GH_PAT_FOR_RUNNER_ADMIN") or self.token
            runner_headers = {
                "Authorization": f"token {runner_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            all_runners = []
            url = f"{self.base_url}/repos/{self.repo}/actions/runners"
            params = {"per_page": 100}

            while url:
                response = requests.get(
                    url, headers=runner_headers, params=params, timeout=30
                )
                if response.status_code != 200:
                    print(
                        f"  Warning: Runner API returned {response.status_code}: {response.text[:200]}"
                    )
                    return {}
                data = response.json()
                runners = data.get("runners", [])
                all_runners.extend(runners)

                
                link_header = response.headers.get("Link", "")
                next_url = None
                if link_header:
                    links = link_header.split(", ")
                    for link in links:
                        if 'rel="next"' in link:
                            try:
                                parts = link.split(";")
                                if parts:
                                    next_url = parts[0].strip("<>")
                            except Exception as e:
                                print(f"Error parsing Link header: {link}, error: {e}")
                                next_url = None
                            break
                url = next_url
                params = {}  

            print(f"  Found {len(all_runners)} self-hosted runners")

            
            
            runner_stats_by_label = defaultdict(
                lambda: {"online": 0, "total": 0, "busy": 0}
            )

            
            excluded_labels = {"self-hosted", "Linux", "X64", "ARM64"}

            for runner in all_runners:
                
                labels = [
                    label.get("name", "")
                    for label in runner.get("labels", [])
                    if label.get("name", "") not in excluded_labels
                ]

                
                for runner_label in labels:
                    runner_stats_by_label[runner_label]["total"] += 1
                    if runner.get("status") == "online":
                        runner_stats_by_label[runner_label]["online"] += 1
                    if runner.get("busy", False):
                        runner_stats_by_label[runner_label]["busy"] += 1

            return dict(runner_stats_by_label)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching runners: {e}")
            return {}

    def find_last_running_test(self, logs: str) -> Optional[Dict]:
        """
        Find the last test that was running before logs cut off (for timeout/exit scenarios).
        Finds the last instance of 'server_args:' and looks for the test file a few lines above it.

        Returns:
            Dict with test info if found, or None if no test found.
        """
        import re

        
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        logs = ansi_escape.sub("", logs)

        lines = logs.split("\n")

        
        
        
        
        
        test_patterns = [
            r"(\S+\.py)::",  
            r"python3?\s+(\S+\.py)",  
        ]

        
        server_args_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if "server_args:" in lines[i].lower() or "server_args =" in lines[i]:
                server_args_idx = i
                break

        if server_args_idx is not None:
            
            for j in range(1, 11):
                line_idx = server_args_idx - j
                if line_idx >= 0:
                    line = lines[line_idx]
                    for pattern in test_patterns:
                        match = re.search(pattern, line)
                        if match:
                            full_path = match.group(1)
                            test_file = (
                                full_path.split("/")[-1]
                                if "/" in full_path
                                else full_path
                            )
                            if test_file.endswith(".py"):
                                return {
                                    "test_file": test_file,
                                    "full_path": full_path,
                                    "context": "last_running",
                                }

        return None

    def parse_test_summary(self, logs: str) -> Optional[Dict]:
        """
        Parse the test summary block from job logs.

        Returns:
            Dict with passed/total counts and list of failed tests, or None if no summary found.
            If no summary found, attempts to find the last running test (for timeout scenarios).
        """
        import re

        
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        logs = ansi_escape.sub("", logs)

        
        
        summary_match = re.search(r"Test Summary:\s*(\d+)/(\d+)\s*passed", logs)
        if not summary_match:
            
            last_test = self.find_last_running_test(logs)
            if last_test:
                return {
                    "passed": 0,
                    "total": 0,
                    "failed_tests": [last_test],
                    "incomplete": True,  
                }
            return None

        try:
            passed = int(summary_match.group(1))
            total = int(summary_match.group(2))
        except (ValueError, TypeError) as e:
            print(f"Error parsing test summary numbers: {e}")
            return None

        
        
        failed_tests = []
        
        failed_section_match = re.search(
            r".?\s*FAILED:\s*\n(.*?)(?:={10,}|$)", logs, re.DOTALL
        )

        if failed_section_match:
            failed_section = failed_section_match.group(1)
            
            for match in re.finditer(r"(\S+\.py)", failed_section):
                full_path = match.group(1)
                
                test_file = full_path.split("/")[-1] if "/" in full_path else full_path
                failed_tests.append(
                    {
                        "test_file": test_file,
                        "full_path": full_path,
                    }
                )

        return {
            "passed": passed,
            "total": total,
            "failed_tests": failed_tests,
        }

    def analyze_test_failures_for_job(self, recent_runs: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze test-level failures for a specific job across its recent runs.

        Args:
            recent_runs: List of recent run info dicts with job_id, job_url, conclusion, etc.
            debug: Enable debug logging

        Returns:
            Dict mapping test_file -> {
                "total_failures": int,
                "current_streak": int,
                "recent_runs": [{"run_number": ..., "job_url": ..., "status": ..., "failed": bool}, ...]
            }
        """
        test_failures: Dict[str, Dict] = defaultdict(
            lambda: {"total_failures": 0, "current_streak": 0, "recent_runs": []}
        )

        
        parsed_any_test_summary = False

        
        for run_info in recent_runs:
            job_id = run_info.get("job_id")
            conclusion = run_info.get("conclusion")

            
            if conclusion == "failure" and job_id:
                logs = self.get_job_logs(job_id)
                test_summary = self.parse_test_summary(logs) if logs else None
                self.test_summaries[job_id] = test_summary

                
                if not test_summary:
                    job_name = run_info.get("job_name", "unknown")
                    run_number = run_info.get("run_number", "unknown")
                    job_url = run_info.get("job_url", "N/A")
                    log_size = len(logs) if logs else 0
                    print(
                        f"  ⚠️  Job failed without test summary: {job_name} (Run #{run_number})"
                    )
                    print(f"      URL: {job_url}")
                    print(
                        f"      Log size: {log_size} chars, Logs available: {bool(logs)}"
                    )
                    if logs:
                        
                        log_snippet = logs[-500:] if len(logs) > 500 else logs
                        print(f"      Last 500 chars of logs: {log_snippet[:200]}...")
                elif test_summary.get("incomplete"):
                    
                    job_name = run_info.get("job_name", "unknown")
                    run_number = run_info.get("run_number", "unknown")
                    inferred_tests = [
                        t["test_file"] for t in test_summary.get("failed_tests", [])
                    ]
                    print(
                        f"  ⏱️  Inferred timeout test for {job_name} (Run #{run_number}): {inferred_tests}"
                    )

                if test_summary and test_summary["failed_tests"]:
                    parsed_any_test_summary = True
                    
                    failed_test_files = set()
                    is_incomplete = test_summary.get("incomplete", False)

                    for failed_test in test_summary["failed_tests"]:
                        test_file = failed_test["test_file"]
                        failed_test_files.add(test_file)
                        test_failures[test_file]["total_failures"] += 1
                        test_failures[test_file]["current_streak"] += 1

                        
                        is_last_running = failed_test.get("context") == "last_running"
                        status = "⏱️" if is_last_running else "❌"

                        test_failures[test_file]["recent_runs"].append(
                            {
                                "run_number": run_info.get("run_number"),
                                "job_url": run_info.get("job_url"),
                                "status": status,
                                "failed": True,
                                "last_running": is_last_running,
                            }
                        )

                        
                        if (
                            is_last_running
                            and "has_timeout" not in test_failures[test_file]
                        ):
                            test_failures[test_file]["has_timeout"] = True

                    
                    
                    for test_file in test_failures.keys():
                        if test_file not in failed_test_files:
                            
                            test_failures[test_file]["current_streak"] = 0
                            test_failures[test_file]["recent_runs"].append(
                                {
                                    "run_number": run_info.get("run_number"),
                                    "job_url": run_info.get("job_url"),
                                    "status": "✅",
                                    "failed": False,
                                }
                            )
                else:
                    
                    for test_file in test_failures.keys():
                        test_failures[test_file]["recent_runs"].append(
                            {
                                "run_number": run_info.get("run_number"),
                                "job_url": run_info.get("job_url"),
                                "status": "⚪",  
                                "failed": None,
                            }
                        )
            elif conclusion == "success":
                
                for test_file in test_failures.keys():
                    test_failures[test_file]["current_streak"] = 0
                    test_failures[test_file]["recent_runs"].append(
                        {
                            "run_number": run_info.get("run_number"),
                            "job_url": run_info.get("job_url"),
                            "status": "✅",
                            "failed": False,
                        }
                    )
            else:
                
                for test_file in test_failures.keys():
                    test_failures[test_file]["recent_runs"].append(
                        {
                            "run_number": run_info.get("run_number"),
                            "job_url": run_info.get("job_url"),
                            "status": "⚪",
                            "failed": None,
                        }
                    )

            time.sleep(0.1)  

        
        if not parsed_any_test_summary:
            return {"_no_test_summary": True}

        
        result = {}
        for test_file, data in test_failures.items():
            
            
            
            
            current_streak = data["current_streak"]
            recent_runs = data["recent_runs"]

            
            if current_streak > 0:
                
                
                streak_runs = recent_runs[-current_streak:]
                has_actual_failure = any(
                    run.get("failed") == True for run in streak_runs
                )

                
                if not has_actual_failure:
                    continue

            result[test_file] = {
                "total_failures": data["total_failures"],
                "current_streak": current_streak,
                "recent_runs": recent_runs[-10:],  
            }

        return result

    def analyze_runner_health(
        self, runs: List[Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """
        Analyze runner health by tracking failures per runner and consecutive failure streaks.

        Returns:
            Tuple of (runner_stats, runner_instance_data, runner_streak_data, runner_instance_streak_data)
            - runner_stats: Overall stats per runner (failure rate, total jobs, etc.)
            - runner_instance_data: Per-instance breakdown of failures
            - runner_streak_data: Consecutive failure streaks per runner label
            - runner_instance_streak_data: Consecutive failure streaks per runner instance
        """
        print("\nAnalyzing runner health and consecutive failures...")

        
        sorted_runs = sorted(runs, key=lambda x: x.get("created_at", ""))

        
        runner_total_jobs: Dict[str, int] = defaultdict(int)
        runner_failed_jobs: Dict[str, int] = defaultdict(int)
        runner_job_failures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        runner_job_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        
        runner_instance_queue_times: Dict[str, List[float]] = defaultdict(list)

        
        runner_instance_stats: Dict[str, Dict] = defaultdict(
            lambda: {"total_jobs": 0, "failed_jobs": 0, "jobs_failed": defaultdict(int)}
        )

        
        runner_current_streak: Dict[str, int] = defaultdict(int)
        runner_max_streak: Dict[str, int] = defaultdict(int)
        runner_first_failure_in_streak: Dict[str, Optional[Dict]] = {}
        runner_last_failure_in_streak: Dict[str, Optional[Dict]] = {}
        runner_recovery_info: Dict[str, Optional[Dict]] = {}

        
        runner_instance_current_streak: Dict[str, int] = defaultdict(int)
        runner_instance_max_streak: Dict[str, int] = defaultdict(int)
        runner_instance_first_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_last_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_recovery: Dict[str, Optional[Dict]] = {}

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed} for runner analysis: #{run.get('run_number')}"
                )

            head_commit = run.get("head_commit") or {}
            run_info = {
                "run_number": run.get("run_number"),
                "run_id": run.get("id"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": head_commit.get("author", {}).get("name", "Unknown"),
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            
            jobs = self.get_jobs_for_run(run.get("id"))

            
            runner_had_failure: Dict[str, bool] = defaultdict(bool)
            runner_had_success: Dict[str, bool] = defaultdict(bool)
            runner_instance_had_failure: Dict[str, bool] = defaultdict(bool)
            runner_instance_had_success: Dict[str, bool] = defaultdict(bool)
            
            runner_first_failed_job: Dict[str, Dict] = {}
            runner_instance_first_failed_job: Dict[str, Dict] = {}

            for job in jobs:
                job_name = job.get("name", "")

                
                if any(
                    job_name.startswith(excluded) for excluded in self.excluded_jobs
                ):
                    continue

                
                
                runner_name = (
                    job.get("runner_name")
                    or job.get("runner", {}).get("name")
                    or "unknown"
                )
                runner_id = job.get("runner_id") or job.get("runner", {}).get("id")

                
                runner_labels = job.get("labels", [])
                runner_labels_str = (
                    ", ".join(runner_labels) if runner_labels else "unknown"
                )

                
                if not runner_labels_str or runner_labels_str == "unknown":
                    continue

                
                
                runner_key = runner_labels_str
                runner_total_jobs[runner_key] += 1
                runner_job_totals[runner_key][job_name] += 1

                
                if runner_id:
                    runner_instance_key = f"{runner_labels_str}_{runner_id}"
                    runner_instance_stats[runner_instance_key]["total_jobs"] += 1
                    
                    runner_instance_stats[runner_instance_key][
                        "runner_name"
                    ] = runner_name

                    
                    created_at = job.get("created_at")
                    started_at = job.get("started_at")
                    if created_at and started_at:
                        try:
                            from datetime import datetime

                            created_time = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                            started_time = datetime.fromisoformat(
                                started_at.replace("Z", "+00:00")
                            )
                            queue_time_seconds = (
                                started_time - created_time
                            ).total_seconds()
                            if queue_time_seconds >= 0:  
                                runner_instance_queue_times[runner_instance_key].append(
                                    queue_time_seconds
                                )
                        except (ValueError, AttributeError, TypeError) as e:
                            print(
                                f"Error parsing timestamps for job {job.get('id')}: {e}"
                            )
                            pass  

                conclusion = job.get("conclusion")

                if conclusion == "failure":
                    
                    runner_failed_jobs[runner_key] += 1
                    runner_job_failures[runner_key][job_name] += 1
                    runner_had_failure[runner_key] = True

                    
                    if runner_key not in runner_first_failed_job:
                        runner_first_failed_job[runner_key] = {
                            "job_id": job.get("id"),
                            "job_url": job.get("html_url", run_info["url"]),
                            "job_name": job_name,
                        }

                    if runner_id:
                        runner_instance_stats[runner_instance_key]["failed_jobs"] += 1
                        runner_instance_stats[runner_instance_key]["jobs_failed"][
                            job_name
                        ] += 1
                        runner_instance_had_failure[runner_instance_key] = True

                        
                        if runner_instance_key not in runner_instance_first_failed_job:
                            runner_instance_first_failed_job[runner_instance_key] = {
                                "job_id": job.get("id"),
                                "job_url": job.get("html_url", run_info["url"]),
                                "job_name": job_name,
                            }

                elif conclusion == "success":
                    runner_had_success[runner_key] = True
                    if runner_id:
                        runner_instance_had_success[runner_instance_key] = True

            
            
            for runner_key in set(
                list(runner_had_failure.keys()) + list(runner_had_success.keys())
            ):
                if runner_had_failure[runner_key]:
                    runner_current_streak[runner_key] += 1
                    failure_info = {
                        **run_info,
                        "runner_key": runner_key,
                    }

                    
                    if runner_key in runner_first_failed_job:
                        failure_info.update(runner_first_failed_job[runner_key])

                    
                    if runner_current_streak[runner_key] == 1:
                        runner_first_failure_in_streak[runner_key] = failure_info
                    
                    runner_last_failure_in_streak[runner_key] = failure_info

                    
                    if (
                        runner_current_streak[runner_key]
                        > runner_max_streak[runner_key]
                    ):
                        runner_max_streak[runner_key] = runner_current_streak[
                            runner_key
                        ]

                elif runner_had_success[runner_key]:
                    
                    if runner_current_streak[runner_key] > 0:
                        runner_recovery_info[runner_key] = {
                            **run_info,
                            "runner_key": runner_key,
                            "streak_length": runner_current_streak[runner_key],
                        }

                    runner_current_streak[runner_key] = 0
                    runner_first_failure_in_streak[runner_key] = None
                    runner_last_failure_in_streak[runner_key] = None

            
            for runner_instance_key in set(
                list(runner_instance_had_failure.keys())
                + list(runner_instance_had_success.keys())
            ):
                if runner_instance_had_failure[runner_instance_key]:
                    runner_instance_current_streak[runner_instance_key] += 1

                    if runner_instance_current_streak[runner_instance_key] == 1:
                        failure_info = {
                            **run_info,
                            "runner_instance": runner_instance_key,
                        }
                        
                        if runner_instance_key in runner_instance_first_failed_job:
                            failure_info.update(
                                runner_instance_first_failed_job[runner_instance_key]
                            )
                        runner_instance_first_failure[runner_instance_key] = (
                            failure_info
                        )

                    
                    failure_info = {
                        **run_info,
                        "runner_instance": runner_instance_key,
                    }
                    
                    if runner_instance_key in runner_instance_first_failed_job:
                        failure_info.update(
                            runner_instance_first_failed_job[runner_instance_key]
                        )
                    runner_instance_last_failure[runner_instance_key] = failure_info

                    if (
                        runner_instance_current_streak[runner_instance_key]
                        > runner_instance_max_streak[runner_instance_key]
                    ):
                        runner_instance_max_streak[runner_instance_key] = (
                            runner_instance_current_streak[runner_instance_key]
                        )

                elif runner_instance_had_success[runner_instance_key]:
                    if runner_instance_current_streak[runner_instance_key] > 0:
                        runner_instance_recovery[runner_instance_key] = {
                            **run_info,
                            "runner_instance": runner_instance_key,
                            "streak_length": runner_instance_current_streak[
                                runner_instance_key
                            ],
                        }

                    runner_instance_current_streak[runner_instance_key] = 0
                    runner_instance_first_failure[runner_instance_key] = None
                    runner_instance_last_failure[runner_instance_key] = None

            time.sleep(0.05)

        
        runner_stats = {}
        for runner_key in runner_total_jobs.keys():
            total = runner_total_jobs[runner_key]
            failed = runner_failed_jobs[runner_key]
            failure_rate = (failed / total * 100) if total > 0 else 0

            
            
            aggregated_queue_times = []
            for instance_key, queue_times in runner_instance_queue_times.items():
                
                instance_labels = (
                    instance_key.rsplit("_", 1)[0]
                    if "_" in instance_key
                    else instance_key
                )
                if instance_labels == runner_key:
                    aggregated_queue_times.extend(queue_times)

            avg_queue_time = (
                sum(aggregated_queue_times) / len(aggregated_queue_times)
                if aggregated_queue_times
                else 0
            )
            p90_queue_time = 0
            if aggregated_queue_times:
                sorted_queue_times = sorted(aggregated_queue_times)
                p90_index = int(len(sorted_queue_times) * 0.9)
                p90_queue_time = (
                    sorted_queue_times[p90_index]
                    if p90_index < len(sorted_queue_times)
                    else sorted_queue_times[-1]
                )

            runner_stats[runner_key] = {
                "total_jobs": total,
                "failed_jobs": failed,
                "failure_rate": failure_rate,
                "unique_jobs_with_failures": len(runner_job_failures[runner_key]),
                "jobs_failed": dict(runner_job_failures[runner_key]),
                "jobs_total": dict(runner_job_totals[runner_key]),
                "avg_queue_time_seconds": avg_queue_time,
                "p90_queue_time_seconds": p90_queue_time,
                "queue_time_samples": len(aggregated_queue_times),
            }

        
        runner_instance_data = {}
        for instance_key, stats in runner_instance_stats.items():
            
            queue_times = runner_instance_queue_times[instance_key]
            avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
            p90_queue_time = 0
            if queue_times:
                sorted_queue_times = sorted(queue_times)
                p90_index = int(len(sorted_queue_times) * 0.9)
                p90_queue_time = (
                    sorted_queue_times[p90_index]
                    if p90_index < len(sorted_queue_times)
                    else sorted_queue_times[-1]
                )

            runner_instance_data[instance_key] = {
                "total_jobs": stats["total_jobs"],
                "failed_jobs": stats["failed_jobs"],
                "failure_rate": (
                    stats["failed_jobs"] / stats["total_jobs"] * 100
                    if stats["total_jobs"] > 0
                    else 0
                ),
                "jobs_failed": dict(stats["jobs_failed"]),
                "runner_name": stats.get("runner_name", "unknown"),
                "avg_queue_time_seconds": avg_queue_time,
                "p90_queue_time_seconds": p90_queue_time,
                "queue_time_samples": len(queue_times),
            }

        
        runner_streak_data = {}
        for runner_key in runner_total_jobs.keys():
            runner_streak_data[runner_key] = {
                "current_streak": runner_current_streak[runner_key],
                "max_streak": runner_max_streak[runner_key],
                "total_failures": runner_failed_jobs[runner_key],
                "total_jobs": runner_total_jobs[runner_key],
                "failure_rate": (
                    runner_failed_jobs[runner_key] / runner_total_jobs[runner_key] * 100
                    if runner_total_jobs[runner_key] > 0
                    else 0
                ),
                "jobs_failed": dict(runner_job_failures[runner_key]),
                "first_failure_in_streak": runner_first_failure_in_streak.get(
                    runner_key
                ),
                "last_failure_in_streak": runner_last_failure_in_streak.get(runner_key),
                "recovery_info": runner_recovery_info.get(runner_key),
            }

        
        runner_instance_streak_data = {}
        for instance_key in runner_instance_stats.keys():
            runner_instance_streak_data[instance_key] = {
                "current_streak": runner_instance_current_streak[instance_key],
                "max_streak": runner_instance_max_streak[instance_key],
                "total_failures": runner_instance_stats[instance_key]["failed_jobs"],
                "total_jobs": runner_instance_stats[instance_key]["total_jobs"],
                "failure_rate": (
                    runner_instance_stats[instance_key]["failed_jobs"]
                    / runner_instance_stats[instance_key]["total_jobs"]
                    * 100
                    if runner_instance_stats[instance_key]["total_jobs"] > 0
                    else 0
                ),
                "runner_name": runner_instance_stats[instance_key].get(
                    "runner_name", "unknown"
                ),
                "jobs_failed": dict(runner_instance_stats[instance_key]["jobs_failed"]),
                "first_failure_in_streak": runner_instance_first_failure.get(
                    instance_key
                ),
                "last_failure_in_streak": runner_instance_last_failure.get(
                    instance_key
                ),
                "recovery_info": runner_instance_recovery.get(instance_key),
            }

        return (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        )

    def analyze_consecutive_failures(
        self, runs: List[Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Analyze consecutive failures for each job.

        "Current Streak" = consecutive failures ending at the most recent run (NOW)
        If the most recent run succeeded, current streak = 0 (streak is broken)
        "Max Streak" = the longest consecutive failure streak seen in the analyzed period

        Returns:
            Tuple of (job_streak_data, job_current_streaks)
        """
        print("\nAnalyzing consecutive failures...")

        
        sorted_runs = sorted(runs, key=lambda x: x.get("created_at", ""))

        
        job_current_streak: Dict[str, int] = defaultdict(int)
        job_max_streak: Dict[str, int] = defaultdict(int)
        job_total_failures: Dict[str, int] = defaultdict(int)
        job_total_runs: Dict[str, int] = defaultdict(int)
        job_first_failure_in_streak: Dict[str, Optional[Dict]] = {}
        job_last_failure_in_streak: Dict[str, Optional[Dict]] = {}
        job_recovery_info: Dict[str, Optional[Dict]] = {}
        job_recent_runs: Dict[str, List[Dict]] = defaultdict(list)  

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed}: #{run.get('run_number')}"
                )

            head_commit = run.get("head_commit") or {}
            run_info = {
                "run_number": run.get("run_number"),
                "run_id": run.get("id"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": head_commit.get("author", {}).get("name", "Unknown"),
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            
            jobs = self.get_jobs_for_run(run.get("id"))

            for job in jobs:
                job_name = job.get("name", "")

                
                if any(
                    job_name.startswith(excluded) for excluded in self.excluded_jobs
                ):
                    continue

                job_total_runs[job_name] += 1
                conclusion = job.get("conclusion")

                if conclusion == "failure":
                    
                    job_total_failures[job_name] += 1
                    job_current_streak[job_name] += 1

                    
                    if job_current_streak[job_name] == 1:
                        job_first_failure_in_streak[job_name] = {
                            **run_info,
                            "job_name": job_name,
                            "job_id": job.get("id"),
                            "job_url": job.get("html_url", run_info["url"]),
                            "conclusion": conclusion,
                        }

                    
                    job_last_failure_in_streak[job_name] = {
                        **run_info,
                        "job_name": job_name,
                        "job_id": job.get("id"),
                        "job_url": job.get("html_url", run_info["url"]),
                        "conclusion": conclusion,
                    }

                    
                    if job_current_streak[job_name] > job_max_streak[job_name]:
                        job_max_streak[job_name] = job_current_streak[job_name]

                elif conclusion == "success":
                    
                    if job_current_streak[job_name] > 0:
                        
                        job_recovery_info[job_name] = {
                            **run_info,
                            "job_name": job_name,
                            "streak_length": job_current_streak[job_name],
                        }

                    job_current_streak[job_name] = 0
                    job_first_failure_in_streak[job_name] = None
                    job_last_failure_in_streak[job_name] = None

                
                run_attempt = job.get("run_attempt", 1)

                
                if conclusion == "success":
                    status = "✅"
                elif conclusion == "failure":
                    status = "❌"
                else:
                    status = "⚪"

                
                if run_attempt > 1:
                    superscript_map = {
                        "2": "²",
                        "3": "³",
                        "4": "⁴",
                        "5": "⁵",
                        "6": "⁶",
                        "7": "⁷",
                        "8": "⁸",
                        "9": "⁹",
                    }
                    status += superscript_map.get(str(run_attempt), f"^{run_attempt}")

                job_recent_runs[job_name].append(
                    {
                        "run_number": run_info["run_number"],
                        "job_id": job.get("id"),  
                        "job_url": job.get("html_url", run_info["url"]),
                        "conclusion": conclusion,
                        "status": status,
                        "run_attempt": run_attempt,
                    }
                )

            time.sleep(0.05)

        
        job_streak_data = {}
        for job_name in job_current_streak.keys():
            
            recent_runs = job_recent_runs.get(job_name, [])[-10:]

            job_streak_data[job_name] = {
                "current_streak": job_current_streak[job_name],
                "max_streak": job_max_streak[job_name],
                "total_failures": job_total_failures[job_name],
                "total_runs": job_total_runs[job_name],
                "failure_rate": (
                    job_total_failures[job_name] / job_total_runs[job_name] * 100
                    if job_total_runs[job_name] > 0
                    else 0
                ),
                "first_failure_in_streak": job_first_failure_in_streak.get(job_name),
                "last_failure_in_streak": job_last_failure_in_streak.get(job_name),
                "recovery_info": job_recovery_info.get(job_name),
                "recent_runs": recent_runs,  
            }

        return job_streak_data, job_current_streak

    def analyze_test_failures_for_broken_jobs(
        self, job_streak_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Analyze test-level failures for jobs with current_streak >= 2 or failure_rate >= 50%.

        Args:
            job_streak_data: Dict mapping job_name -> job stats including recent_runs

        Returns:
            Dict mapping job_name -> {test_file -> test failure stats}
        """
        
        jobs_to_analyze = [
            (job_name, data)
            for job_name, data in job_streak_data.items()
            if data["current_streak"] >= 2 or data["failure_rate"] >= 50.0
        ]

        if not jobs_to_analyze:
            print("No broken or high-failure-rate jobs to analyze for test failures")
            return {}

        print(f"\nAnalyzing test-level failures for {len(jobs_to_analyze)} jobs...")

        job_test_failures = {}
        for i, (job_name, data) in enumerate(jobs_to_analyze, 1):
            print(
                f"  [{i}/{len(jobs_to_analyze)}] Analyzing test failures for: {job_name}"
            )
            recent_runs = data.get("recent_runs", [])

            if recent_runs:
                test_failures = self.analyze_test_failures_for_job(recent_runs)
                if test_failures:
                    job_test_failures[job_name] = test_failures

        print(f"Found test-level failures for {len(job_test_failures)} jobs")
        return job_test_failures

    def analyze_runner_specific_test_failures(
        self, runs: List[Dict]
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Analyze test failures grouped by runner to identify runner-specific issues.

        Args:
            runs: List of workflow runs to analyze

        Returns:
            Dict mapping runner_instance -> {test_file -> {"count": int, "jobs": [job_names]}}
        """
        print("\nAnalyzing runner-specific test failures...")

        runner_test_failures: Dict[str, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "jobs": [], "job_urls": []})
        )

        for run in runs:
            
            jobs = self.get_jobs_for_run(run.get("id"))

            for job in jobs:
                job_name = job.get("name", "")
                conclusion = job.get("conclusion")

                
                if any(
                    job_name.startswith(excluded) for excluded in self.excluded_jobs
                ):
                    continue

                
                if conclusion != "failure":
                    continue

                
                runner_name = (
                    job.get("runner_name")
                    or job.get("runner", {}).get("name")
                    or "unknown"
                )
                runner_id = job.get("runner_id") or job.get("runner", {}).get("id")
                runner_labels = job.get("labels", [])
                runner_labels_str = (
                    ", ".join(runner_labels) if runner_labels else "unknown"
                )

                
                if not runner_id or runner_labels_str == "unknown":
                    continue

                
                runner_instance_key = f"{runner_name}_{runner_id}"

                
                job_id = job.get("id")
                if job_id:
                    if job_id not in self.test_summaries:
                        logs = self.get_job_logs(job_id)
                        test_summary = self.parse_test_summary(logs) if logs else None
                    else:
                        test_summary = self.test_summaries[job_id]

                    if test_summary and test_summary.get("failed_tests"):
                        
                        for failed_test in test_summary["failed_tests"]:
                            test_file = failed_test["test_file"]

                            runner_test_failures[runner_instance_key][test_file][
                                "count"
                            ] += 1
                            runner_test_failures[runner_instance_key][test_file][
                                "jobs"
                            ].append(job_name)
                            runner_test_failures[runner_instance_key][test_file][
                                "job_urls"
                            ].append(
                                job.get(
                                    "html_url",
                                    f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
                                )
                            )

                            
                            if (
                                "runner_name"
                                not in runner_test_failures[runner_instance_key][
                                    test_file
                                ]
                            ):
                                runner_test_failures[runner_instance_key][test_file][
                                    "runner_name"
                                ] = runner_name
                                runner_test_failures[runner_instance_key][test_file][
                                    "runner_labels"
                                ] = runner_labels_str

            time.sleep(0.05)

        
        filtered_results = {}
        for runner_key, tests in runner_test_failures.items():
            
            multi_failure_tests = {
                test: data for test, data in tests.items() if data["count"] >= 2
            }
            if multi_failure_tests:
                filtered_results[runner_key] = multi_failure_tests

        print(f"Found {len(filtered_results)} runners with repeated test failures")
        return filtered_results

    
    def generate_failure_report(
        self,
        
        pr_test_nvidia_scheduled_data: Dict[str, Dict],
        pr_test_amd_scheduled_data: Dict[str, Dict],
        pr_test_xeon_scheduled_data: Dict[str, Dict],
        pr_test_xpu_scheduled_data: Dict[str, Dict],
        pr_test_npu_scheduled_data: Dict[str, Dict],
        nightly_nvidia_scheduled_data: Dict[str, Dict],
        nightly_amd_scheduled_data: Dict[str, Dict],
        nightly_intel_scheduled_data: Dict[str, Dict],
        nightly_npu_scheduled_data: Dict[str, Dict],
        
        pr_test_nvidia_general_data: Dict[str, Dict],
        pr_test_amd_general_data: Dict[str, Dict],
        pr_test_xeon_general_data: Dict[str, Dict],
        pr_test_xpu_general_data: Dict[str, Dict],
        pr_test_npu_general_data: Dict[str, Dict],
        nightly_nvidia_general_data: Dict[str, Dict],
        nightly_amd_general_data: Dict[str, Dict],
        nightly_intel_general_data: Dict[str, Dict],
        nightly_npu_general_data: Dict[str, Dict],
        
        runner_stats: Optional[Dict[str, Dict]] = None,
        runner_instance_data: Optional[Dict[str, Dict]] = None,
        runner_streak_data: Optional[Dict[str, Dict]] = None,
        runner_instance_streak_data: Optional[Dict[str, Dict]] = None,
        online_runners: Optional[Dict[str, Dict]] = None,
        
        job_test_failures: Optional[Dict[str, Dict[str, Dict]]] = None,
        
        job_test_failures_general: Optional[Dict[str, Dict[str, Dict]]] = None,
        
        runner_test_failures: Optional[Dict[str, Dict[str, Dict]]] = None,
        
        output_file: Optional[str] = None,
        pr_test_scheduled_limit: int = 12,
        nightly_scheduled_limit: int = 6,
        general_limit: int = 100,
    ):
        """Generate detailed failure analysis report."""
        print("\n" + "=" * 80)
        print("SGLang Consecutive Failures Analysis Report")
        print("=" * 80)

        
        combined_general_data = {
            **pr_test_nvidia_general_data,
            **pr_test_amd_general_data,
            **pr_test_xeon_general_data,
            **pr_test_xpu_general_data,
            **pr_test_npu_general_data,
            **nightly_nvidia_general_data,
            **nightly_amd_general_data,
            **nightly_intel_general_data,
            **nightly_npu_general_data,
        }

        
        sorted_jobs = sorted(
            combined_general_data.items(),
            key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
            reverse=True,
        )

        
        
        overall_avg_queue = 0
        overall_p90_queue = 0
        if runner_stats:
            all_avg_queue_times = [
                stats["avg_queue_time_seconds"]
                for stats in runner_stats.values()
                if stats["queue_time_samples"] > 0
            ]
            all_p90_queue_times = [
                stats["p90_queue_time_seconds"]
                for stats in runner_stats.values()
                if stats["queue_time_samples"] > 0
            ]
            if all_avg_queue_times:
                overall_avg_queue = sum(all_avg_queue_times) / len(all_avg_queue_times)
                overall_p90_queue = sum(all_p90_queue_times) / len(all_p90_queue_times)

        
        pr_scheduled_combined = {
            **pr_test_nvidia_scheduled_data,
            **pr_test_amd_scheduled_data,
            **pr_test_xeon_scheduled_data,
            **pr_test_xpu_scheduled_data,
            **pr_test_npu_scheduled_data,
        }
        nightly_scheduled_combined = {
            **nightly_nvidia_scheduled_data,
            **nightly_amd_scheduled_data,
            **nightly_intel_scheduled_data,
            **nightly_npu_scheduled_data,
        }

        pr_main_count = len(pr_scheduled_combined)
        pr_main_with_streaks = sum(
            1 for d in pr_scheduled_combined.values() if d["current_streak"] >= 2
        )
        nightly_main_count = len(nightly_scheduled_combined)
        nightly_main_with_streaks = sum(
            1 for d in nightly_scheduled_combined.values() if d["current_streak"] >= 2
        )

        report_data = {
            "summary": {
                "total_jobs": len(sorted_jobs),
                "jobs_with_streaks": sum(
                    1 for j in sorted_jobs if j[1]["current_streak"] > 0
                ),
                "total_runners": len(runner_stats) if runner_stats else 0,
                "analysis_timestamp": datetime.now().isoformat(),
                "avg_queue_time_seconds": overall_avg_queue,
                "p90_queue_time_seconds": overall_p90_queue,
                "pr_main_count": pr_main_count,
                "pr_main_with_streaks": pr_main_with_streaks,
                "nightly_main_count": nightly_main_count,
                "nightly_main_with_streaks": nightly_main_with_streaks,
            },
            "pr_test_scheduled_limit": pr_test_scheduled_limit,
            "nightly_scheduled_limit": nightly_scheduled_limit,
            "general_limit": general_limit,
            
            "pr_test_nvidia_scheduled_data": pr_test_nvidia_scheduled_data,
            "pr_test_amd_scheduled_data": pr_test_amd_scheduled_data,
            "pr_test_xeon_scheduled_data": pr_test_xeon_scheduled_data,
            "pr_test_xpu_scheduled_data": pr_test_xpu_scheduled_data,
            "pr_test_npu_scheduled_data": pr_test_npu_scheduled_data,
            "nightly_nvidia_scheduled_data": nightly_nvidia_scheduled_data,
            "nightly_amd_scheduled_data": nightly_amd_scheduled_data,
            "nightly_intel_scheduled_data": nightly_intel_scheduled_data,
            "nightly_npu_scheduled_data": nightly_npu_scheduled_data,
            
            "pr_test_nvidia_general_data": pr_test_nvidia_general_data,
            "pr_test_amd_general_data": pr_test_amd_general_data,
            "pr_test_xeon_general_data": pr_test_xeon_general_data,
            "pr_test_xpu_general_data": pr_test_xpu_general_data,
            "pr_test_npu_general_data": pr_test_npu_general_data,
            "nightly_nvidia_general_data": nightly_nvidia_general_data,
            "nightly_amd_general_data": nightly_amd_general_data,
            "nightly_intel_general_data": nightly_intel_general_data,
            "nightly_npu_general_data": nightly_npu_general_data,
            "runner_stats": runner_stats if runner_stats else {},
            "runner_instance_data": (
                runner_instance_data if runner_instance_data else {}
            ),
            "runner_streak_data": runner_streak_data if runner_streak_data else {},
            "runner_instance_streak_data": (
                runner_instance_streak_data if runner_instance_streak_data else {}
            ),
            "job_test_failures": job_test_failures if job_test_failures else {},
            "job_test_failures_general": (
                job_test_failures_general if job_test_failures_general else {}
            ),
            "runner_test_failures": (
                runner_test_failures if runner_test_failures else {}
            ),
            "online_runners": online_runners if online_runners else {},
        }

        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed report saved to: {output_file}")

        print("=" * 80)

        return report_data

    def generate_github_summary(self, report_data: Dict):
        """Generate GitHub Actions Step Summary."""
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("Not running in GitHub Actions, skipping summary generation")
                return

            print("Generating GitHub Actions summary...")

            summary_lines = []
            summary_lines.append("# SGLang Consecutive Failures Analysis")
            summary_lines.append("")
            summary_lines.append(
                f"**Analysis Timestamp:** {report_data['summary']['analysis_timestamp']}"
            )
            summary_lines.append(
                "_Note: Recent runs are shown oldest → latest (left to right)_"
            )
            summary_lines.append("")

            
            summary_lines.append("<details>")
            summary_lines.append(
                "<summary>📊 Summary Statistics (click to expand)</summary>"
            )
            summary_lines.append("")
            summary_lines.append("| Metric | Count |")
            summary_lines.append("|--------|-------|")
            summary_lines.append(
                f"| Total (unique) jobs analyzed | {report_data['summary']['total_jobs']} |"
            )
            summary_lines.append(
                f"| Jobs with Active Failure Streaks | {report_data['summary']['jobs_with_streaks']} |"
            )

            
            pr_main_count = report_data["summary"].get("pr_main_count", 0)
            pr_main_with_streaks = report_data["summary"].get("pr_main_with_streaks", 0)
            nightly_main_count = report_data["summary"].get("nightly_main_count", 0)
            nightly_main_with_streaks = report_data["summary"].get(
                "nightly_main_with_streaks", 0
            )

            summary_lines.append(
                f"| PR Test Jobs on Main (scheduled) | {pr_main_count} ({pr_main_with_streaks} with streaks) |"
            )
            summary_lines.append(
                f"| Nightly Test Jobs on Main (scheduled) | {nightly_main_count} ({nightly_main_with_streaks} with streaks) |"
            )

            summary_lines.append(
                f"| Total Runners Analyzed | {report_data['summary']['total_runners']} |"
            )
            summary_lines.append("")
            summary_lines.append("</details>")
            summary_lines.append("")

            
            runner_stats = report_data.get("runner_stats", {})
            online_runners = report_data.get("online_runners", {})
            if runner_stats:
                summary_lines.append("<details>")
                summary_lines.append(
                    "<summary>📊 Runner Statistics (by type) (click to expand)</summary>"
                )
                summary_lines.append("")
                summary_lines.append(
                    "_High queue times indicate that runner type may need more workers. Online column shows current runner availability._"
                )
                summary_lines.append("")
                summary_lines.append(
                    "| Runner Type | Online | Avg Queue | P90 Queue | # of Jobs Processed | Jobs Using This Runner |"
                )
                summary_lines.append(
                    "|-------------|--------|-----------|-----------|---------------------|------------------------|"
                )

                
                sorted_runners = sorted(
                    runner_stats.items(),
                    key=lambda x: x[1].get("p90_queue_time_seconds", 0),
                    reverse=True,
                )

                for runner_key, stats in sorted_runners:
                    avg_queue = stats.get("avg_queue_time_seconds", 0)
                    p90_queue = stats.get("p90_queue_time_seconds", 0)
                    total_jobs = stats.get("total_jobs", 0)

                    
                    
                    online_count = online_runners.get(runner_key)
                    if not online_count:
                        
                        best_match = None
                        best_match_len = 0
                        for online_key, online_stats in online_runners.items():
                            if online_key in runner_key or runner_key in online_key:
                                
                                if len(online_key) > best_match_len:
                                    best_match = online_stats
                                    best_match_len = len(online_key)
                        online_count = best_match
                    if online_count:
                        online_str = f"{online_count['online']}/{online_count['total']}"
                    else:
                        online_str = "N/A"

                    
                    jobs_total = stats.get("jobs_total", {})
                    unique_jobs = list(jobs_total.keys())
                    
                    job_names_short = [
                        (j if len(j) <= 25 else j[:22] + "...") for j in unique_jobs[:3]
                    ]
                    jobs_str = ", ".join(f"`{j}`" for j in job_names_short)
                    if len(unique_jobs) > 3:
                        jobs_str += f" +{len(unique_jobs) - 3} more"

                    
                    avg_str = f"{avg_queue / 60:.1f}m" if avg_queue > 0 else "N/A"
                    p90_str = f"{p90_queue / 60:.1f}m" if p90_queue > 0 else "N/A"

                    
                    display_name = (
                        runner_key if len(runner_key) <= 35 else runner_key[:32] + "..."
                    )

                    
                    if p90_queue > 600:
                        summary_lines.append(
                            f"| <span style='color:orange'>`{display_name}`</span> | <span style='color:orange'>{online_str}</span> | <span style='color:orange'>{avg_str}</span> | <span style='color:orange'>{p90_str}</span> | <span style='color:orange'>{total_jobs}</span> | {jobs_str} |"
                        )
                    else:
                        summary_lines.append(
                            f"| `{display_name}` | {online_str} | {avg_str} | {p90_str} | {total_jobs} | {jobs_str} |"
                        )

                summary_lines.append("")
                summary_lines.append("</details>")
                summary_lines.append("")

            
            job_test_failures = report_data.get("job_test_failures", {})
            job_test_failures_general = report_data.get("job_test_failures_general", {})

            
            def generate_job_section_md(
                title: str,
                data: Dict[str, Dict],
                show_test_failures: bool = True,
                test_failures_dict: Optional[Dict[str, Dict[str, Dict]]] = None,
            ):
                sorted_data = sorted(
                    data.items(),
                    key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                    reverse=True,
                )
                broken = [
                    (name, d) for name, d in sorted_data if d["current_streak"] >= 2
                ]
                high_failure_rate = [
                    (name, d)
                    for name, d in sorted_data
                    if d["current_streak"] < 2
                    and d["failure_rate"] >= 50.0
                    and d["total_failures"] > 0
                ]
                recently_failed = [
                    (name, d)
                    for name, d in sorted_data
                    if d["current_streak"] < 2
                    and d["failure_rate"] < 50.0
                    and d["total_failures"] > 0
                ]

                
                summary_lines.append(f"## {title}")
                summary_lines.append("")

                
                if show_test_failures:
                    
                    active_test_failures = (
                        test_failures_dict
                        if test_failures_dict is not None
                        else job_test_failures
                    )

                    
                    all_test_failures = []

                    
                    for job_name, job_data in broken:
                        test_failures = active_test_failures.get(job_name, {})
                        if test_failures and not test_failures.get("_no_test_summary"):
                            for test_file, test_data in test_failures.items():
                                if not test_file.startswith("_"):  
                                    all_test_failures.append(
                                        {
                                            "job_name": job_name,
                                            "test_file": test_file,
                                            "test_data": test_data,
                                            "job_data": job_data,
                                        }
                                    )

                    
                    for job_name, job_data in high_failure_rate:
                        test_failures = active_test_failures.get(job_name, {})
                        if test_failures and not test_failures.get("_no_test_summary"):
                            for test_file, test_data in test_failures.items():
                                if not test_file.startswith("_"):
                                    all_test_failures.append(
                                        {
                                            "job_name": job_name,
                                            "test_file": test_file,
                                            "test_data": test_data,
                                            "job_data": job_data,
                                        }
                                    )

                    
                    all_test_failures.sort(
                        key=lambda x: (
                            x["test_data"]["current_streak"],
                            x["test_data"]["total_failures"],
                        ),
                        reverse=True,
                    )

                    
                    streak_tests = [
                        t
                        for t in all_test_failures
                        if t["test_data"]["current_streak"] >= 2
                    ]

                    
                    non_streak_tests = []
                    for t in all_test_failures:
                        if t["test_data"]["current_streak"] < 2:
                            
                            recent_runs = t["test_data"].get("recent_runs", [])
                            if recent_runs:
                                
                                total_runs = len(recent_runs)
                                failed_runs = sum(
                                    1 for r in recent_runs if r.get("failed") == True
                                )
                                failure_rate = (
                                    (failed_runs / total_runs * 100)
                                    if total_runs > 0
                                    else 0
                                )

                                
                                if failed_runs >= 1:
                                    
                                    t["failure_rate"] = failure_rate
                                    t["failed_runs"] = failed_runs
                                    t["total_test_runs"] = total_runs
                                    non_streak_tests.append(t)

                    
                    non_streak_tests.sort(key=lambda x: x["failure_rate"], reverse=True)

                    
                    if streak_tests:
                        summary_lines.append(
                            "🔥 **Tests with consecutive failures (≥2) & currently failing**"
                        )
                        summary_lines.append("")

                        
                        has_timeout = any(
                            any(
                                r.get("status") == "⏱️"
                                for r in t["test_data"].get("recent_runs", [])
                            )
                            for t in streak_tests
                        )
                        if has_timeout:
                            summary_lines.append(
                                "_Note: ⏱️ indicates test was last running when logs cut off (possible timeout)_"
                            )
                            summary_lines.append("")
                        summary_lines.append(
                            "| Test File | Job | Failures | Streak | First | Last | Recent Runs (oldest → latest) |"
                        )
                        summary_lines.append(
                            "|-----------|-----|----------|--------|-------|------|-------------------------------|"
                        )

                        for test_info in streak_tests[:20]:  
                            test_file = test_info["test_file"]
                            job_name = test_info["job_name"]
                            test_data = test_info["test_data"]
                            job_data = test_info["job_data"]

                            test_display = test_file
                            job_display = job_name

                            
                            first_failure = job_data.get("first_failure_in_streak")
                            first_str = (
                                f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                                if first_failure
                                else "N/A"
                            )

                            last_failure = job_data.get("last_failure_in_streak")
                            last_str = (
                                f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                                if last_failure
                                else "N/A"
                            )

                            
                            streak_str = f"🔥 {test_data['current_streak']}"

                            
                            recent_runs = test_data.get("recent_runs", [])
                            if recent_runs:
                                history_links = "… " + " ".join(
                                    [
                                        f"[{r['status']}]({r['job_url']})"
                                        for r in recent_runs[-10:]
                                    ]  
                                )
                            else:
                                history_links = "N/A"

                            
                            if test_data["current_streak"] >= 3:
                                summary_lines.append(
                                    f"| <span style='color:red'>`{test_display}`</span> | <span style='color:red'>`{job_display}`</span> | "
                                    f"<span style='color:red'>{test_data['total_failures']}</span> | <span style='color:red'>{streak_str}</span> | "
                                    f"<span style='color:red'>{first_str}</span> | <span style='color:red'>{last_str}</span> | "
                                    f"<span style='color:red'>{history_links}</span> |"
                                )
                            else:
                                summary_lines.append(
                                    f"| `{test_display}` | `{job_display}` | {test_data['total_failures']} | {streak_str} | "
                                    f"{first_str} | {last_str} | {history_links} |"
                                )

                        summary_lines.append("")

                    
                    if non_streak_tests:
                        summary_lines.append(
                            "📋 **Other tests with failures (ranked by failure rate)**"
                        )
                        summary_lines.append("")

                        
                        has_timeout = any(
                            any(
                                r.get("status") == "⏱️"
                                for r in t["test_data"].get("recent_runs", [])
                            )
                            for t in non_streak_tests
                        )
                        if has_timeout:
                            summary_lines.append(
                                "_Note: ⏱️ indicates test was last running when logs cut off (possible timeout)_"
                            )
                            summary_lines.append("")
                        summary_lines.append(
                            "| Test File | Job | Failed | Total | Fail Rate | Recent Runs (oldest → latest) |"
                        )
                        summary_lines.append(
                            "|-----------|-----|--------|-------|-----------|-------------------------------|"
                        )

                        for test_info in non_streak_tests[:20]:  
                            test_file = test_info["test_file"]
                            job_name = test_info["job_name"]
                            test_data = test_info["test_data"]
                            failure_rate = test_info["failure_rate"]
                            failed_runs = test_info["failed_runs"]
                            total_test_runs = test_info["total_test_runs"]

                            test_display = test_file
                            job_display = job_name

                            
                            recent_runs = test_data.get("recent_runs", [])
                            if recent_runs:
                                history_links = "… " + " ".join(
                                    [
                                        f"[{r['status']}]({r['job_url']})"
                                        for r in recent_runs[-10:]
                                    ]
                                )
                            else:
                                history_links = "N/A"

                            
                            if failure_rate >= 50.0:
                                summary_lines.append(
                                    f"| <span style='color:orange'>`{test_display}`</span> | <span style='color:orange'>`{job_display}`</span> | "
                                    f"<span style='color:orange'>{failed_runs}</span> | <span style='color:orange'>{total_test_runs}</span> | "
                                    f"<span style='color:orange'>{failure_rate:.1f}%</span> | <span style='color:orange'>{history_links}</span> |"
                                )
                            else:
                                summary_lines.append(
                                    f"| `{test_display}` | `{job_display}` | {failed_runs} | {total_test_runs} | "
                                    f"{failure_rate:.1f}% | {history_links} |"
                                )

                        summary_lines.append("")

                    
                    if (
                        not streak_tests
                        and not non_streak_tests
                        and (broken or high_failure_rate)
                    ):
                        summary_lines.append(
                            "_No test-level failure data available for this workflow_"
                        )
                        summary_lines.append("")

                
                summary_lines.append("<details>")
                summary_lines.append(
                    "<summary><b>📊 Job-level summary (click to expand)</b></summary>"
                )
                summary_lines.append("")

                
                if broken:
                    summary_lines.append("<details>")
                    summary_lines.append(
                        "<summary>🔥 <b>Consecutive failures (≥2) & currently failing</b></summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Current | Max | Runs | First | Last | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|---------|-----|------|-------|------|-------------------------------|"
                    )
                    for job_name, d in broken[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )

                        first_failure = d.get("first_failure_in_streak")
                        first_str = (
                            f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                            if first_failure
                            else "N/A"
                        )

                        last_failure = d.get("last_failure_in_streak")
                        last_str = (
                            f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                            if last_failure
                            else "N/A"
                        )

                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = "… " + " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        if d["current_streak"] >= 3:
                            summary_lines.append(
                                f"| <span style='color:red'>`{display_name}`</span> | <span style='color:red'>{d['current_streak']}</span> | <span style='color:red'>{d['max_streak']}</span> | <span style='color:red'>{d['total_runs']}</span> | "
                                f"<span style='color:red'>{first_str}</span> | <span style='color:red'>{last_str}</span> | <span style='color:red'>{history_links}</span> |"
                            )
                        else:
                            summary_lines.append(
                                f"| `{display_name}` | {d['current_streak']} | {d['max_streak']} | {d['total_runs']} | "
                                f"{first_str} | {last_str} | {history_links} |"
                            )

                    summary_lines.append("")
                    summary_lines.append("</details>")
                    summary_lines.append("")

                
                if high_failure_rate:
                    summary_lines.append("<details>")
                    summary_lines.append(
                        "<summary>⚠️ <b>No current failure streak but high intermittent failure rate (≥50%)</b></summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Failures | Fail Rate | Total Runs | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|----------|-----------|------------|-------------------------------|"
                    )
                    for job_name, d in high_failure_rate[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = "… " + " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        summary_lines.append(
                            f"| <span style='color:orange'>`{display_name}`</span> | <span style='color:orange'>{d['total_failures']}</span> | <span style='color:orange'>{d['failure_rate']:.1f}%</span> | <span style='color:orange'>{d['total_runs']}</span> | <span style='color:orange'>{history_links}</span> |"
                        )

                    summary_lines.append("")
                    summary_lines.append("</details>")
                    summary_lines.append("")

                
                if recently_failed:
                    max_total_runs = max(d["total_runs"] for _, d in recently_failed)
                    summary_lines.append("<details>")
                    summary_lines.append(
                        f"<summary>📋 <b>No current failure streak, but had failures in the past {max_total_runs} runs - {len(recently_failed)} jobs</b></summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Failures | Fail Rate | Total Runs | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|----------|-----------|------------|-------------------------------|"
                    )
                    for job_name, d in recently_failed[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = "… " + " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        summary_lines.append(
                            f"| `{display_name}` | {d['total_failures']} | {d['failure_rate']:.1f}% | {d['total_runs']} | {history_links} |"
                        )
                    summary_lines.append("")
                    summary_lines.append("</details>")
                    summary_lines.append("")

                
                if not broken and not high_failure_rate and recently_failed:
                    max_total_runs = max(d["total_runs"] for _, d in recently_failed)
                    summary_lines.append(
                        f"✅ No jobs with active failure streaks, but **{len(recently_failed)} jobs** had failures in the past **{max_total_runs} runs**"
                    )
                    summary_lines.append("")
                elif not broken and not high_failure_rate and not recently_failed:
                    summary_lines.append("✅ **No jobs with active failure streaks**")
                    summary_lines.append("")

                summary_lines.append("</details>")
                summary_lines.append("")

            
            summary_lines.append("---")
            summary_lines.append("# 🖥️ RUNNER HEALTH")
            summary_lines.append("")

            
            if report_data.get("runner_instance_data") and report_data.get(
                "runner_instance_streak_data"
            ):
                
                combined_data = []
                for instance_key, stats in report_data["runner_instance_data"].items():
                    streak_data = report_data["runner_instance_streak_data"].get(
                        instance_key, {}
                    )
                    combined_data.append(
                        {
                            "runner_name": stats.get("runner_name", "unknown"),
                            "current_streak": streak_data.get("current_streak", 0),
                            "max_streak": streak_data.get("max_streak", 0),
                            "failure_rate": stats["failure_rate"],
                            "total_jobs": stats["total_jobs"],
                            "unique_jobs": len(stats.get("jobs_failed", {})),
                            "avg_queue": stats.get("avg_queue_time_seconds", 0),
                            "first_failure": streak_data.get("first_failure_in_streak"),
                            "last_failure": streak_data.get("last_failure_in_streak"),
                        }
                    )

                sorted_runners = sorted(
                    combined_data,
                    key=lambda x: (
                        x["current_streak"],
                        x["max_streak"],
                        x["failure_rate"],
                    ),
                    reverse=True,
                )

                
                runners_with_streak = [
                    r for r in sorted_runners if r["current_streak"] >= 2
                ]
                runners_high_fail_rate = [
                    r
                    for r in sorted_runners
                    if r["current_streak"] < 2
                    and r["failure_rate"] >= 50.0
                    and r["total_jobs"] >= 2
                ]

                
                summary_lines.append("## Workers")
                summary_lines.append("")

                
                if runners_with_streak:
                    summary_lines.append(
                        "🔥 **Consecutive failures (≥2) & currently failing**"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Machine Name | Current Streak | Max | Fail Rate | Avg Queue | Total Jobs | Unique Jobs | First Failure | Last Failure |"
                    )
                    summary_lines.append(
                        "|--------------|----------------|-----|-----------|-----------|------------|-------------|---------------|--------------|"
                    )

                    for runner_data in runners_with_streak[:15]:
                        display_name = (
                            runner_data["runner_name"]
                            if len(runner_data["runner_name"]) <= 28
                            else runner_data["runner_name"][:25] + "..."
                        )

                        avg_queue_str = (
                            f"{runner_data['avg_queue'] / 60:.1f}m"
                            if runner_data["avg_queue"] > 0
                            else "N/A"
                        )

                        first_failure = runner_data.get("first_failure")
                        first_str = (
                            f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                            if first_failure
                            else "N/A"
                        )

                        last_failure = runner_data.get("last_failure")
                        last_str = (
                            f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                            if last_failure
                            else "N/A"
                        )

                        
                        if runner_data["current_streak"] >= 3:
                            summary_lines.append(
                                f"| <span style='color:red'>`{display_name}`</span> | <span style='color:red'>{runner_data['current_streak']}</span> | <span style='color:red'>{runner_data['max_streak']}</span> | "
                                f"<span style='color:red'>{runner_data['failure_rate']:.1f}%</span> | <span style='color:red'>{avg_queue_str}</span> | <span style='color:red'>{runner_data['total_jobs']}</span> | <span style='color:red'>{runner_data.get('unique_jobs', 0)}</span> | <span style='color:red'>{first_str}</span> | <span style='color:red'>{last_str}</span> |"
                            )
                        else:
                            summary_lines.append(
                                f"| `{display_name}` | {runner_data['current_streak']} | {runner_data['max_streak']} | "
                                f"{runner_data['failure_rate']:.1f}% | {avg_queue_str} | {runner_data['total_jobs']} | {runner_data.get('unique_jobs', 0)} | {first_str} | {last_str} |"
                            )

                    summary_lines.append("")

                
                if runners_high_fail_rate:
                    summary_lines.append(
                        "⚠️ **No current failure streak but high failure rate (≥50%)**"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Machine Name | Fail Rate | Avg Queue | Total Jobs | Unique Jobs |"
                    )
                    summary_lines.append(
                        "|--------------|-----------|-----------|------------|-------------|"
                    )

                    for runner_data in runners_high_fail_rate[:15]:
                        display_name = (
                            runner_data["runner_name"]
                            if len(runner_data["runner_name"]) <= 28
                            else runner_data["runner_name"][:25] + "..."
                        )

                        avg_queue_str = (
                            f"{runner_data['avg_queue'] / 60:.1f}m"
                            if runner_data["avg_queue"] > 0
                            else "N/A"
                        )

                        summary_lines.append(
                            f"| <span style='color:orange'>`{display_name}`</span> | <span style='color:orange'>{runner_data['failure_rate']:.1f}%</span> | "
                            f"<span style='color:orange'>{avg_queue_str}</span> | <span style='color:orange'>{runner_data['total_jobs']}</span> | "
                            f"<span style='color:orange'>{runner_data.get('unique_jobs', 0)}</span> |"
                        )

                    summary_lines.append("")

                
                if not runners_with_streak and not runners_high_fail_rate:
                    summary_lines.append(
                        "✅ **No runners with active failure streaks or high failure rates**"
                    )
                    summary_lines.append("")

            
            runner_test_failures = report_data.get("runner_test_failures", {})
            if runner_test_failures:
                summary_lines.append("## Runner-Specific Test Failures")
                summary_lines.append("")
                summary_lines.append(
                    "_Tests that fail multiple times on the same runner (possible runner-specific issues)_"
                )
                summary_lines.append("")

                
                sorted_runners = sorted(
                    runner_test_failures.items(),
                    key=lambda x: sum(test["count"] for test in x[1].values()),
                    reverse=True,
                )

                for runner_key, tests in sorted_runners[:10]:  
                    
                    sorted_tests = sorted(
                        tests.items(),
                        key=lambda x: x[1]["count"],
                        reverse=True,
                    )

                    
                    runner_name = sorted_tests[0][1].get("runner_name", runner_key)
                    total_failures = sum(test["count"] for test in tests.values())

                    summary_lines.append("<details>")
                    summary_lines.append(
                        f"<summary>🤖 <b>Runner: {runner_name}</b> ({len(tests)} tests, {total_failures} total failures)</summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append("| Test File | Failures | Jobs |")
                    summary_lines.append("|-----------|----------|------|")

                    for test_file, test_data in sorted_tests[
                        :15
                    ]:  
                        count = test_data["count"]
                        jobs = test_data["jobs"]
                        job_urls = test_data["job_urls"]

                        
                        test_display = (
                            test_file
                            if len(test_file) <= 35
                            else test_file[:32] + "..."
                        )

                        
                        job_links = []
                        for job_name, job_url in zip(jobs[:3], job_urls[:3]):
                            job_short = (
                                job_name
                                if len(job_name) <= 20
                                else job_name[:17] + "..."
                            )
                            job_links.append(f"[{job_short}]({job_url})")

                        jobs_str = ", ".join(job_links)
                        if len(jobs) > 3:
                            jobs_str += f" +{len(jobs) - 3} more"

                        
                        if count >= 3:
                            summary_lines.append(
                                f"| <span style='color:red'>`{test_display}`</span> | <span style='color:red'>{count}</span> | <span style='color:red'>{jobs_str}</span> |"
                            )
                        else:
                            summary_lines.append(
                                f"| `{test_display}` | {count} | {jobs_str} |"
                            )

                    summary_lines.append("")
                    summary_lines.append("</details>")
                    summary_lines.append("")

            
            summary_lines.append("---")
            summary_lines.append("# 📅 SCHEDULED RUNS (Main Branch)")
            summary_lines.append("")

            
            pr_sched_limit = report_data.get("pr_test_scheduled_limit", 12)
            nightly_sched_limit = report_data.get("nightly_scheduled_limit", 6)

            
            generate_job_section_md(
                f"1. PR Test NVIDIA - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_nvidia_scheduled_data", {}),
            )
            generate_job_section_md(
                f"2. PR Test AMD - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_amd_scheduled_data", {}),
            )
            generate_job_section_md(
                f"3. PR Test Xeon - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_xeon_scheduled_data", {}),
            )
            generate_job_section_md(
                f"4. PR Test XPU - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_xpu_scheduled_data", {}),
            )
            generate_job_section_md(
                f"5. PR Test NPU - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_npu_scheduled_data", {}),
            )

            
            generate_job_section_md(
                f"6. Nightly NVIDIA - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_nvidia_scheduled_data", {}),
            )
            generate_job_section_md(
                f"7. Nightly AMD - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_amd_scheduled_data", {}),
            )
            generate_job_section_md(
                f"8. Nightly Intel - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_intel_scheduled_data", {}),
            )
            generate_job_section_md(
                f"9. Nightly NPU - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_npu_scheduled_data", {}),
            )

            
            summary_lines.append("---")
            summary_lines.append("# 🌍 GENERAL RUNS (All Branches)")
            summary_lines.append("")

            gen_limit = report_data.get("general_limit", 100)

            
            generate_job_section_md(
                f"10. PR Test NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("pr_test_nvidia_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"11. PR Test AMD - General (latest {gen_limit} runs)",
                report_data.get("pr_test_amd_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"12. PR Test Xeon - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xeon_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"13. PR Test XPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xpu_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"14. PR Test NPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_npu_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )

            
            generate_job_section_md(
                f"15. Nightly NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("nightly_nvidia_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"16. Nightly AMD - General (latest {gen_limit} runs)",
                report_data.get("nightly_amd_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"17. Nightly Intel - General (latest {gen_limit} runs)",
                report_data.get("nightly_intel_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )
            generate_job_section_md(
                f"18. Nightly NPU - General (latest {gen_limit} runs)",
                report_data.get("nightly_npu_general_data", {}),
                show_test_failures=True,
                test_failures_dict=job_test_failures_general,
            )

            
            with open(github_step_summary, "a", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="SGLang Consecutive Failures Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of workflow runs to analyze per workflow for general analysis (default: 100)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (optional, only writes if specified)",
    )

    args = parser.parse_args()

    analyzer = SGLangFailuresAnalyzer(args.token)

    try:
        
        print("\n" + "=" * 80)
        print("FETCHING WORKFLOW RUNS")
        print("=" * 80)

        
        pr_test_scheduled_limit = 12  
        nightly_scheduled_limit = 6  

        
        
        pr_test_nvidia_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test.yml"],
            filters={"event": "schedule"},
        )
        
        pr_test_amd_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-amd.yml"],
            filters={"branch": "main"},
        )
        pr_test_xeon_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-xeon.yml"],
            filters={"branch": "main"},
        )
        pr_test_xpu_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-xpu.yml"],
            filters={"branch": "main"},
        )
        pr_test_npu_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-npu.yml"],
            filters={"branch": "main"},
        )

        
        nightly_nvidia_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-nvidia.yml"],
            filters={"event": "schedule"},
        )
        nightly_amd_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-amd.yml"],
            filters={"event": "schedule"},
        )
        nightly_intel_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-intel.yml"],
            filters={"event": "schedule"},
        )
        nightly_npu_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-npu.yml"],
            filters={"event": "schedule"},
        )

        
        
        pr_test_nvidia_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test.yml"],
        )
        pr_test_amd_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-amd.yml"],
        )
        pr_test_xeon_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-xeon.yml"],
        )
        pr_test_xpu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-xpu.yml"],
        )
        pr_test_npu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-npu.yml"],
        )

        
        nightly_nvidia_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-nvidia.yml"],
        )
        nightly_amd_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-amd.yml"],
        )
        nightly_intel_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-intel.yml"],
        )
        nightly_npu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-npu.yml"],
        )

        
        runner_runs = pr_test_nvidia_general_runs + nightly_nvidia_general_runs

        if not runner_runs and not pr_test_nvidia_scheduled_runs:
            print("No workflow runs found")
            return

        print("\n" + "=" * 80)
        print("ANALYZING CONSECUTIVE FAILURES")
        print("=" * 80)

        
        pr_test_nvidia_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_nvidia_scheduled_runs)
            if pr_test_nvidia_scheduled_runs
            else ({}, {})
        )
        pr_test_amd_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_amd_scheduled_runs)
            if pr_test_amd_scheduled_runs
            else ({}, {})
        )
        pr_test_xeon_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xeon_scheduled_runs)
            if pr_test_xeon_scheduled_runs
            else ({}, {})
        )
        pr_test_xpu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xpu_scheduled_runs)
            if pr_test_xpu_scheduled_runs
            else ({}, {})
        )
        pr_test_npu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_npu_scheduled_runs)
            if pr_test_npu_scheduled_runs
            else ({}, {})
        )

        nightly_nvidia_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_nvidia_scheduled_runs)
            if nightly_nvidia_scheduled_runs
            else ({}, {})
        )
        nightly_amd_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_amd_scheduled_runs)
            if nightly_amd_scheduled_runs
            else ({}, {})
        )
        nightly_intel_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_intel_scheduled_runs)
            if nightly_intel_scheduled_runs
            else ({}, {})
        )
        nightly_npu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_npu_scheduled_runs)
            if nightly_npu_scheduled_runs
            else ({}, {})
        )

        
        pr_test_nvidia_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_nvidia_general_runs)
            if pr_test_nvidia_general_runs
            else ({}, {})
        )
        pr_test_amd_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_amd_general_runs)
            if pr_test_amd_general_runs
            else ({}, {})
        )
        pr_test_xeon_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xeon_general_runs)
            if pr_test_xeon_general_runs
            else ({}, {})
        )
        pr_test_xpu_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xpu_general_runs)
            if pr_test_xpu_general_runs
            else ({}, {})
        )
        pr_test_npu_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_npu_general_runs)
            if pr_test_npu_general_runs
            else ({}, {})
        )

        nightly_nvidia_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_nvidia_general_runs)
            if nightly_nvidia_general_runs
            else ({}, {})
        )
        nightly_amd_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_amd_general_runs)
            if nightly_amd_general_runs
            else ({}, {})
        )
        nightly_intel_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_intel_general_runs)
            if nightly_intel_general_runs
            else ({}, {})
        )
        nightly_npu_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_npu_general_runs)
            if nightly_npu_general_runs
            else ({}, {})
        )

        
        (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        ) = analyzer.analyze_runner_health(runner_runs)

        
        online_runners = analyzer.get_online_runners()

        
        
        all_scheduled_data = {
            **pr_test_nvidia_scheduled_data,
            **pr_test_amd_scheduled_data,
            **pr_test_xeon_scheduled_data,
            **pr_test_xpu_scheduled_data,
            **pr_test_npu_scheduled_data,
            **nightly_nvidia_scheduled_data,
            **nightly_amd_scheduled_data,
            **nightly_intel_scheduled_data,
            **nightly_npu_scheduled_data,
        }
        job_test_failures = analyzer.analyze_test_failures_for_broken_jobs(
            all_scheduled_data
        )

        
        all_general_data = {
            **pr_test_nvidia_general_data,
            **pr_test_amd_general_data,
            **pr_test_xeon_general_data,
            **pr_test_xpu_general_data,
            **pr_test_npu_general_data,
            **nightly_nvidia_general_data,
            **nightly_amd_general_data,
            **nightly_intel_general_data,
            **nightly_npu_general_data,
        }
        job_test_failures_general = analyzer.analyze_test_failures_for_broken_jobs(
            all_general_data
        )

        
        runner_test_failures = analyzer.analyze_runner_specific_test_failures(
            runner_runs
        )

        
        report_data = analyzer.generate_failure_report(
            
            pr_test_nvidia_scheduled_data,
            pr_test_amd_scheduled_data,
            pr_test_xeon_scheduled_data,
            pr_test_xpu_scheduled_data,
            pr_test_npu_scheduled_data,
            nightly_nvidia_scheduled_data,
            nightly_amd_scheduled_data,
            nightly_intel_scheduled_data,
            nightly_npu_scheduled_data,
            
            pr_test_nvidia_general_data,
            pr_test_amd_general_data,
            pr_test_xeon_general_data,
            pr_test_xpu_general_data,
            pr_test_npu_general_data,
            nightly_nvidia_general_data,
            nightly_amd_general_data,
            nightly_intel_general_data,
            nightly_npu_general_data,
            
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
            online_runners,
            
            job_test_failures,
            job_test_failures_general,
            runner_test_failures,
            
            args.output,
            pr_test_scheduled_limit,
            nightly_scheduled_limit,
            args.limit,
        )

        
        analyzer.generate_github_summary(report_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
