"""
Command-line Batch Runner for IncuBrix AI Background Music Engine.
Accepts a JSON file or directory of test requests, executes them against the API service,
polls until completion, downloads generated audio and manifests, and outputs a formatted summary report.

Usage:
  python -m incubrix.batch_runner --input tests/batch_samples.json --output results/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import requests


def run_batch(input_path: str, output_dir: str, base_url: str = "http://127.0.0.1:8000"):
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[Error] Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Collect requests
    requests_to_run: List[Dict[str, Any]] = []
    if in_path.is_file():
        data = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            requests_to_run = data
        else:
            requests_to_run = [data]
    else:
        for json_file in sorted(in_path.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                requests_to_run.extend(data)
            else:
                requests_to_run.append(data)

    print(f"=== IncuBrix AI Background Music Engine Batch Runner ===")
    print(f"Target URL: {base_url}")
    print(f"Total Requests to Process: {len(requests_to_run)}")
    print(f"Saving Results to: {out_dir.resolve()}\n")

    summary_records = []

    for idx, test_case in enumerate(requests_to_run, 1):
        name = test_case.get("name", f"Test_{idx}")
        payload = test_case.get("payload", test_case)
        expected_status = test_case.get("expected_http_status", 202)

        print(f"[{idx}/{len(requests_to_run)}] Running '{name}'...")
        idempotency_key = f"batch_{int(time.time())}_{idx}_{name[:10].replace(' ', '_')}"

        record: Dict[str, Any] = {
            "name": name,
            "idempotency_key": idempotency_key,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "pending",
        }

        try:
            start_t = time.time()
            post_resp = requests.post(
                f"{base_url}/api/v1/music/jobs",
                json=payload,
                headers={"Idempotency-Key": idempotency_key, "Content-Type": "application/json"},
                timeout=10,
            )

            record["initial_http_status"] = post_resp.status_code

            if post_resp.status_code != 202:
                # Check if failure was expected
                err_data = post_resp.json()
                record["status"] = "rejected_as_expected" if post_resp.status_code == expected_status else "unexpected_failure"
                record["error"] = err_data
                print(f"  -> HTTP {post_resp.status_code}: {err_data.get('error', {}).get('code', 'Error')}")
                summary_records.append(record)
                continue

            job_data = post_resp.json()
            job_id = job_data["job_id"]
            record["job_id"] = job_id
            print(f"  -> Accepted (Job ID: {job_id}). Polling status...")

            # Poll until completion
            max_wait_s = 60
            poll_interval = 1.0
            elapsed_wait = 0.0
            final_job_status = "unknown"

            while elapsed_wait < max_wait_s:
                time.sleep(poll_interval)
                elapsed_wait += poll_interval
                status_resp = requests.get(f"{base_url}/api/v1/jobs/{job_id}", timeout=5)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    final_job_status = status_data["status"]
                    if final_job_status in ("succeeded", "failed"):
                        record["job_details"] = status_data
                        break

            record["final_status"] = final_job_status
            record["elapsed_s"] = round(time.time() - start_t, 2)

            if final_job_status == "succeeded":
                print(f"  -> SUCCEEDED in {record['elapsed_s']}s. Downloading assets...")
                # Download manifest
                m_resp = requests.get(f"{base_url}/api/v1/jobs/{job_id}/manifest", timeout=10)
                if m_resp.status_code == 200:
                    manifest = m_resp.json()
                    record["manifest"] = manifest
                    manifest_file = out_dir / f"{job_id}_manifest.json"
                    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                # Download mix.wav
                w_resp = requests.get(f"{base_url}/api/v1/jobs/{job_id}/download/mix.wav", timeout=15)
                if w_resp.status_code == 200:
                    (out_dir / f"{job_id}_mix.wav").write_bytes(w_resp.content)

                record["status"] = "success"
            else:
                print(f"  -> FAILED or Timed out. Final status: {final_job_status}")
                record["status"] = "job_failed"

        except Exception as e:
            print(f"  -> Exception: {str(e)}")
            record["status"] = "runner_exception"
            record["error"] = str(e)

        summary_records.append(record)

    # Save summary report
    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    print(f"\nBatch completed. Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="IncuBrix AI Background Music Engine Batch Runner")
    parser.add_argument("--input", default="tests/batch_samples.json", help="Path to input test requests JSON")
    parser.add_argument("--output", default="results", help="Directory to save audio outputs and manifests")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API service base URL")
    args = parser.parse_args()

    run_batch(args.input, args.output, args.url)


if __name__ == "__main__":
    main()
