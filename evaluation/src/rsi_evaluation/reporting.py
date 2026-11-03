"""Read Harbor results without importing generators or assuming a task verifier schema."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(job: Path) -> dict:
    result_path = job / "result.json"
    if not result_path.is_file():
        raise ValueError(f"No Harbor job result: {result_path}")
    overall = json.loads(result_path.read_text())
    snapshot_path = job / "task-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    config_path = job / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    rows = []
    for path in sorted(job.glob("*/result.json")):
        trial = json.loads(path.read_text())
        reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exception = trial.get("exception_info")
        verifier_path = path.parent / "verifier" / "report.json"
        diagnostic = json.loads(verifier_path.read_text()) if verifier_path.exists() else {}
        infrastructure = diagnostic.get("infrastructure_error") or diagnostic.get("error")
        if not trial.get("finished_at"):
            status = "incomplete"
        elif exception or infrastructure:
            status = "error"
        elif isinstance(reward, (int, float)) and not isinstance(reward, bool) and reward in (0, 1):
            status = "pass" if reward == 1 else "fail"
        else:
            status = "unscored"
        rows.append({
            "task": trial["task_name"], "trial": trial.get("trial_name", path.parent.name),
            "task_directory": Path((trial.get("task_id") or {}).get("path", "")).name or trial["task_name"],
            "agent": trial.get("agent_info"), "reward": reward, "status": status,
            "exception_info": exception, "verifier_infrastructure_error": infrastructure,
            "mismatch_count": diagnostic.get("mismatch_count", len(diagnostic["mismatches"]) if "mismatches" in diagnostic else None),
            "scoring": diagnostic.get("scoring"), "target_count": diagnostic.get("target_count"),
            "outside_target_mismatch_count": diagnostic.get("outside_target_mismatch_count"),
        })
    counts = Counter(row["status"] for row in rows)
    by_task = Counter(row["task_directory"] for row in rows)
    expected = snapshot.get("expected_trials", overall.get("n_total_trials"))
    attempts = snapshot.get("n_attempts", config.get("n_attempts", 1))
    expected_tasks = set(snapshot.get("tasks", {}))
    coverage = (not expected_tasks or expected_tasks == set(by_task))
    complete = bool(overall.get("finished_at")) and bool(rows) and len(rows) == expected and coverage
    scored = counts["pass"] + counts["fail"]
    all_scored = complete and scored == len(rows)
    agents = {json.dumps(row["agent"], sort_keys=True) for row in rows}
    single_attempt = attempts == 1 and all(n == 1 for n in by_task.values()) and len(agents) == 1
    return {
        "job": job.name, "expected_trials": expected, "recorded_trials": len(rows),
        "task_count": len(by_task), "configured_attempts_per_task": attempts,
        "complete": complete, "counts": dict(counts),
        "unrecorded_trials": max(0, expected - len(rows)) if expected is not None else None,
        "observed_scored_attempt_pass_rate": counts["pass"] / scored if scored else None,
        "pass_at_1": counts["pass"] / len(rows) if all_scored and single_attempt else None,
        "score_note": "Pass@1 is reported only for a complete, single-agent, one-attempt-per-task job with binary scores and no exceptions. The observed scored-attempt rate excludes errors and unscored trials; it is not the full-job score.",
        "trials": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one Harbor job into a JSON report")
    parser.add_argument("job", type=Path, help="Job directory, e.g. runs/oracle-check")
    parser.add_argument("--output", type=Path, help="Default: reports/<job-name>.json under current directory")
    args = parser.parse_args()
    try:
        report = summarize(args.job.expanduser().resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = args.output or Path("reports") / f"{args.job.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
