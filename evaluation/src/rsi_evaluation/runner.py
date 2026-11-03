"""Standalone Harbor evaluation for released spreadsheet tasks.

Paths are relative to --root (the standalone evaluation directory), not the
construction repository. Credentials are handled by Harbor via the environment.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path


def resolve_path(root: Path, value: str) -> str:
    return str((root / Path(value).expanduser()).resolve())


def prepare(root: Path, config: dict, job_name: str) -> tuple[dict, dict]:
    """Validate a local dataset and capture the exact task bytes before running."""
    from harbor.models.job.config import JobConfig
    from harbor.models.task.task import Task

    config = json.loads(json.dumps(config))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_name):
        raise ValueError("job-name must be a single name containing letters, digits, '.', '_' or '-'")
    config["job_name"] = job_name
    config["jobs_dir"] = resolve_path(root, config.get("jobs_dir", "runs"))
    if (Path(config["jobs_dir"]) / job_name).exists():
        raise ValueError(f"Job already exists: {job_name}; choose a new name")
    if config.get("tasks"):
        raise ValueError("This entry point uses local datasets; use Harbor directly for explicit tasks")
    selected = {}
    for dataset in config.get("datasets", []):
        if not dataset.get("path") or any(dataset.get(k) for k in ("repo", "name", "registry_url", "registry_path")):
            raise ValueError("Use a local dataset path in this entry point")
        dataset["path"] = resolve_path(root, dataset["path"])
        folder = Path(dataset["path"])
        if not folder.is_dir():
            raise ValueError(f"Task directory does not exist: {folder}")
        candidates = sorted(p for p in folder.iterdir() if Task.is_valid_dir(p))
        names = dataset.get("task_names") or ["*"]
        excluded = dataset.get("exclude_task_names") or []
        candidates = [p for p in candidates if any(fnmatch(p.name, n) for n in names)
                      and not any(fnmatch(p.name, n) for n in excluded)]
        if dataset.get("n_tasks") is not None:
            raise ValueError("Select explicit task_names instead of n_tasks for reproducible task selection")
        if not candidates:
            raise ValueError(f"No Harbor tasks selected in {folder}. Copy complete task folders into tasks/ first.")
        dataset["task_names"] = [p.name for p in candidates]
        for task in candidates:
            if task.is_symlink():
                raise ValueError(f"Task bundles must not be symlinked: {task}")
            if task.name in selected:
                raise ValueError(f"Duplicate task name: {task.name}")
            files = {}
            for path in sorted(task.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Task bundles must not contain symlinks: {path}")
                if path.is_file():
                    files[str(path.relative_to(task))] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "executable": bool(path.stat().st_mode & 0o111),
                    }
            selected[task.name] = {"path": str(task), "files": files}
    if not selected:
        raise ValueError("No local datasets configured")
    agents = config.get("agents", [])
    if len(agents) != 1:
        raise ValueError("Use one agent per job; run separate jobs to compare models")
    agent = agents[0]
    if agent.get("name") not in ("oracle", "nop") and not agent.get("model_name"):
        raise ValueError("Specify --model or agents[0].model_name")
    native = agent.get("kwargs", {}).get("config")
    if isinstance(native, str):
        agent["kwargs"]["config"] = resolve_path(root, native)
    environment = config.get("environment", {})
    if environment.get("extra_docker_compose"):
        environment["extra_docker_compose"] = [
            resolve_path(root, p) for p in environment["extra_docker_compose"]
        ]
    if config.get("n_attempts", 1) < 1 or config.get("n_concurrent_trials", 1) < 1:
        raise ValueError("attempts and concurrency must be positive")
    if agent.get("override_timeout_sec") is not None and agent["override_timeout_sec"] <= 0:
        raise ValueError("agent timeout must be positive")
    JobConfig.model_validate(config)
    manifest = {"job_name": job_name, "n_attempts": config.get("n_attempts", 1),
                "expected_trials": len(selected) * config.get("n_attempts", 1),
                "tasks": selected, "agent": agent,
                "created_at": datetime.now(timezone.utc).isoformat()}
    return config, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen local Harbor tasks from an independent evaluation directory")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Evaluation directory (default: current directory)")
    parser.add_argument("--config", default="configs/oracle.json", help="Harbor JSON config, relative to root")
    parser.add_argument("--model")
    parser.add_argument("--job-name")
    parser.add_argument("--task-name", action="append", help="Task name glob; repeat to select multiple groups")
    parser.add_argument("--attempts", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--agent-timeout-sec", type=float, help="Omit to honor each task's time limit")
    parser.add_argument("--dry-run", action="store_true", help="Validate and save configuration without running Harbor")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        config = json.loads(Path(resolve_path(root, args.config)).read_text())
        agents = config.get("agents", [])
        if len(agents) != 1:
            raise ValueError("Configure exactly one agent per job")
        agent = agents[0]
        if args.model:
            agent["model_name"] = args.model
        if args.reasoning_effort:
            agent.setdefault("kwargs", {})["reasoning_effort"] = args.reasoning_effort
        if args.agent_timeout_sec is not None:
            agent["override_timeout_sec"] = args.agent_timeout_sec
        for flag, key in [(args.attempts, "n_attempts"), (args.concurrency, "n_concurrent_trials")]:
            if flag is not None:
                config[key] = flag
        if args.task_name:
            for dataset in config.get("datasets", []):
                dataset["task_names"] = args.task_name
        job = args.job_name or "eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config, manifest = prepare(root, config, job)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    cache = root / ".cache" / job
    cache.mkdir(parents=True, exist_ok=False)
    config_path = cache / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest_path = cache / "task-snapshot.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Job: {job}; tasks: {len(manifest['tasks'])}; attempts/task: {manifest['n_attempts']}", flush=True)
    if args.dry_run:
        print(f"Validated configuration: {config_path}")
        return
    harbor = shutil.which("harbor")
    if not harbor:
        parser.error("Harbor is not on PATH; install dependencies and use uv run rsi-eval")
    try:
        subprocess.run([harbor, "run", "--config", str(config_path)], cwd=root, check=True)
    finally:
        job_dir = Path(config["jobs_dir"]) / job
        if job_dir.is_dir():
            shutil.copy2(manifest_path, job_dir / "task-snapshot.json")
    print(f"Results: {job_dir}")


if __name__ == "__main__":
    main()
