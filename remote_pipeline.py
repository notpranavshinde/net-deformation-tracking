#!/usr/bin/env python3
"""Move portable pipeline queues between setup and GPU processing machines."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
QUEUE_ROOT = REPO_ROOT / "work" / "pipeline_queue"
CONFIG_PATH = REPO_ROOT / "remote_config.json"
RUNNER = REPO_ROOT / "run_pipeline_queue.py"

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional dependency
    Console = None
    Table = None

console = Console() if Console else None


@dataclass
class CommandResult:
    code: int
    stdout: str = ""
    stderr: str = ""


class RemotePipelineError(RuntimeError):
    pass


def print_line(message: str = "") -> None:
    if console:
        console.print(message)
    else:
        print(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def remote_join(*parts: str) -> str:
    cleaned = []
    for idx, part in enumerate(parts):
        text = str(part).replace("\\", "/")
        if idx == 0:
            text = text.rstrip("/")
        else:
            text = text.strip("/")
        if text:
            cleaned.append(text)
    if not cleaned:
        return "."
    return "/".join(cleaned)


def local_repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def queue_manifest_path(queue_arg: str | None, allow_missing_dry_run: bool = False) -> Path:
    if queue_arg:
        path = Path(queue_arg).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir() or (allow_missing_dry_run and path.suffix != ".json"):
            path = path / "queue_manifest.json"
        if path.is_file():
            return path
        if allow_missing_dry_run:
            return path
        raise RemotePipelineError(f"Queue manifest not found: {path}")

    candidates = sorted(
        QUEUE_ROOT.glob("*/queue_manifest.json"),
        key=lambda item: item.parent.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RemotePipelineError(
            "No queue found under work/pipeline_queue. Pass --queue with a queue directory "
            "or queue_manifest.json."
        )
    return candidates[0]


def queue_id_from_manifest(manifest_path: Path) -> str:
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            if manifest.get("queue_id"):
                return str(manifest["queue_id"])
        except Exception:
            pass
    return manifest_path.parent.name


def load_config(required: bool = True) -> dict:
    if not CONFIG_PATH.is_file():
        if required:
            raise RemotePipelineError(
                "remote_config.json is missing. Run `python remote_pipeline.py init` first."
            )
        return {}
    try:
        payload = read_json(CONFIG_PATH)
    except json.JSONDecodeError as exc:
        raise RemotePipelineError(f"remote_config.json is not valid JSON: {exc}") from exc
    for key in ("host", "remote_repo", "remote_python"):
        if required and not payload.get(key):
            raise RemotePipelineError(f"remote_config.json is missing `{key}`.")
    payload.setdefault("ssh_extra_args", [])
    if not isinstance(payload["ssh_extra_args"], list):
        raise RemotePipelineError("remote_config.json `ssh_extra_args` must be a JSON list.")
    return payload


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(shlex.quote(x) for x in command)


def run_command(
    command: list[str],
    *,
    verbose: bool = False,
    dry_run: bool = False,
    input_text: str | None = None,
    timeout: int | None = None,
    stream: bool = False,
) -> CommandResult:
    if verbose or dry_run:
        print_line("$ " + display_command(command))
    if dry_run:
        return CommandResult(0, "", "")
    try:
        if stream:
            code = subprocess.call(command)
            return CommandResult(code, "", "")
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        tool = command[0] if command else "command"
        raise RemotePipelineError(f"`{tool}` was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, exc.stdout or "", exc.stderr or f"Timed out after {timeout}s")


def ssh_command(config: dict, remote_shell: str, *, tty: bool = False) -> list[str]:
    command = ["ssh", *config.get("ssh_extra_args", [])]
    if tty:
        command.append("-t")
    command.extend([config["host"], remote_shell])
    return command


def scp_command(config: dict, source: str, dest: str) -> list[str]:
    return ["scp", *config.get("ssh_extra_args", []), source, dest]


def remote_shell(config: dict, command: str) -> str:
    return f"cd {shell_quote(config['remote_repo'])} && {command}"


def translate_ssh_failure(result: CommandResult, config: dict) -> str:
    text = (result.stderr or result.stdout or "").strip()
    low = text.lower()
    if "could not resolve hostname" in low:
        return f"Could not resolve SSH host `{config['host']}`. Check the host or ~/.ssh/config alias."
    if "connection timed out" in low or "operation timed out" in low:
        return f"SSH connection to `{config['host']}` timed out."
    if "permission denied" in low:
        return f"SSH authentication failed for `{config['host']}`."
    if "no such file or directory" in low and "run_pipeline_queue.py" in low:
        return "Remote repo is missing run_pipeline_queue.py. Check `remote_repo` in remote_config.json."
    return text or f"Command failed with exit code {result.code}."


def local_check_queue(manifest_path: Path, verbose: bool) -> CommandResult:
    command = [sys.executable, str(RUNNER), "--resume", str(manifest_path), "--check-only"]
    return run_command(command, verbose=verbose)


def check_output_process_ready(output: str) -> bool:
    for line in output.splitlines():
        if "process-ready" not in line:
            continue
        fields = line.strip().split()
        return bool(fields and fields[-1] == "complete")
    return False


def remote_check_queue(config: dict, queue_id: str, verbose: bool, dry_run: bool = False) -> CommandResult:
    queue_rel = remote_join("work/pipeline_queue", queue_id)
    py = config["remote_python"]
    command = f"{py} run_pipeline_queue.py --resume {shell_quote(queue_rel)} --check-only"
    return run_command(ssh_command(config, remote_shell(config, command)), verbose=verbose, dry_run=dry_run)


def excluded_from_push(rel: str) -> bool:
    parts = set(Path(rel).parts)
    if "objectwise_frames" in parts:
        return True
    if "sam2" in parts:
        return True
    return False


def excluded_from_pull(rel: str) -> bool:
    return "objectwise_frames" in set(Path(rel).parts)


def make_queue_archive(queue_dir: Path, queue_id: str, archive_path: Path, dry_run: bool = False) -> list[str]:
    included: list[str] = []
    for path in sorted(queue_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(queue_dir).as_posix()
        if excluded_from_push(rel):
            continue
        included.append(rel)
    if dry_run:
        return included
    with tarfile.open(archive_path, "w:gz") as archive:
        for rel in included:
            archive.add(queue_dir / rel, arcname=f"{queue_id}/{rel}")
    return included


def create_remote_parent(config: dict, remote_target: str) -> str:
    parent = remote_join(remote_target, "..")
    return f"mkdir -p {shell_quote(parent)}"


def extract_archive(archive: tarfile.TarFile, target: Path) -> None:
    try:
        archive.extractall(target, filter="data")
    except TypeError:
        archive.extractall(target)


def transfer_queue_to_remote(
    config: dict,
    queue_dir: Path,
    queue_id: str,
    verbose: bool,
    dry_run: bool,
) -> None:
    remote_parent = remote_join(config["remote_repo"], "work/pipeline_queue")
    remote_archive = f"/tmp/{queue_id}.{int(time.time())}.tar.gz"
    temp_dir = Path(tempfile.mkdtemp(prefix=".remote-pipeline-push-", dir=REPO_ROOT))
    try:
        archive_path = temp_dir / f"{queue_id}.tar.gz"
        included = make_queue_archive(queue_dir, queue_id, archive_path, dry_run=dry_run)
        print_line(f"[PUSH] Queue files selected: {len(included)}")
        for rel in included[:30]:
            print_line(f"  {rel}")
        if len(included) > 30:
            print_line(f"  ... {len(included) - 30} more")
        if dry_run:
            print_line(f"[DRY-RUN] Would create archive: {archive_path}")
        result = run_command(
            ssh_command(config, f"mkdir -p {shell_quote(remote_parent)}"),
            verbose=verbose,
            dry_run=dry_run,
        )
        if result.code != 0:
            raise RemotePipelineError(translate_ssh_failure(result, config))
        result = run_command(
            scp_command(config, str(archive_path), f"{config['host']}:{remote_archive}"),
            verbose=verbose,
            dry_run=dry_run,
        )
        if result.code != 0:
            raise RemotePipelineError(translate_ssh_failure(result, config))
        extract = (
            f"mkdir -p {shell_quote(remote_parent)} && "
            f"tar -xzf {shell_quote(remote_archive)} -C {shell_quote(remote_parent)} && "
            f"rm -f {shell_quote(remote_archive)}"
        )
        result = run_command(ssh_command(config, extract), verbose=verbose, dry_run=dry_run)
        if result.code != 0:
            raise RemotePipelineError(translate_ssh_failure(result, config))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def remote_tmux_present(config: dict, verbose: bool, dry_run: bool = False) -> bool:
    result = run_command(
        ssh_command(config, "command -v tmux >/dev/null 2>&1"),
        verbose=verbose,
        dry_run=dry_run,
    )
    return result.code == 0


def tmux_session_alive(config: dict, session: str, verbose: bool, dry_run: bool = False) -> bool:
    if dry_run:
        return False
    result = run_command(
        ssh_command(config, f"tmux has-session -t {shell_quote(session)} >/dev/null 2>&1"),
        verbose=verbose,
        dry_run=dry_run,
    )
    return result.code == 0


def process_alive(config: dict, queue_id: str, verbose: bool, dry_run: bool = False) -> bool:
    if dry_run:
        return False
    pattern = f"run_pipeline_queue.py --resume work/pipeline_queue/{queue_id} --process-only"
    result = run_command(
        ssh_command(config, f"pgrep -af {shell_quote(pattern)} >/dev/null 2>&1"),
        verbose=verbose,
        dry_run=dry_run,
    )
    return result.code == 0


def run_doctor(config: dict, verbose: bool, dry_run: bool = False) -> int:
    checks: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, detail: str = "") -> None:
        checks.append((label, ok, detail.strip()))

    result = run_command(ssh_command(config, "true"), verbose=verbose, dry_run=dry_run)
    add("SSH connectivity", result.code == 0, "" if result.code == 0 else translate_ssh_failure(result, config))

    repo_cmd = (
        f"test -d {shell_quote(config['remote_repo'])} && "
        f"test -f {shell_quote(remote_join(config['remote_repo'], 'run_pipeline_queue.py'))}"
    )
    result = run_command(ssh_command(config, repo_cmd), verbose=verbose, dry_run=dry_run)
    add("Remote repo + runner", result.code == 0, "" if result.code == 0 else "Check `remote_repo`.")

    import_cmd = remote_shell(config, f"{config['remote_python']} -c {shell_quote('import torch, cv2')}")
    result = run_command(ssh_command(config, import_cmd), verbose=verbose, dry_run=dry_run)
    add("Remote python imports torch/cv2", result.code == 0, result.stderr or result.stdout)

    gpu_cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null"
    result = run_command(ssh_command(config, gpu_cmd), verbose=verbose, dry_run=dry_run)
    add("nvidia-smi GPU summary", result.code == 0, result.stdout or result.stderr)

    result = run_command(ssh_command(config, "command -v tmux >/dev/null 2>&1"), verbose=verbose, dry_run=dry_run)
    add("tmux present", result.code == 0, "nohup fallback will be used" if result.code != 0 else "")

    print_checks(checks)
    return 0 if all(ok for label, ok, _ in checks if label != "tmux present") else 1


def print_checks(checks: list[tuple[str, bool, str]]) -> None:
    if Table and console:
        table = Table(title="Remote Pipeline Doctor")
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Detail")
        for label, ok, detail in checks:
            table.add_row(label, "PASS" if ok else "FAIL", detail)
        console.print(table)
        return
    for label, ok, detail in checks:
        suffix = f" - {detail}" if detail else ""
        print_line(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}")


def status_rank(status: str | None) -> int:
    return {"pending": 0, "running": 1, "failed": 1, "complete": 2}.get(str(status or ""), 0)


def manifest_progress_score(manifest: dict) -> tuple[int, int]:
    score = 0
    total = 0
    calibration = manifest.get("preprocessing", {}).get("calibration", {})
    for stage in calibration.get("stages", {}).values():
        total += 1
        score += status_rank(stage.get("status"))
    for job in manifest.get("jobs", []):
        for stage in job.get("stages", {}).values():
            total += 1
            score += status_rank(stage.get("status"))
    return score, total


def remote_manifest_superset(local_manifest: dict, remote_manifest: dict) -> bool:
    if remote_manifest.get("queue_id") != local_manifest.get("queue_id"):
        return False
    if manifest_progress_score(remote_manifest)[0] < manifest_progress_score(local_manifest)[0]:
        return False
    local_jobs = {int(job.get("index", -1)): job for job in local_manifest.get("jobs", [])}
    remote_jobs = {int(job.get("index", -1)): job for job in remote_manifest.get("jobs", [])}
    for index, local_job in local_jobs.items():
        remote_job = remote_jobs.get(index)
        if not remote_job:
            return False
        for name, local_stage in local_job.get("stages", {}).items():
            if local_stage.get("status") == "complete":
                if remote_job.get("stages", {}).get(name, {}).get("status") != "complete":
                    return False
    return True


def copy_tree_contents(src: Path, dest: Path, exclude_manifest: bool = False) -> None:
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src).as_posix()
        if excluded_from_pull(rel):
            continue
        if exclude_manifest and rel == "queue_manifest.json":
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def pull_from_remote(config: dict, manifest_path: Path, queue_id: str, verbose: bool, dry_run: bool) -> None:
    manifest = read_json(manifest_path) if manifest_path.is_file() else {"queue_id": queue_id, "jobs": []}
    result_paths = []
    for job in manifest.get("jobs", []):
        slug = job.get("velocity_slug") or Path(str(job.get("result_dir", ""))).name
        if slug:
            result_paths.append(remote_join("triangulation/results", slug))
    queue_rel = remote_join("work/pipeline_queue", queue_id)
    archive_name = f"/tmp/{queue_id}.pull.{int(time.time())}.tar.gz"
    existing_result_paths: list[str] = []
    if result_paths:
        probe = "for p in " + " ".join(shell_quote(item) for item in result_paths) + '; do [ -e "$p" ] && printf "%s\\n" "$p"; done'
        result = run_command(
            ssh_command(config, remote_shell(config, probe)),
            verbose=verbose,
            dry_run=dry_run,
        )
        if result.code != 0:
            raise RemotePipelineError(translate_ssh_failure(result, config))
        existing = set(result.stdout.splitlines())
        existing_result_paths = [item for item in result_paths if item in existing]
        for item in result_paths:
            if item not in existing:
                print_line(f"[PULL] Skipping missing remote result dir: {item}")
    archive_paths = [queue_rel, *existing_result_paths]
    print_line("[PULL] Remote paths selected:")
    for item in archive_paths:
        print_line(f"  {item}")
    tar_cmd = (
        "tar -czf "
        + shell_quote(archive_name)
        + " --exclude='*/objectwise_frames/*' --exclude='*/objectwise_frames' "
        + " ".join(shell_quote(item) for item in archive_paths)
    )
    result = run_command(
        ssh_command(config, remote_shell(config, tar_cmd)),
        verbose=verbose,
        dry_run=dry_run,
    )
    if result.code != 0:
        raise RemotePipelineError(translate_ssh_failure(result, config))
    temp_dir = Path(tempfile.mkdtemp(prefix=".remote-pipeline-pull-", dir=REPO_ROOT))
    try:
        local_archive = temp_dir / "remote_pull.tar.gz"
        result = run_command(
            scp_command(config, f"{config['host']}:{archive_name}", str(local_archive)),
            verbose=verbose,
            dry_run=dry_run,
        )
        if result.code != 0:
            raise RemotePipelineError(translate_ssh_failure(result, config))
        cleanup = run_command(ssh_command(config, f"rm -f {shell_quote(archive_name)}"), verbose=verbose, dry_run=dry_run)
        if cleanup.code != 0 and verbose:
            print_line(cleanup.stderr)
        if dry_run:
            return
        stage = Path(temp_dir) / "stage"
        stage.mkdir()
        with tarfile.open(local_archive, "r:gz") as archive:
            extract_archive(archive, stage)
        for result_rel in result_paths:
            src = stage / Path(result_rel)
            if src.exists():
                copy_tree_contents(src, REPO_ROOT / result_rel)
                print_line(f"[PULL] Updated {result_rel}")
        remote_queue = stage / Path(queue_rel)
        local_queue = QUEUE_ROOT / queue_id
        if remote_queue.exists():
            copy_tree_contents(remote_queue, local_queue, exclude_manifest=True)
            remote_manifest_path = remote_queue / "queue_manifest.json"
            if remote_manifest_path.is_file():
                remote_manifest = read_json(remote_manifest_path)
                local_queue.mkdir(parents=True, exist_ok=True)
                saved_remote = local_queue / "queue_manifest.remote.json"
                shutil.copy2(remote_manifest_path, saved_remote)
                print_line(f"[PULL] Saved remote manifest as {local_repo_rel(saved_remote)}")
                if manifest_path.is_file():
                    local_manifest = read_json(manifest_path)
                    if remote_manifest_superset(local_manifest, remote_manifest):
                        backup = local_queue / "queue_manifest.local.bak.json"
                        shutil.copy2(manifest_path, backup)
                        shutil.copy2(remote_manifest_path, manifest_path)
                        print_line(
                            "[PULL] Remote manifest is a progress superset; replaced local "
                            f"manifest after backup to {local_repo_rel(backup)}"
                        )
                    else:
                        print_line("[PULL] Local manifest kept; remote progress was not a clear superset.")
                else:
                    shutil.copy2(remote_manifest_path, local_queue / "queue_manifest.json")
                    print_line("[PULL] No local manifest existed; copied remote manifest into place.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def cmd_init(args: argparse.Namespace) -> int:
    existing = load_config(required=False)
    host = input(f"host [{existing.get('host', '')}]: ").strip() or existing.get("host", "")
    remote_repo = input(f"remote_repo [{existing.get('remote_repo', '')}]: ").strip() or existing.get("remote_repo", "")
    default_python = existing.get("remote_python", "conda run --no-capture-output -n sam2py311 python")
    remote_python = input(f"remote_python [{default_python}]: ").strip() or default_python
    raw_extra = input("ssh_extra_args JSON list [[]]: ").strip()
    if raw_extra:
        try:
            ssh_extra_args = json.loads(raw_extra)
            if not isinstance(ssh_extra_args, list):
                raise ValueError
        except Exception as exc:
            raise RemotePipelineError("ssh_extra_args must be a JSON list, e.g. [\"-i\", \"key.pem\"].") from exc
    else:
        ssh_extra_args = existing.get("ssh_extra_args", [])
    if not host or not remote_repo:
        raise RemotePipelineError("host and remote_repo are required.")
    config = {
        "host": host,
        "remote_repo": remote_repo.replace("\\", "/").rstrip("/"),
        "remote_python": remote_python,
        "ssh_extra_args": ssh_extra_args,
    }
    write_json(CONFIG_PATH, config)
    print_line(f"[INIT] Wrote {CONFIG_PATH.name}")
    return run_doctor(config, args.verbose)


def cmd_doctor(args: argparse.Namespace) -> int:
    return run_doctor(load_config(), args.verbose)


def cmd_push(args: argparse.Namespace) -> int:
    config = load_config()
    manifest_path = queue_manifest_path(args.queue, allow_missing_dry_run=args.dry_run)
    queue_id = queue_id_from_manifest(manifest_path)
    queue_dir = manifest_path.parent
    if not args.dry_run:
        result = local_check_queue(manifest_path, args.verbose)
        if result.stdout:
            print_line(result.stdout.rstrip())
        if result.stderr:
            print_line(result.stderr.rstrip())
        if result.code != 0 or not check_output_process_ready(result.stdout):
            raise RemotePipelineError(
                "Local queue did not pass setup-phase validation. Run "
                f"`python run_pipeline_queue.py --resume {manifest_path} --check-only` and fix the listed items."
            )
    else:
        print_line(f"[DRY-RUN] Would validate local queue with --check-only: {manifest_path}")
    if not args.dry_run:
        session = f"pipeline_{queue_id}"
        if tmux_session_alive(config, session, args.verbose) or process_alive(config, queue_id, args.verbose):
            raise RemotePipelineError(
                f"Remote processing for queue `{queue_id}` appears to be running. "
                "Stop it or wait for it to finish before pushing over the queue manifest."
            )
    transfer_queue_to_remote(config, queue_dir, queue_id, args.verbose, args.dry_run)
    result = remote_check_queue(config, queue_id, args.verbose, dry_run=args.dry_run)
    if result.stdout:
        print_line(result.stdout.rstrip())
    if result.stderr:
        print_line(result.stderr.rstrip())
    if result.code != 0:
        raise RemotePipelineError(
            "Remote --check-only failed. If video resolution failed, create or fix "
            "machine_paths.json at the remote repo root."
        )
    print_line(f"[PUSH] Remote queue ready: work/pipeline_queue/{queue_id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    manifest_path = queue_manifest_path(args.queue, allow_missing_dry_run=args.dry_run)
    queue_id = queue_id_from_manifest(manifest_path)
    session = f"pipeline_{queue_id}"
    log_rel = remote_join("work/pipeline_queue", queue_id, "remote_run.log")
    process_cmd = (
        f"{config['remote_python']} run_pipeline_queue.py --resume "
        f"{shell_quote(remote_join('work/pipeline_queue', queue_id))} --process-only"
    )
    attach = f"ssh -t {config['host']} tmux attach -t {session}"
    if remote_tmux_present(config, args.verbose, args.dry_run):
        if tmux_session_alive(config, session, args.verbose, args.dry_run):
            raise RemotePipelineError(f"tmux session `{session}` is already alive. Attach with: {attach}")
        command = (
            f"tmux new-session -d -s {shell_quote(session)} "
            f"{shell_quote(process_cmd + ' 2>&1 | tee -a ' + log_rel)}"
        )
    else:
        if process_alive(config, queue_id, args.verbose, args.dry_run):
            raise RemotePipelineError(f"A previous process for queue `{queue_id}` still appears to be alive.")
        command = f"nohup setsid sh -lc {shell_quote(process_cmd + ' >> ' + log_rel + ' 2>&1')} >/dev/null 2>&1 &"
    result = run_command(ssh_command(config, remote_shell(config, command)), verbose=args.verbose, dry_run=args.dry_run)
    if result.code != 0:
        raise RemotePipelineError(translate_ssh_failure(result, config))
    print_line(f"[RUN] Started remote processing for {queue_id}.")
    print_line(f"[RUN] Attach: {attach}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config()
    manifest_path = queue_manifest_path(args.queue, allow_missing_dry_run=args.dry_run)
    queue_id = queue_id_from_manifest(manifest_path)
    result = remote_check_queue(config, queue_id, args.verbose)
    if result.stdout:
        print_line(result.stdout.rstrip())
    if result.stderr:
        print_line(result.stderr.rstrip())
    tmux_alive = tmux_session_alive(config, f"pipeline_{queue_id}", args.verbose)
    proc_alive = process_alive(config, queue_id, args.verbose)
    print_line(f"[STATUS] tmux session alive: {'yes' if tmux_alive else 'no'}")
    print_line(f"[STATUS] process alive: {'yes' if proc_alive else 'no'}")
    log_rel = remote_join("work/pipeline_queue", queue_id, "remote_run.log")
    tail_cmd = f"test -f {shell_quote(log_rel)} && tail -n 15 {shell_quote(log_rel)} || true"
    tail = run_command(ssh_command(config, remote_shell(config, tail_cmd)), verbose=args.verbose)
    print_line("[STATUS] Last remote_run.log lines:")
    print_line((tail.stdout or "").rstrip() or "  <no log yet>")
    return 0 if result.code == 0 else result.code


def cmd_logs(args: argparse.Namespace) -> int:
    config = load_config()
    manifest_path = queue_manifest_path(args.queue)
    queue_id = queue_id_from_manifest(manifest_path)
    log_rel = remote_join("work/pipeline_queue", queue_id, "remote_run.log")
    tail = f"tail {'-f' if args.follow else '-n 80'} {shell_quote(log_rel)}"
    result = run_command(
        ssh_command(config, remote_shell(config, tail), tty=args.follow),
        verbose=args.verbose,
        stream=args.follow,
    )
    if not args.follow:
        if result.stdout:
            print_line(result.stdout.rstrip())
        if result.stderr:
            print_line(result.stderr.rstrip())
    return result.code


def cmd_pull(args: argparse.Namespace) -> int:
    config = load_config()
    manifest_path = queue_manifest_path(args.queue, allow_missing_dry_run=args.dry_run)
    queue_id = queue_id_from_manifest(manifest_path)
    pull_from_remote(config, manifest_path, queue_id, args.verbose, args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSH transfer/launch/monitor CLI for portable pipeline queues.")
    parser.add_argument("--verbose", action="store_true", help="Print ssh/scp/tar commands before running them.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create remote_config.json and run doctor.")
    init.set_defaults(func=cmd_init)

    doctor = sub.add_parser("doctor", help="Check SSH, remote repo, Python env, GPU, and tmux.")
    doctor.set_defaults(func=cmd_doctor)

    def add_queue_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--queue", help="Queue directory or queue_manifest.json. Defaults to newest work/pipeline_queue queue.")

    push = sub.add_parser("push", help="Validate and transfer a setup queue to the remote.")
    add_queue_arg(push)
    push.add_argument("--dry-run", action="store_true", help="Print transfer commands and selected files without mutating remote state.")
    push.set_defaults(func=cmd_push)

    run = sub.add_parser("run", help="Start remote processing detached.")
    add_queue_arg(run)
    run.add_argument("--dry-run", action="store_true", help="Print launch commands without starting remote processing.")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Run remote --check-only and show process/log status.")
    add_queue_arg(status)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="Tail remote_run.log.")
    add_queue_arg(logs)
    logs.add_argument("-f", "--follow", action="store_true", help="Follow the remote log with tail -f.")
    logs.set_defaults(func=cmd_logs)

    pull = sub.add_parser("pull", help="Pull results, queue logs, SAM2 tracks, and remote manifest back locally.")
    add_queue_arg(pull)
    pull.add_argument("--dry-run", action="store_true", help="Print pull commands and selected paths without changing local files.")
    pull.set_defaults(func=cmd_pull)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RemotePipelineError as exc:
        print_line(f"[REMOTE] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
