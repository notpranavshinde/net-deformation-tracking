#!/usr/bin/env python3
"""Plain assert checks for the Windows setup-only execution layer."""

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_pipeline_queue as queue


def test_path_and_command_formatting():
    path = r"C:\Users\Example User\raw videos\left.MP4"
    assert queue.normalize_pasted_path(path) == path
    assert queue.normalize_pasted_path(f'"{path}"') == path
    assert queue.normalize_pasted_path(f"& '{path}'") == path
    command = ["python", r"C:\Project Folder\run_pipeline_queue.py", "--setup-only"]
    assert queue.format_command(command, windows=True) == subprocess.list2cmdline(command)


def test_preflight_decisions():
    all_modules = lambda _name: object()
    no_executables = lambda _name: None
    issues, warnings = queue.collect_windows_setup_preflight_issues(
        python_version=(3, 13),
        module_finder=all_modules,
        ui_framework_probe=lambda: "WIN32",
        executable_finder=no_executables,
    )
    assert any("Python 3.11 is required" in issue for issue in issues)
    assert any("ffprobe" in warning for warning in warnings)

    missing_skimage = lambda name: None if name == "skimage" else object()
    issues, _ = queue.collect_windows_setup_preflight_issues(
        python_version=(3, 11),
        module_finder=missing_skimage,
        ui_framework_probe=lambda: "WIN32",
        executable_finder=lambda _name: "found",
    )
    assert any("scikit-image" in issue for issue in issues)

    issues, _ = queue.collect_windows_setup_preflight_issues(
        python_version=(3, 11),
        module_finder=all_modules,
        ui_framework_probe=lambda: "",
        executable_finder=lambda _name: "found",
    )
    assert any("no interactive UI backend" in issue for issue in issues)


def test_interrupted_stage_becomes_resumable():
    manifest = {
        "preprocessing": {
            "splitter": {"stage": {"status": "complete"}},
            "calibration": {"stages": {"sync": {"status": "complete"}}},
        },
        "jobs": [
            {
                "velocity": "trial 1",
                "stages": {
                    "setup": {"status": "running"},
                    "sam2": {"status": "pending"},
                    "triangulation": {"status": "pending"},
                },
            }
        ],
    }
    saved = []
    original_save_manifest = queue.save_manifest
    try:
        queue.save_manifest = lambda path, payload: saved.append((path, payload))
        labels = queue.mark_setup_only_interrupted(Path("queue_manifest.json"), manifest)
    finally:
        queue.save_manifest = original_save_manifest
    stage = manifest["jobs"][0]["stages"]["setup"]
    assert labels == ["marker setup for trial 1"]
    assert stage["status"] == "pending"
    assert stage["error"].startswith("Interrupted by user")
    assert "interrupted_at" in stage
    assert manifest["status"] == "incomplete"
    assert len(saved) == 1


def test_unbuffered_utf8_child_output(tmp_root):
    log_path = tmp_root / "stream.log"
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('PROMPT: READY ✓', end='', flush=False); "
            "time.sleep(1.5); "
            "print('DONE', flush=False)"
        ),
    ]
    result = {}

    def run_child():
        result["code"] = queue.run_logged(command, Path.cwd(), log_path, use_pty=False)

    thread = threading.Thread(target=run_child)
    thread.start()
    deadline = time.monotonic() + 1.0
    saw_ready_while_running = False
    while time.monotonic() < deadline:
        if log_path.is_file() and "PROMPT: READY ✓" in log_path.read_text(encoding="utf-8"):
            saw_ready_while_running = thread.is_alive()
            break
        time.sleep(0.05)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result["code"] == 0
    assert saw_ready_while_running
    logged = log_path.read_text(encoding="utf-8")
    assert "PROMPT: READY ✓" in logged and "DONE" in logged


def test_setup_rerun_command():
    args = SimpleNamespace(resume=None, sam2_scale=0.25, sam2_model_id=None)
    command = queue.setup_rerun_command(args, resume_path=Path(r"C:\Queue Folder\queue_a"))
    assert "conda" not in command
    assert "--setup-only" in command
    assert "--resume" in command
    assert "--sam2-scale 0.25" in command
    assert r'"C:\Queue Folder\queue_a"' in command


def test_windows_process_tree_fallback():
    if queue.os.name != "nt":
        return

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.signals = []
            self.waits = 0

        def poll(self):
            return None

        def send_signal(self, value):
            self.signals.append(value)

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0

        def kill(self):
            raise AssertionError("taskkill fallback should have completed the fake process")

    calls = []
    original_run = queue.subprocess.run
    try:
        queue.subprocess.run = lambda command, **_kwargs: calls.append(command)
        process = FakeProcess()
        queue.stop_process_tree(process, graceful_timeout=0)
    finally:
        queue.subprocess.run = original_run
    assert process.signals == [queue.signal.CTRL_BREAK_EVENT]
    assert calls == [["taskkill", "/PID", "12345", "/T", "/F"]]


def main():
    tmp_root = Path(__file__).resolve().parent / "_tmp" / "windows_setup"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True)
    try:
        test_path_and_command_formatting()
        test_preflight_decisions()
        test_interrupted_stage_becomes_resumable()
        test_unbuffered_utf8_child_output(tmp_root)
        test_setup_rerun_command()
        test_windows_process_tree_fallback()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("Windows setup-only tests passed")


if __name__ == "__main__":
    main()
