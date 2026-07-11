#!/usr/bin/env python3
"""Plain assert checks for remote pipeline configuration helpers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import remote_pipeline as remote


def main():
    captured = {}
    original_run_command = remote.run_command

    def fake_run_command(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return remote.CommandResult(0, "/remote/repo/machine_paths.json\n", "")

    config = {
        "host": "gpu-box",
        "remote_repo": "/remote/repo",
        "remote_python": "python3",
        "ssh_extra_args": ["-i", "gpu-key"],
    }
    try:
        remote.run_command = fake_run_command
        remote.configure_remote_video_roots(
            config,
            ["/data/videos", "/data/videos", "/mnt/archive"],
            verbose=True,
        )
    finally:
        remote.run_command = original_run_command

    assert captured["command"][:4] == ["ssh", "-i", "gpu-key", "gpu-box"]
    assert captured["command"][4].startswith("cd /remote/repo && python3 -c ")
    assert json.loads(captured["input_text"]) == {
        "video_roots": ["/data/videos", "/mnt/archive"]
    }

    args = remote.build_parser().parse_args(
        ["configure-paths", "/data/videos", "/mnt/archive", "--dry-run"]
    )
    assert args.func is remote.cmd_configure_paths
    assert args.video_roots == ["/data/videos", "/mnt/archive"]
    assert args.dry_run is True
    print("remote pipeline tests passed")


if __name__ == "__main__":
    main()
