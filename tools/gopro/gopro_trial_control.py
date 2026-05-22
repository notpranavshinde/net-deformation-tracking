r"""
Basic two-GoPro trial control over Open GoPro HTTP.

Paste-ready examples:

    python .\tools\gopro\gopro_trial_control.py --left-url http://LEFT_IP:8080 --right-url http://RIGHT_IP:8080 status

    python .\tools\gopro\gopro_trial_control.py --left-url http://LEFT_IP:8080 --right-url http://RIGHT_IP:8080 set-time

    python .\tools\gopro\gopro_trial_control.py --left-url http://LEFT_IP:8080 --right-url http://RIGHT_IP:8080 start

    python .\tools\gopro\gopro_trial_control.py --left-url http://LEFT_IP:8080 --right-url http://RIGHT_IP:8080 stop

This is useful for repeatable trial control, but API start commands are not a
sub-frame stereo sync guarantee. Keep audio/timecode validation in the pipeline.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def normalize_base_url(url: str) -> str:
    url = str(url).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def http_get(base_url: str, path: str, timeout: float = 10.0):
    url = normalize_base_url(base_url) + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def try_paths(base_url: str, paths):
    last_error = None
    for path in paths:
        try:
            return path, http_get(base_url, path)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"All candidate endpoints failed for {base_url}: {last_error}")


def set_time_paths(now: dt.datetime):
    # Open GoPro firmware generations have used slightly different spellings.
    # Try the documented/common shapes first, then fallbacks.
    date_us = now.strftime("%m_%d_%y")
    time_us = now.strftime("%H_%M_%S")
    date_iso = now.strftime("%Y_%m_%d")
    return [
        f"/gopro/camera/set_date_time?date={date_us}&time={time_us}",
        f"/gopro/camera/set_date_time?date={date_iso}&time={time_us}",
        f"/gp/gpControl/command/setup/date_time?p={now.strftime('%y%m%d%H%M%S')}",
    ]


def command_paths(command: str):
    if command == "start":
        return [
            "/gopro/camera/shutter/start",
            "/gp/gpControl/command/shutter?p=1",
        ]
    if command == "stop":
        return [
            "/gopro/camera/shutter/stop",
            "/gp/gpControl/command/shutter?p=0",
        ]
    if command == "status":
        return [
            "/gopro/camera/state",
            "/gp/gpControl/status",
        ]
    raise ValueError(command)


def run_for_camera(side: str, base_url: str, command: str):
    if command == "set-time":
        now = dt.datetime.now()
        path, result = try_paths(base_url, set_time_paths(now))
    else:
        path, result = try_paths(base_url, command_paths(command))
    return {"side": side, "url": normalize_base_url(base_url), "command": command, "endpoint": path, "result": result}


def parse_args():
    parser = argparse.ArgumentParser(description="Control two GoPros over Open GoPro HTTP.")
    parser.add_argument("--left-url", required=True)
    parser.add_argument("--right-url", required=True)
    parser.add_argument("command", choices=["status", "set-time", "start", "stop"])
    return parser.parse_args()


def main():
    args = parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(run_for_camera, "left", args.left_url, args.command),
            ex.submit(run_for_camera, "right", args.right_url, args.command),
        ]
        results = [f.result() for f in futures]
    for result in results:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
