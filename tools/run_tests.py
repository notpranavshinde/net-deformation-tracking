#!/usr/bin/env python3
"""Run the repository's standalone assertion-based test scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIRECTORY = REPOSITORY_ROOT / "tools" / "tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        metavar="SUBSTRING",
        help="run only test script names containing this substring",
    )
    return parser.parse_args()


def print_summary(results: list[tuple[Path, bool, float]]) -> None:
    name_width = max(len("Test"), *(len(path.name) for path, _, _ in results))
    print("\nSummary")
    print(f"{'Test':<{name_width}}  {'Status':<6}  Duration")
    print(f"{'-' * name_width}  {'-' * 6}  {'-' * 8}")
    for path, passed, duration in results:
        status = "PASS" if passed else "FAIL"
        print(f"{path.name:<{name_width}}  {status:<6}  {duration:7.2f}s")


def main() -> int:
    args = parse_args()
    tests = sorted(TESTS_DIRECTORY.glob("test_*.py"))
    if args.filter:
        tests = [path for path in tests if args.filter in path.name]

    if not tests:
        print("No test scripts matched.", file=sys.stderr)
        return 2

    results: list[tuple[Path, bool, float]] = []
    for test_path in tests:
        print(f"\n==> {test_path.relative_to(REPOSITORY_ROOT)}")
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        duration = time.perf_counter() - started
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
        results.append((test_path, completed.returncode == 0, duration))

    print_summary(results)
    failures = sum(not passed for _, passed, _ in results)
    print(f"\n{len(results) - failures}/{len(results)} test scripts passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
