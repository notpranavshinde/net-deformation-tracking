r"""
Download latest GoPro recordings from two cameras without removing SD cards.

Paste-ready example:

    python .\tools\gopro\gopro_fetch_recordings.py

Single-side downloads after running gopro-wifi:

    python .\tools\gopro\gopro_fetch_recordings.py --side left --out raw_trials\trial_001 --concat-to-sam2-input
    python .\tools\gopro\gopro_fetch_recordings.py --side right --out raw_trials\trial_001 --concat-to-sam2-input

The script assumes the practical HERO11 workflow:

1. Connect the laptop to the LEFT GoPro Wi-Fi network, press Enter.
2. It downloads the newest LEFT MP4 recording group.
3. Connect the laptop to the RIGHT GoPro Wi-Fi network, press Enter.
4. It downloads the newest RIGHT MP4 recording group.

The GoPro Wi-Fi AP URL is normally http://10.5.5.9:8080.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


VIDEO_EXTS = {".mp4", ".lrv"}


def normalize_base_url(url: str) -> str:
    url = str(url).strip().rstrip("/")
    if not url:
        raise ValueError("Empty GoPro URL")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def http_json(url: str, timeout: float = 15.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_media(base_url: str):
    return http_json(f"{base_url}/gopro/media/list")


def flatten_media(media_payload):
    items = []
    media = media_payload.get("media", [])
    for folder in media:
        directory = folder.get("d") or folder.get("directory")
        for f in folder.get("fs", []):
            name = f.get("n") or f.get("name") or f.get("filename")
            if not directory or not name:
                continue
            ext = Path(name).suffix.lower()
            if ext not in VIDEO_EXTS:
                continue
            try:
                size = int(f.get("s", f.get("size", 0)) or 0)
            except ValueError:
                size = 0
            timestamp = f.get("cre") or f.get("mod") or f.get("created") or f.get("modified") or ""
            items.append({
                "directory": str(directory),
                "name": str(name),
                "ext": ext,
                "size": size,
                "timestamp": str(timestamp),
                "raw": f,
            })
    return items


def clip_group_key(name: str):
    stem = Path(name).stem.upper()
    # GoPro chapters commonly look like GX010123.MP4, GX020123.MP4.
    m = re.match(r"^([A-Z]{2})(\d{2})(\d{4})$", stem)
    if m:
        prefix, _chapter, clip = m.groups()
        return f"{prefix}{clip}"
    # Older naming commonly looks like GOPR0123 / GP010123.
    m = re.match(r"^([A-Z]{2,4})(\d{4})$", stem)
    if m:
        prefix, clip = m.groups()
        return f"{prefix[:2]}{clip}"
    return stem


def chapter_sort_key(item):
    stem = Path(item["name"]).stem.upper()
    m = re.match(r"^[A-Z]{2}(\d{2})(\d{4})$", stem)
    if m:
        return int(m.group(1)), item["name"]
    return 0, item["name"]


def group_mp4_recordings(items):
    groups = {}
    for item in items:
        if item["ext"] != ".mp4":
            continue
        key = clip_group_key(item["name"])
        groups.setdefault(key, []).append(item)
    out = []
    for key, files in groups.items():
        files = sorted(files, key=chapter_sort_key)
        newest_timestamp = max((f["timestamp"] for f in files), default="")
        newest_name = max((f["name"] for f in files), default="")
        out.append({
            "key": key,
            "files": files,
            "timestamp": newest_timestamp,
            "sort_name": newest_name,
            "total_size": sum(int(f["size"]) for f in files),
        })
    return sorted(out, key=lambda g: (g["timestamp"], g["sort_name"]), reverse=True)


def download_file(base_url: str, item, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    url_path = "/".join([
        "videos",
        "DCIM",
        urllib.parse.quote(item["directory"]),
        urllib.parse.quote(item["name"]),
    ])
    url = f"{base_url}/{url_path}"
    dest = out_dir / item["name"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[GET] {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    tmp.replace(dest)
    return dest


def concat_chapters(files, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(files) == 1:
        shutil.copy2(files[0], output_path)
        return "copy"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Multiple GoPro chapters found but ffmpeg is not on PATH for concat.")
    list_path = output_path.with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            safe = str(Path(p).resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)],
        check=True,
    )
    return "ffmpeg_concat"


def prompt_if_missing(value, prompt):
    if value:
        return value
    return input(prompt).strip()


def fetch_side(side: str, base_url: str, out_root: Path, group_index: int):
    base_url = normalize_base_url(base_url)
    payload = list_media(base_url)
    groups = group_mp4_recordings(flatten_media(payload))
    if not groups:
        raise RuntimeError(f"No MP4 media found on {side} camera at {base_url}")
    if group_index < 0 or group_index >= len(groups):
        raise RuntimeError(f"--group-index {group_index} out of range; {side} has {len(groups)} MP4 groups")

    group = groups[group_index]
    side_dir = out_root / side / group["key"]
    downloaded = [download_file(base_url, item, side_dir) for item in group["files"]]
    manifest = {
        "side": side,
        "base_url": base_url,
        "downloaded_at_unix": time.time(),
        "group_index": group_index,
        "group": group,
        "local_files": [str(p) for p in downloaded],
    }
    (side_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return side, side_dir, downloaded, manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Download newest MP4 recording group from two GoPros.")
    parser.add_argument("--url", default="http://10.5.5.9:8080", help="GoPro Wi-Fi AP base URL")
    parser.add_argument("--out", default=str(Path("raw_trials") / time.strftime("trial_%Y%m%d_%H%M%S")))
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--group-index", type=int, default=0, help="0=newest MP4 group, 1=previous, etc.")
    parser.add_argument("--concat-to-sam2-input", action="store_true", help="Copy/concat downloaded chapters to sam2/sam2/in/left.mp4 and right.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = normalize_base_url(args.url)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.side == "both":
        print("")
        print("[STEP 1] Turn on LEFT GoPro wireless connections and connect this computer to the LEFT GoPro Wi-Fi.")
        print(f"         Expected camera URL: {base_url}")
        input("         Press Enter when LEFT is connected...")
        left_result = fetch_side("left", base_url, out_root, args.group_index)

        print("")
        print("[STEP 2] Disconnect from LEFT Wi-Fi, then connect this computer to the RIGHT GoPro Wi-Fi.")
        print(f"         Expected camera URL: {base_url}")
        input("         Press Enter when RIGHT is connected...")
        right_result = fetch_side("right", base_url, out_root, args.group_index)
        results = [left_result, right_result]
    else:
        print(f"[INFO] Downloading {args.side.upper()} from {base_url}")
        results = [fetch_side(args.side, base_url, out_root, args.group_index)]

    summary = {side: {"dir": str(side_dir), "files": [str(p) for p in files], "manifest": manifest} for side, side_dir, files, manifest in results}
    (out_root / "download_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] Download summary: {out_root / 'download_summary.json'}")

    if args.concat_to_sam2_input:
        sam2_in = Path("sam2") / "sam2" / "in"
        sam2_in.mkdir(parents=True, exist_ok=True)
        for side, _side_dir, files, _manifest in results:
            mode = concat_chapters(files, sam2_in / f"{side}.mp4")
            print(f"[OK] {side}: wrote {sam2_in / f'{side}.mp4'} ({mode})")


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
