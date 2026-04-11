#!/usr/bin/env python3
"""
Live timelapse preview stitcher.
Watches timelapse_frames/ and rebuilds the video every REBUILD_INTERVAL_SEC.
Run this alongside timelapse_capture.py to get a rolling preview.

Requirements:
    ffmpeg on PATH (no pip deps needed)

Usage:
    python3 timelapse_preview.py
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

FRAMES_DIR           = Path("timelapse_frames")
PREVIEW_VIDEO        = "timelapse_preview.mp4"
REBUILD_INTERVAL_SEC = 300      # how often to rebuild the preview
FFMPEG_FPS           = 24       # playback framerate
MIN_FRAMES           = 5        # don't bother building until we have at least this many

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_frames(frames_dir: Path) -> int:
    return len(sorted(frames_dir.glob("frame_*.jpg")))


def build_preview(frames_dir: Path, output: str, fps: int) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # makes it seekable before fully written
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Live preview stitcher started -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Watching : {FRAMES_DIR}/")
    print(f"  Output   : {PREVIEW_VIDEO}")
    print(f"  Rebuild  : every {REBUILD_INTERVAL_SEC}s (min {MIN_FRAMES} frames)")
    print("\nCtrl+C to stop.\n")

    last_frame_count = 0

    try:
        while True:
            if not FRAMES_DIR.exists():
                print("  Waiting for frames directory to appear ...")
                time.sleep(REBUILD_INTERVAL_SEC)
                continue

            n = count_frames(FRAMES_DIR)

            if n < MIN_FRAMES:
                print(f"  Only {n} frames so far, waiting for {MIN_FRAMES} ...")
                time.sleep(REBUILD_INTERVAL_SEC)
                continue

            if n == last_frame_count:
                print(f"  No new frames since last build ({n} total), skipping.")
                time.sleep(REBUILD_INTERVAL_SEC)
                continue

            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Building preview from {n} frames ... ", end="", flush=True)
            ok = build_preview(FRAMES_DIR, PREVIEW_VIDEO, FFMPEG_FPS)

            if ok:
                duration = n / FFMPEG_FPS
                print(f"done. ({duration:.1f}s @ {FFMPEG_FPS}fps)  ->  {PREVIEW_VIDEO}")
                last_frame_count = n
            else:
                print("ffmpeg error (frames may be mid-write, will retry).")

            time.sleep(REBUILD_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nStopped.")

        # Final build on exit
        n = count_frames(FRAMES_DIR)
        if n >= MIN_FRAMES:
            print(f"Building final preview from {n} frames ... ", end="", flush=True)
            ok = build_preview(FRAMES_DIR, PREVIEW_VIDEO, FFMPEG_FPS)
            print("done." if ok else "failed.")


if __name__ == "__main__":
    main()
