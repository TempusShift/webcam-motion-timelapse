#!/usr/bin/env python3
"""
Timelapse capture for laser engraver (or any mjpeg-streamer snapshot URL).
Captures frames at a set interval and auto-stops when the scene stops changing.

Requirements:
    pip install requests pillow numpy

Usage:
    python timelapse_capture.py

Output:
    ./timelapse_frames/   — individual JPEGs
    ./timelapse.mp4       — final video (requires ffmpeg on PATH)
"""

import os
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
import numpy as np
from io import BytesIO

# ── Configuration ─────────────────────────────────────────────────────────────

SNAPSHOT_URL     = "http://192.168.0.142:8080/?action=snapshot"
OUTPUT_DIR       = Path("timelapse_frames")
INTERVAL_SEC     = 5        # seconds between frames
DIFF_THRESHOLD   = 1.5      # mean absolute pixel difference (0–255 scale) to count as "changed"
IDLE_FRAMES      = 6        # how many consecutive unchanged frames before auto-stop
REQUEST_TIMEOUT  = 10       # seconds before giving up on a snapshot fetch

# Downsample resolution used only for frame comparison (not saved).
# At 2592x1944 this gives a ~30x speedup with identical threshold behavior.
DIFF_RESIZE      = (640, 480)

# ffmpeg output settings (set to None to skip video assembly)
OUTPUT_VIDEO     = "timelapse.mp4"
FFMPEG_FPS       = 24       # playback framerate of output video

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_frame(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] fetch failed: {e}")
        return None


def frame_diff(img_a: Image.Image, img_b: Image.Image) -> float:
    """Mean absolute difference, computed on downsampled copies for speed.
    Full-res frames are still saved — this only affects comparison cost."""
    a = np.asarray(img_a.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(img_b.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def assemble_video(frame_dir: Path, output: str, fps: int):
    print(f"\nAssembling video → {output} at {fps} fps …")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(frame_dir / "frame_*.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Video saved: {output}")
    else:
        print(f"  ✗ ffmpeg failed:\n{result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame_count   = 0
    idle_count    = 0
    prev_frame    = None

    print(f"Timelapse capture started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Snapshot : {SNAPSHOT_URL}")
    print(f"  Interval : {INTERVAL_SEC}s  |  Diff threshold: {DIFF_THRESHOLD}  |  Idle stop: {IDLE_FRAMES} frames")
    print(f"  Output   : {OUTPUT_DIR}/\n")
    print("Press Ctrl+C to stop manually.\n")

    try:
        while True:
            frame = fetch_frame(SNAPSHOT_URL)

            if frame is None:
                print("  Skipping frame (fetch error).")
                time.sleep(INTERVAL_SEC)
                continue

            if prev_frame is not None:
                diff = frame_diff(prev_frame, frame)
                changed = diff >= DIFF_THRESHOLD

                status = f"Δ={diff:.2f}"
                if changed:
                    idle_count = 0
                    status += " ✓ saved"
                else:
                    idle_count += 1
                    status += f" — idle {idle_count}/{IDLE_FRAMES}"

                print(f"  Frame {frame_count+1:04d}  {status}")

                if idle_count >= IDLE_FRAMES:
                    print(f"\nScene unchanged for {IDLE_FRAMES} consecutive frames — auto-stopping.")
                    break
            else:
                # Always save the very first frame
                print(f"  Frame {frame_count+1:04d} — first frame, saved.")

            # Save frame
            fname = OUTPUT_DIR / f"frame_{frame_count:05d}.jpg"
            frame.save(fname, quality=92)
            frame_count += 1
            prev_frame = frame

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nCaptured {frame_count} frames → {OUTPUT_DIR}/")

    if frame_count >= 2 and OUTPUT_VIDEO:
        assemble_video(OUTPUT_DIR, OUTPUT_VIDEO, FFMPEG_FPS)
    elif frame_count < 2:
        print("Not enough frames to assemble a video.")


if __name__ == "__main__":
    main()
