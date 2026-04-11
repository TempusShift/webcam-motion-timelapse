#!/usr/bin/env python3
"""
Timelapse capture for laser engraver (or any mjpeg-streamer snapshot URL).
- Waits for activity before recording starts
- Auto-stops when the scene stops changing after IDLE_FRAMES consecutive unchanged frames

Requirements:
    pip install requests pillow numpy

Usage:
    python timelapse_capture.py

Output:
    ./timelapse_frames/   — individual JPEGs
    ./timelapse.mp4       — final video (requires ffmpeg on PATH)
"""

import time
import subprocess
from datetime import datetime
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

SNAPSHOT_URL     = "http://192.168.0.142:8080/?action=snapshot"
OUTPUT_DIR       = Path("timelapse_frames")
INTERVAL_SEC     = 5        # seconds between frames
DIFF_THRESHOLD   = 1.5      # mean absolute pixel difference (0–255) to count as "changed"
IDLE_FRAMES      = 6        # consecutive unchanged frames before auto-stop
REQUEST_TIMEOUT  = 10       # seconds before giving up on a snapshot fetch

# Downsample size used only for comparison — full-res frames are still saved.
# At 2592x1944 this gives ~30x faster diffs with identical threshold behavior.
DIFF_RESIZE      = (640, 480)

# ffmpeg output (set OUTPUT_VIDEO = None to skip video assembly)
OUTPUT_VIDEO     = "timelapse.mp4"
FFMPEG_FPS       = 24

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
    """Mean absolute difference computed on downsampled copies for speed."""
    a = np.asarray(img_a.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(img_b.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def save_frame(frame: Image.Image, index: int):
    fname = OUTPUT_DIR / f"frame_{index:05d}.jpg"
    frame.save(fname, quality=92)


def assemble_video(frame_dir: Path, output: str, fps: int):
    print(f"\nAssembling video -> {output} at {fps} fps ...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Video saved: {output}")
    else:
        print(f"  ffmpeg failed:\n{result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame_count = 0
    idle_count  = 0
    prev_frame  = None
    started     = False     # flips True on first meaningful change

    print(f"Timelapse capture ready -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Snapshot : {SNAPSHOT_URL}")
    print(f"  Interval : {INTERVAL_SEC}s  |  Diff threshold: {DIFF_THRESHOLD}  |  Idle stop: {IDLE_FRAMES} frames")
    print(f"  Output   : {OUTPUT_DIR}/\n")
    print("Waiting for activity ... (Ctrl+C to cancel)\n")

    try:
        while True:
            frame = fetch_frame(SNAPSHOT_URL)

            if frame is None:
                print("  Skipping frame (fetch error).")
                time.sleep(INTERVAL_SEC)
                continue

            # -- Pre-start: wait for first meaningful change --
            if not started:
                if prev_frame is None:
                    print("  Baseline captured, watching for changes ...")
                    prev_frame = frame
                    time.sleep(INTERVAL_SEC)
                    continue

                diff = frame_diff(prev_frame, frame)
                if diff < DIFF_THRESHOLD:
                    print(f"  Waiting ... delta={diff:.2f}")
                    prev_frame = frame
                    time.sleep(INTERVAL_SEC)
                    continue

                # First change detected -- save baseline + this frame to anchor the timelapse
                started = True
                print(f"  Activity detected (delta={diff:.2f}) -- recording started!\n")
                save_frame(prev_frame, frame_count)
                frame_count += 1

            # -- Recording active --
            else:
                diff = frame_diff(prev_frame, frame)
                changed = diff >= DIFF_THRESHOLD
                status = f"delta={diff:.2f}"

                if changed:
                    idle_count = 0
                    status += " saved"
                else:
                    idle_count += 1
                    status += f" -- idle {idle_count}/{IDLE_FRAMES}"

                print(f"  Frame {frame_count+1:04d}  {status}")

                if idle_count >= IDLE_FRAMES:
                    print(f"\nScene unchanged for {IDLE_FRAMES} consecutive frames -- auto-stopping.")
                    save_frame(frame, frame_count)
                    frame_count += 1
                    break

            save_frame(frame, frame_count)
            frame_count += 1
            prev_frame = frame
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nCaptured {frame_count} frames -> {OUTPUT_DIR}/")

    if frame_count >= 2 and OUTPUT_VIDEO:
        assemble_video(OUTPUT_DIR, OUTPUT_VIDEO, FFMPEG_FPS)
    else:
        print("Not enough frames to assemble a video.")


if __name__ == "__main__":
    main()
