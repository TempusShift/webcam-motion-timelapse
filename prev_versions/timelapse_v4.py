#!/usr/bin/env python3
"""
Continuous timelapse capture for laser engraver.
- Runs forever, watching for activity
- Starts a new session when movement is detected
- Ends the session (and saves video) after 2 min of inactivity
- Discards sessions shorter than 2 minutes of real capture time
- Loops back to watching after each session
- Sends Discord notifications on session start/end, uploads video if small enough

Requirements:
    pip3 install requests pillow numpy python-dotenv --break-system-packages
    ffmpeg on PATH (sudo apt install ffmpeg)

.env file (same directory as script):
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

Usage:
    python3 timelapse_capture.py
"""

import os
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

SNAPSHOT_URL        = "http://192.168.0.142:8080/?action=snapshot"
OUTPUT_DIR          = Path("timelapse_sessions")
INTERVAL_SEC        = 5

DIFF_THRESHOLD      = 1.5
MIN_SESSION_SEC     = 120        # discard session shorter than this
IDLE_STOP_SEC       = 120        # end session after this many seconds of no movement

DIFF_RESIZE         = (640, 480)
REQUEST_TIMEOUT     = 10
FFMPEG_FPS          = 24

# Discord video uploads are capped at 25MB (free) / 500MB (Nitro).
# If the video exceeds this, a download link is posted instead.
DISCORD_MAX_MB      = 25

# ── Discord ───────────────────────────────────────────────────────────────────

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def discord_send(content: str, file_path: Path | None = None):
    """Send a message to Discord, optionally attaching a file."""
    if not WEBHOOK_URL:
        return
    try:
        if file_path and file_path.exists():
            size_mb = file_path.stat().st_size / 1_048_576
            if size_mb <= DISCORD_MAX_MB:
                with open(file_path, "rb") as f:
                    requests.post(
                        WEBHOOK_URL,
                        data={"content": content},
                        files={"file": (file_path.name, f, "video/mp4")},
                        timeout=60,
                    )
                return
            else:
                # File too large — append a note and fall through to text-only
                content += f"\n_(video is {size_mb:.1f} MB — too large to upload directly)_"

        requests.post(WEBHOOK_URL, json={"content": content}, timeout=10)
    except Exception as e:
        print(f"  [WARN] Discord notify failed: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def frames_for_seconds(seconds: float) -> int:
    return max(1, int(seconds / INTERVAL_SEC))

def fetch_frame(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] fetch failed: {e}")
        return None

def frame_diff(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.asarray(img_a.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(img_b.resize(DIFF_RESIZE, Image.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))

def assemble_video(frame_dir: Path, output_path: Path, fps: int) -> bool:
    print(f"  Assembling {output_path.name} ...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Saved: {output_path}")
        return True
    else:
        print(f"  ffmpeg failed:\n{result.stderr}")
        return False

def fmt_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"

# ── Session ───────────────────────────────────────────────────────────────────

def run_session(session_dir: Path, first_frame: Image.Image, prev_frame: Image.Image,
                idle_limit: int) -> tuple[int, int]:
    """Returns (total_frames, active_frames) where active_frames excludes the idle tail."""
    session_dir.mkdir(parents=True, exist_ok=True)
    frame_count  = 0
    active_frames = 2   # baseline + first triggering frame both count as active
    idle_count   = 0

    prev_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
    frame_count += 1
    first_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
    frame_count += 1
    prev = first_frame

    print(f"  Recording -> {session_dir.name}  (idle limit: {idle_limit} frames = {IDLE_STOP_SEC}s)\n")

    while True:
        time.sleep(INTERVAL_SEC)
        frame = fetch_frame(SNAPSHOT_URL)

        if frame is None:
            print("  [WARN] fetch failed, skipping.")
            continue

        diff = frame_diff(prev, frame)
        changed = diff >= DIFF_THRESHOLD

        if changed:
            idle_count = 0
            active_frames += 1
            status = f"delta={diff:.2f}"
        else:
            idle_count += 1
            status = f"delta={diff:.2f} -- idle {idle_count}/{idle_limit}"

        # Always advance prev so gradual lighting drift doesn't
        # prevent the idle counter from reaching its limit
        prev = frame

        print(f"  Frame {frame_count+1:04d}  {status}  (active: {active_frames})")

        frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
        frame_count += 1

        if idle_count >= idle_limit:
            print(f"\n  Idle limit reached -- session ending.")
            break

    return frame_count, active_frames

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    idle_limit  = frames_for_seconds(IDLE_STOP_SEC)
    min_frames  = frames_for_seconds(MIN_SESSION_SEC)
    session_num = 0

    if WEBHOOK_URL:
        print("  Discord notifications: enabled")
    else:
        print("  Discord notifications: disabled (no DISCORD_WEBHOOK_URL in .env)")

    print(f"\nContinuous timelapse monitor started -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Snapshot  : {SNAPSHOT_URL}")
    print(f"  Interval  : {INTERVAL_SEC}s")
    print(f"  Threshold : {DIFF_THRESHOLD}  |  Idle stop: {IDLE_STOP_SEC}s ({idle_limit} frames)")
    print(f"  Min session: {MIN_SESSION_SEC}s ({min_frames} frames to keep)")
    print(f"  Sessions  : {OUTPUT_DIR}/\n")
    print("Watching for activity ... (Ctrl+C to quit)\n")

    prev_frame = None

    try:
        while True:
            frame = fetch_frame(SNAPSHOT_URL)

            if frame is None:
                time.sleep(INTERVAL_SEC)
                continue

            if prev_frame is None:
                print("  Baseline captured.")
                prev_frame = frame
                time.sleep(INTERVAL_SEC)
                continue

            diff = frame_diff(prev_frame, frame)

            if diff < DIFF_THRESHOLD:
                print(f"  Watching ... delta={diff:.2f}")
                # Don't update prev_frame -- compare against a fixed reference
                # so gradual changes (sunlight, shadows) don't accumulate past threshold
                time.sleep(INTERVAL_SEC)
                continue

            # ── Activity detected ──────────────────────────────────────────
            session_num += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = OUTPUT_DIR / f"session_{ts}"

            print(f"\n[Session {session_num}] Activity detected (delta={diff:.2f}) -- {ts}")
            discord_send(f"🔥 **Laser session started** — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            count, active_frames = run_session(session_dir, frame, prev_frame, idle_limit)

            real_seconds   = count * INTERVAL_SEC
            active_seconds = active_frames * INTERVAL_SEC
            duration_str   = fmt_duration(real_seconds)
            active_str     = fmt_duration(active_seconds)
            print(f"\n[Session {session_num}] {count} total frames, {active_frames} active frames (~{active_str} of movement)")

            if active_frames < min_frames:
                print(f"  Too short (< {MIN_SESSION_SEC}s active) -- discarding.")
                discord_send(f"⚠️ **Session discarded** — only {active_str} of activity (minimum {fmt_duration(MIN_SESSION_SEC)})")
                shutil.rmtree(session_dir)
            else:
                video_path = OUTPUT_DIR / f"session_{ts}.mp4"
                assembled = assemble_video(session_dir, video_path, FFMPEG_FPS)

                if assembled:
                    size_mb = video_path.stat().st_size / 1_048_576
                    msg = (
                        f"✅ **Laser session complete** — {duration_str}\n"
                        f"📹 `{video_path.name}` ({size_mb:.1f} MB)"
                    )
                    discord_send(msg, file_path=video_path)
                else:
                    discord_send(
                        f"✅ **Laser session complete** — {duration_str}\n"
                        f"⚠️ Video assembly failed — frames saved to `{session_dir.name}`"
                    )

            prev_frame = None
            print(f"\nBack to watching ... (Ctrl+C to quit)\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")
        discord_send("🛑 **Timelapse monitor stopped** (manual)")

if __name__ == "__main__":
    main()
