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
import json
import time
import shutil
import signal
import subprocess
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

def _detect_version() -> str:
    """Infer version from the highest timelapse_vN.py file in the same directory."""
    script_dir = Path(__file__).parent
    versions = [
        int(p.stem.replace("timelapse_v", ""))
        for p in script_dir.glob("timelapse_v*.py")
        if p.stem.replace("timelapse_v", "").isdigit()
    ]
    return f"v{max(versions) + 1}" if versions else "v1"

VERSION = _detect_version()

TZ                  = ZoneInfo("America/Chicago")

SNAPSHOT_URL        = "http://192.168.0.142:8080/?action=snapshot"
OUTPUT_DIR          = Path("timelapse_sessions")
INTERVAL_SEC        = 5

DIFF_THRESHOLD      = 1.5        # threshold during a session to count a frame as active
START_THRESHOLD     = 3.0        # higher threshold required to trigger a new session
START_CONSECUTIVE   = 3          # number of consecutive frames above START_THRESHOLD to begin a session
IDLE_RESET_FRAMES   = 3          # consecutive active frames required to reset the idle counter
MIN_SESSION_SEC     = 120        # discard session shorter than this
IDLE_STOP_SEC       = 120        # end session after this many seconds of no movement

DIFF_RESIZE         = (640, 480)
REQUEST_TIMEOUT     = 10
FFMPEG_FPS          = 24
FILE_SERVER_BASE    = "http://nas.lan:8181/sessions"  # base URL for video links in Discord

# Discord video uploads are capped at 25MB for webhooks regardless of Nitro.
# Videos over this limit will be posted as a LAN download link instead.
DISCORD_MAX_MB          = 25
FILESERVER_BASE_URL     = "http://192.168.0.142:8181/sessions"  # LAN URL for the nginx file server
DISCORD_SNAPSHOT_SEC    = 300    # post a progress snapshot every N seconds (0 to disable)

# ── Discord ───────────────────────────────────────────────────────────────────

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def discord_send(content: str, file_path: Path | None = None):
    """Send a message to Discord, optionally attaching a file. Retries once on timeout."""
    if not WEBHOOK_URL:
        return
    for attempt in range(2):
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
                    lan_url = f"{FILESERVER_BASE_URL}/{file_path.parent.name}/{file_path.name}"
                    content += f"\n_(video is {size_mb:.1f} MB — too large for webhook)_\n📥 {lan_url}"

            requests.post(WEBHOOK_URL, json={"content": content}, timeout=20)
            return
        except Exception as e:
            print(f"  [WARN] Discord notify failed (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(3)


def discord_send_image(content: str, img: Image.Image):
    """Send a message with an inline JPEG snapshot."""
    if not WEBHOOK_URL:
        return
    try:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        requests.post(
            WEBHOOK_URL,
            data={"content": content},
            files={"file": ("snapshot.jpg", buf, "image/jpeg")},
            timeout=30,
        )
    except Exception as e:
        print(f"  [WARN] Discord snapshot failed: {e}")


def start_snapshot_thread(stop_event: threading.Event, get_latest_frame, session_start: datetime) -> threading.Thread:
    """Fires every DISCORD_SNAPSHOT_SEC, posts the latest frame as a progress update."""
    def _loop():
        while not stop_event.wait(timeout=DISCORD_SNAPSHOT_SEC):
            frame = get_latest_frame()
            if frame is None:
                continue
            elapsed = fmt_duration(int((now() - session_start).total_seconds()))
            print(f"  [Discord] Posting progress snapshot ({elapsed} elapsed)")
            discord_send_image(f"📷 **Progress snapshot** — {elapsed} into session", frame)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

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

def now() -> datetime:
    """Current time in configured timezone."""
    return datetime.now(tz=TZ)

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def write_meta(session_dir: Path, video_path: Path, total_frames: int,
               active_frames: int | None = None,
               ended_dt: datetime | None = None,
               recovered: bool = False):
    """Write meta.json. Start time is always inferred from the session dir name."""
    import re
    m = re.match(r'session_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', session_dir.name)
    started_dt = datetime(int(m[1]), int(m[2]), int(m[3]),
                          int(m[4]), int(m[5]), int(m[6]), tzinfo=TZ) if m else now()
    total_seconds  = total_frames * INTERVAL_SEC
    active_seconds = (active_frames * INTERVAL_SEC) if active_frames is not None else None
    meta = {
        "started":        started_dt.isoformat(),
        "ended":          ended_dt.isoformat() if ended_dt else None,
        "active_seconds": active_seconds,
        "total_seconds":  total_seconds,
        "active_frames":  active_frames,
        "total_frames":   total_frames,
        "video":          video_path.name,
        "recovered":      recovered,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# ── Session ───────────────────────────────────────────────────────────────────

def run_session(session_dir: Path, first_frame: Image.Image, prev_frame: Image.Image,
                idle_limit: int) -> tuple[int, int]:
    """Returns (total_frames, active_frames) where active_frames excludes the idle tail."""
    session_dir.mkdir(parents=True, exist_ok=True)
    frame_count   = 0
    active_frames = 2   # baseline + first triggering frame both count as active
    idle_count    = 0
    active_streak = 0   # consecutive active frames; must reach IDLE_RESET_FRAMES to reset idle_count
    session_start = now()

    # Shared latest frame for the snapshot thread (wrapped in a list for mutability)
    latest = [first_frame]

    prev_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
    frame_count += 1
    first_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
    frame_count += 1
    prev = first_frame

    print(f"  Recording -> {session_dir.name}  (idle limit: {idle_limit} frames = {IDLE_STOP_SEC}s)\n")

    # Start progress snapshot thread if enabled
    stop_event = threading.Event()
    if DISCORD_SNAPSHOT_SEC > 0:
        start_snapshot_thread(stop_event, lambda: latest[0], session_start)

    try:
        while not _shutdown.is_set():
            if _shutdown.wait(timeout=INTERVAL_SEC):
                print("\n  Shutdown signal received -- ending session.")
                break
            frame = fetch_frame(SNAPSHOT_URL)

            if frame is None:
                print("  [WARN] fetch failed, skipping.")
                continue

            latest[0] = frame

            diff = frame_diff(prev, frame)
            changed = diff >= DIFF_THRESHOLD

            if changed:
                active_streak += 1
                active_frames += 1
                if active_streak >= IDLE_RESET_FRAMES:
                    if idle_count > 0:
                        print(f"  Streak of {active_streak} active frames -- idle counter reset.")
                    idle_count = 0
                status = f"delta={diff:.2f} streak={active_streak}"
            else:
                active_streak = 0
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
    finally:
        stop_event.set()

    return frame_count, active_frames

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    idle_limit  = frames_for_seconds(IDLE_STOP_SEC)
    min_frames  = frames_for_seconds(MIN_SESSION_SEC)
    session_num = 0

    # On startup, check for incomplete sessions from a previous run (dir exists but no mp4)
    for d in sorted(OUTPUT_DIR.glob("session_*")):
        if not d.is_dir() or any(d.glob("*.mp4")):
            continue
        frames = sorted(d.glob("frame_*.jpg"))
        if len(frames) >= min_frames:
            print(f"  Incomplete session {d.name} has {len(frames)} frames -- attempting recovery ...")
            video_path = d / f"{d.name}.mp4"
            assembled = assemble_video(d, video_path, FFMPEG_FPS)
            if assembled:
                write_meta(d, video_path, total_frames=len(frames), recovered=True)

                size_mb = video_path.stat().st_size / 1_048_576
                video_url = f"{FILE_SERVER_BASE}/{d.name}/{video_path.name}"
                if size_mb <= DISCORD_MAX_MB:
                    discord_send(f"♻️ **Recovered incomplete session** — {len(frames)} frames\n📹 `{video_path.name}` ({size_mb:.1f} MB)", file_path=video_path)
                else:
                    discord_send(f"♻️ **Recovered incomplete session** — {len(frames)} frames\n📹 [{video_path.name}]({video_url}) ({size_mb:.1f} MB)")
            else:
                print(f"  Recovery failed — leaving {d.name} for manual inspection.")
        else:
            print(f"  Cleaning up incomplete session {d.name} ({len(frames)} frames, below minimum).")
            shutil.rmtree(d)

    if WEBHOOK_URL:
        print("  Discord notifications: enabled")
    else:
        print("  Discord notifications: disabled (no DISCORD_WEBHOOK_URL in .env)")

    print(f"\nContinuous timelapse monitor {VERSION} started -- {fmt_ts(now())}")
    print(f"  Snapshot   : {SNAPSHOT_URL}")
    print(f"  Interval   : {INTERVAL_SEC}s")
    print(f"  Start      : delta >= {START_THRESHOLD} for {START_CONSECUTIVE} consecutive frames")
    print(f"  Active     : delta >= {DIFF_THRESHOLD}  |  idle reset after {IDLE_RESET_FRAMES} consecutive active frames")
    print(f"  Idle stop  : {IDLE_STOP_SEC}s ({idle_limit} frames)")
    print(f"  Min session: {MIN_SESSION_SEC}s ({min_frames} frames to keep)")
    print(f"  Sessions   : {OUTPUT_DIR}/\n")
    print("Watching for activity ... (Ctrl+C to quit)\n")

    # Send startup notification in background so a slow network doesn't block capture
    threading.Thread(
        target=discord_send,
        args=(f"🟢 **Timelapse monitor {VERSION} started** — {fmt_ts(now())}",),
        daemon=False,
    ).start()

    prev_frame = None
    start_streak = 0   # consecutive frames above START_THRESHOLD

    try:
        while not _shutdown.is_set():
            frame = fetch_frame(SNAPSHOT_URL)

            if frame is None:
                if _shutdown.wait(timeout=INTERVAL_SEC):
                    break
                continue

            if prev_frame is None:
                print("  Baseline captured.")
                prev_frame = frame
                if _shutdown.wait(timeout=INTERVAL_SEC):
                    break
                continue

            diff = frame_diff(prev_frame, frame)

            if diff < START_THRESHOLD:
                if start_streak > 0:
                    print(f"  Streak broken (delta={diff:.2f}) -- resetting.")
                    start_streak = 0
                else:
                    print(f"  Watching ... delta={diff:.2f}")
                # Don't update prev_frame -- compare against a fixed reference
                # so gradual changes (sunlight, shadows) don't accumulate past threshold
                if _shutdown.wait(timeout=INTERVAL_SEC):
                    return
                continue

            start_streak += 1
            print(f"  Start streak {start_streak}/{START_CONSECUTIVE} delta={diff:.2f}")

            if start_streak < START_CONSECUTIVE:
                if _shutdown.wait(timeout=INTERVAL_SEC):
                    return
                continue

            # ── START_CONSECUTIVE frames above START_THRESHOLD -- begin session ──
            start_streak = 0
            session_num += 1
            session_start_dt = now()
            ts = session_start_dt.strftime("%Y%m%d_%H%M%S")
            session_dir = OUTPUT_DIR / f"session_{ts}"

            print(f"\n[Session {session_num}] Activity detected (delta={diff:.2f}) -- {fmt_ts(session_start_dt)}")
            discord_send(f"🔥 **Laser session started** — {fmt_ts(session_start_dt)}")

            count, active_frames = run_session(session_dir, frame, prev_frame, idle_limit)

            session_end_dt   = now()
            real_seconds     = count * INTERVAL_SEC
            active_seconds   = active_frames * INTERVAL_SEC
            duration_str     = fmt_duration(real_seconds)
            active_str       = fmt_duration(active_seconds)
            print(f"\n[Session {session_num}] {count} total frames, {active_frames} active frames (~{active_str} of movement)")

            if active_frames < min_frames:
                print(f"  Too short (< {MIN_SESSION_SEC}s active) -- discarding.")
                discord_send(f"⚠️ **Session discarded** — only {active_str} of activity (minimum {fmt_duration(MIN_SESSION_SEC)})")
                shutil.rmtree(session_dir)
            else:
                video_path = session_dir / f"session_{ts}.mp4"
                assembled = assemble_video(session_dir, video_path, FFMPEG_FPS)

                if assembled:
                    write_meta(session_dir, video_path,
                               total_frames=count,
                               active_frames=active_frames,
                               ended_dt=session_end_dt)

                    size_mb = video_path.stat().st_size / 1_048_576
                    video_url = f"{FILE_SERVER_BASE}/{session_dir.name}/{video_path.name}"
                    if size_mb <= DISCORD_MAX_MB:
                        msg = (
                            f"✅ **Laser session complete** — {active_str} active, {duration_str} total\n"
                            f"📹 `{video_path.name}` ({size_mb:.1f} MB)"
                        )
                        discord_send(msg, file_path=video_path)
                    else:
                        msg = (
                            f"✅ **Laser session complete** — {active_str} active, {duration_str} total\n"
                            f"📹 [{video_path.name}]({video_url}) ({size_mb:.1f} MB)"
                        )
                        discord_send(msg)
                else:
                    discord_send(
                        f"✅ **Laser session complete** — {active_str} active\n"
                        f"⚠️ Video assembly failed — frames saved to `{session_dir.name}`"
                    )

            prev_frame = None
            print(f"\nBack to watching ... (Ctrl+C to quit)\n")

    except (KeyboardInterrupt, SystemExit):
        pass

    print("\nStopped.")
    t = threading.Thread(target=discord_send, args=("🛑 **Timelapse monitor stopped**",), daemon=False)
    t.start()
    t.join(timeout=25)  # wait up to 25s for message to send before process exits

_shutdown = threading.Event()

def _handle_sigterm(sig, frame):
    print("  [SIGTERM received] shutting down ...")
    _shutdown.set()

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGHUP, _handle_sigterm)
    main()
