#!/usr/bin/env python3
"""
Continuous timelapse capture for laser engraver.
- Runs forever, watching for activity
- Starts a new session when movement is detected
- Ends the session (and saves video) after 2 min of inactivity
- Discards sessions shorter than 2 minutes of real capture time
- Loops back to watching after each session
- Sends Discord notifications on session start/end, uploads video if small enough
- Hot-reload: touch reload.flag to reload code at next session boundary (or mid-session safely)
- Control API on configurable port for manual start/stop, delete, config updates

Requirements:
    pip3 install requests pillow numpy python-dotenv pyyaml tzdata --break-system-packages
    ffmpeg on PATH (sudo apt install ffmpeg)

.env file (same directory as script):
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

Usage:
    python3 timelapse.py
"""

import os
import re
import sys
import json
import time
import shutil
import signal
import subprocess
import tempfile
import threading
import collections
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler

import yaml
import requests
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Load config ──────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.yml"
with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

# ── Configuration ─────────────────────────────────────────────────────────────

def _detect_version() -> str:
    script_dir = Path(__file__).parent
    versions = [
        int(p.stem.replace("timelapse_v", ""))
        for p in script_dir.glob("timelapse_v*.py")
        if p.stem.replace("timelapse_v", "").isdigit()
    ]
    return f"v{max(versions) + 1}" if versions else "v1"

VERSION             = _detect_version()
TZ                  = ZoneInfo(_cfg["timezone"])

SNAPSHOT_URL        = _cfg["snapshot_url"]
OUTPUT_DIR          = Path(_cfg["output_dir"])
INTERVAL_SEC        = _cfg["interval_sec"]

DIFF_THRESHOLD      = _cfg["diff_threshold"]
START_THRESHOLD     = _cfg["start_threshold"]
START_CONSECUTIVE   = _cfg["start_consecutive"]
IDLE_RESET_FRAMES   = _cfg["idle_reset_frames"]
MIN_SESSION_SEC     = _cfg["min_session_sec"]
IDLE_STOP_SEC       = _cfg["idle_stop_sec"]

DIFF_RESIZE         = tuple(_cfg["diff_resize"])
REQUEST_TIMEOUT     = _cfg["request_timeout"]
FFMPEG_FPS          = _cfg["ffmpeg_fps"]
FILE_SERVER_BASE    = _cfg["file_server_base"]

DISCORD_MAX_MB      = _cfg["discord_max_mb"]
DISCORD_SNAPSHOT_SEC = _cfg["discord_snapshot_sec"]
API_PORT            = _cfg.get("api_port", 7820)

RELOAD_FLAG         = Path(__file__).parent / "reload.flag"
RESUME_FILE         = OUTPUT_DIR / "resume.json"

# ── Log buffer ───────────────────────────────────────────────────────────────

_log_buffer: collections.deque[str] = collections.deque(maxlen=100)

class _LogTee:
    """Intercepts stdout writes, splits on newlines, and stores lines in _log_buffer."""
    def __init__(self, real):
        self._real = real
        self._pending = ""

    def write(self, s: str):
        self._real.write(s)
        self._pending += s
        while '\n' in self._pending:
            line, self._pending = self._pending.split('\n', 1)
            _log_buffer.append(line)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

sys.stdout = _LogTee(sys.stdout)

# ── Signals / events ──────────────────────────────────────────────────────────

_shutdown    = threading.Event()
_reload      = threading.Event()
_force_start = threading.Event()
_force_stop  = threading.Event()
_wakeup      = threading.Event()

# Shared state readable by API handlers (written only by main thread)
_api_state       = "idle"
_api_session_dir = None

def _handle_sigterm(sig, frame):
    print("  [SIGTERM received] shutting down ...")
    _shutdown.set()
    _wakeup.set()

# ── Discord ───────────────────────────────────────────────────────────────────

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def discord_send(content: str, file_path: Path | None = None):
    if not WEBHOOK_URL:
        return
    for attempt in range(2):
        try:
            if file_path and file_path.exists():
                size_mb = file_path.stat().st_size / 1_048_576
                if size_mb <= DISCORD_MAX_MB:
                    with open(file_path, "rb") as f:
                        requests.post(WEBHOOK_URL, data={"content": content},
                                      files={"file": (file_path.name, f, "video/mp4")}, timeout=60)
                    return
                else:
                    content += f"\n_(video is {size_mb:.1f} MB — too large for webhook)_"
            requests.post(WEBHOOK_URL, json={"content": content}, timeout=20)
            return
        except Exception as e:
            print(f"  [WARN] Discord notify failed (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(3)

def discord_send_image(content: str, img: Image.Image):
    if not WEBHOOK_URL:
        return
    try:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        requests.post(WEBHOOK_URL, data={"content": content},
                      files={"file": ("snapshot.jpg", buf, "image/jpeg")}, timeout=30)
    except Exception as e:
        print(f"  [WARN] Discord snapshot failed: {e}")

def start_snapshot_thread(stop_event: threading.Event, get_latest_frame, session_start: datetime):
    def _loop():
        while not stop_event.wait(timeout=DISCORD_SNAPSHOT_SEC):
            frame = get_latest_frame()
            if frame is None:
                continue
            elapsed = fmt_duration(int((now() - session_start).total_seconds()))
            print(f"  [Discord] Posting progress snapshot ({elapsed} elapsed)")
            discord_send_image(f"📷 **Progress snapshot** — {elapsed} into session", frame)
    threading.Thread(target=_loop, daemon=True).start()

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
    a = np.asarray(img_a.resize(DIFF_RESIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    b = np.asarray(img_b.resize(DIFF_RESIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))

def assemble_video(frame_dir: Path, output_path: Path, fps: int) -> bool:
    print(f"  Assembling {output_path.name} ...")
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", str(frame_dir / "frame_%05d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Saved: {output_path}")
        return True
    print(f"  ffmpeg failed:\n{result.stderr}")
    return False

def fmt_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"

def now() -> datetime:
    return datetime.now(tz=TZ)

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def write_status(state: str, session_dir: Path | None = None, expected_sec: int | None = None):
    status_path = OUTPUT_DIR / "status.json"
    # Preserve session_started timestamp across repeated recording writes
    session_started = now().isoformat() if state == "recording" else None
    if state == "recording" and status_path.exists():
        try:
            prev = json.loads(status_path.read_text())
            if prev.get("session") == (session_dir.name if session_dir else None):
                session_started = prev.get("session_started", session_started)
        except Exception:
            pass
    status = {
        "state":           state,
        "updated":         now().isoformat(),
        "session":         session_dir.name if session_dir else None,
        "session_started": session_started,
        "expected_sec":    expected_sec,
        "version":         VERSION,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=status_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status, f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, status_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

def write_meta(session_dir: Path, video_path: Path, total_frames: int,
               active_frames: int | None = None,
               ended_dt: datetime | None = None,
               recovered: bool = False):
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

# ── Reload ────────────────────────────────────────────────────────────────────

def check_reload_flag():
    """Returns True if reload.flag exists. Removes it."""
    if RELOAD_FLAG.exists():
        RELOAD_FLAG.unlink()
        return True
    return False

def write_resume(session_dir: Path, frame_count: int, active_frames: int, idle_count: int):
    resume = {
        "session_dir":   str(session_dir),
        "frame_count":   frame_count,
        "active_frames": active_frames,
        "idle_count":    idle_count,
    }
    RESUME_FILE.write_text(json.dumps(resume))
    print(f"  [Reload] Resume state written: {frame_count} frames, {active_frames} active")

def load_resume() -> dict | None:
    if not RESUME_FILE.exists():
        return None
    try:
        data = json.loads(RESUME_FILE.read_text())
        RESUME_FILE.unlink()
        return data
    except Exception as e:
        print(f"  [WARN] Failed to read resume.json: {e}")
        RESUME_FILE.unlink(missing_ok=True)
        return None

def do_reload():
    print(f"\n  [Reload] Reloading script...")
    discord_send(f"🔄 **Timelapse monitor reloading** — {fmt_ts(now())}")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ── Control API ──────────────────────────────────────────────────────────────

_SESSION_NAME_RE = re.compile(r'^session_\d{8}_\d{6}$')

class ApiHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request stderr noise

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ── Routes ────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/api/config":
            return self._handle_get_config()
        if self.path == "/api/logs":
            return self._handle_get_logs()
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self):
        if self.path == "/api/session/start":
            return self._handle_start()
        if self.path == "/api/session/stop":
            return self._handle_stop()
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_PUT(self):
        if self.path == "/api/config":
            return self._handle_put_config()
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_DELETE(self):
        # /api/session/<name>
        m = re.match(r'^/api/session/(.+)$', self.path)
        if m:
            return self._handle_delete(m.group(1))
        self._send_json(404, {"ok": False, "error": "Not found"})

    # ── Handlers ──────────────────────────────────────────────────────────

    def _handle_start(self):
        if _api_state == "recording":
            return self._send_json(409, {"ok": False, "error": "Already recording"})
        if _api_state == "idle":
            return self._send_json(409, {"ok": False, "error": "Monitor is not running"})
        _force_start.set()
        _wakeup.set()
        self._send_json(200, {"ok": True})

    def _handle_stop(self):
        if _api_state != "recording":
            return self._send_json(409, {"ok": False, "error": "Not recording"})
        _force_stop.set()
        _wakeup.set()
        self._send_json(200, {"ok": True})

    def _handle_delete(self, name: str):
        if not _SESSION_NAME_RE.match(name):
            return self._send_json(400, {"ok": False, "error": "Invalid session name"})
        session_path = OUTPUT_DIR / name
        if not session_path.is_dir():
            return self._send_json(404, {"ok": False, "error": "Session not found"})
        if _api_session_dir and _api_session_dir.resolve() == session_path.resolve():
            return self._send_json(409, {"ok": False, "error": "Cannot delete the active session"})
        shutil.rmtree(session_path)
        print(f"  [API] Deleted session {name}")
        self._send_json(200, {"ok": True})

    def _handle_get_config(self):
        try:
            with open(_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            self._send_json(200, {"ok": True, "config": cfg})
        except Exception as e:
            self._send_json(500, {"ok": False, "error": str(e)})

    def _handle_get_logs(self):
        self._send_json(200, {"ok": True, "lines": list(_log_buffer)})

    def _handle_put_config(self):
        try:
            updates = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError) as e:
            return self._send_json(400, {"ok": False, "error": f"Invalid JSON: {e}"})

        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        unknown = [k for k in updates if k not in cfg]
        if unknown:
            return self._send_json(400, {"ok": False, "error": f"Unknown keys: {', '.join(unknown)}"})

        cfg.update(updates)
        with open(_CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Touch reload flag so the process picks up the new config
        RELOAD_FLAG.touch()
        print(f"  [API] Config updated: {list(updates.keys())} -- reload triggered")
        self._send_json(200, {"ok": True, "config": cfg, "reloading": True})


def start_api_server():
    server = HTTPServer(("0.0.0.0", API_PORT), ApiHandler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"  Control API: http://0.0.0.0:{API_PORT}/api/")

# ── Sleep helper ─────────────────────────────────────────────────────────────

def _sleep(seconds: float = INTERVAL_SEC) -> bool:
    """Sleep for `seconds`, but wake immediately on any event.
    Returns True if shutdown was requested."""
    _wakeup.wait(timeout=seconds)
    _wakeup.clear()
    return _shutdown.is_set()

# ── Session ───────────────────────────────────────────────────────────────────

def run_session(session_dir: Path, first_frame: Image.Image, prev_frame: Image.Image,
                idle_limit: int,
                resume_frame_count: int = 0,
                resume_active_frames: int = 0,
                resume_idle_count: int = 0) -> tuple[int, int]:
    """Returns (total_frames, active_frames) where active_frames excludes the idle tail."""
    session_dir.mkdir(parents=True, exist_ok=True)

    # If resuming, pick up counters from where we left off
    if resume_frame_count > 0:
        frame_count   = resume_frame_count
        active_frames = resume_active_frames
        idle_count    = resume_idle_count
        print(f"  Resuming session at frame {frame_count}, {active_frames} active, idle {idle_count}/{idle_limit}\n")
    else:
        frame_count   = 0
        active_frames = 2   # baseline + triggering frame
        idle_count    = 0
        prev_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
        frame_count += 1
        first_frame.save(session_dir / f"frame_{frame_count:05d}.jpg", quality=92)
        frame_count += 1

    active_streak = 0
    session_start = now()
    latest = [first_frame]
    prev   = first_frame

    print(f"  Recording -> {session_dir.name}  (idle limit: {idle_limit} frames = {IDLE_STOP_SEC}s)\n")

    stop_event = threading.Event()
    if DISCORD_SNAPSHOT_SEC > 0:
        start_snapshot_thread(stop_event, lambda: latest[0], session_start)

    try:
        while not _shutdown.is_set():
            if _sleep():
                print("\n  Shutdown signal received -- ending session.")
                break

            if _force_stop.is_set():
                _force_stop.clear()
                print("\n  [API] Manual stop requested -- ending session.")
                break

            # Check for reload flag mid-session -- write resume and execv
            if check_reload_flag():
                write_resume(session_dir, frame_count, active_frames, idle_count)
                stop_event.set()
                do_reload()  # does not return

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
    global _api_state, _api_session_dir

    OUTPUT_DIR.mkdir(exist_ok=True)

    idle_limit  = frames_for_seconds(IDLE_STOP_SEC)
    min_frames  = frames_for_seconds(MIN_SESSION_SEC)
    session_num = 0

    start_api_server()

    # ── Check for resume state from a mid-session reload ──────────────────────
    resume = load_resume()
    if resume:
        session_dir   = Path(resume["session_dir"])
        frame_count   = resume["frame_count"]
        active_frames = resume["active_frames"]
        idle_count    = resume["idle_count"]
        print(f"  [Reload] Resuming session {session_dir.name} at frame {frame_count}")
        expected_sec = None
        ref_file = OUTPUT_DIR / "reference.json"
        if ref_file.exists():
            try:
                expected_sec = json.loads(ref_file.read_text()).get("active_seconds")
            except Exception:
                pass
        write_status("recording", session_dir, expected_sec=expected_sec)

        _api_state = "recording"
        _api_session_dir = session_dir

        # Grab a fresh frame to use as reference
        prev_frame = fetch_frame(SNAPSHOT_URL)
        first_frame = prev_frame or Image.new("RGB", (640, 480))

        count, active_frames = run_session(
            session_dir, first_frame, first_frame, idle_limit,
            resume_frame_count=frame_count,
            resume_active_frames=active_frames,
            resume_idle_count=idle_count,
        )

        _api_state = "watching"
        _api_session_dir = None

        session_end_dt = now()
        real_seconds   = count * INTERVAL_SEC
        active_seconds = active_frames * INTERVAL_SEC
        duration_str   = fmt_duration(real_seconds)
        active_str     = fmt_duration(active_seconds)
        ts             = session_dir.name.replace("session_", "")

        print(f"\n[Resumed Session] {count} total frames, {active_frames} active (~{active_str})")

        if active_frames < min_frames:
            print(f"  Too short -- discarding.")
            shutil.rmtree(session_dir)
        else:
            video_path = session_dir / f"session_{ts}.mp4"
            assembled  = assemble_video(session_dir, video_path, FFMPEG_FPS)
            if assembled:
                write_meta(session_dir, video_path,
                           total_frames=count, active_frames=active_frames, ended_dt=session_end_dt)
                size_mb   = video_path.stat().st_size / 1_048_576
                video_url = f"{FILE_SERVER_BASE}/{session_dir.name}/{video_path.name}"
                msg = (f"✅ **Laser session complete (reloaded)** — {active_str} active, {duration_str} total\n"
                       f"📹 `{video_path.name}` ({size_mb:.1f} MB)" if size_mb <= DISCORD_MAX_MB
                       else f"✅ **Laser session complete (reloaded)** — {active_str} active, {duration_str} total\n"
                            f"📹 [{video_path.name}]({video_url}) ({size_mb:.1f} MB)")
                discord_send(msg, file_path=video_path if size_mb <= DISCORD_MAX_MB else None)

        write_status("watching")
        # Fall through to normal watching loop after resumed session ends

    # ── Check for incomplete sessions from a crashed/killed run ───────────────
    for d in sorted(OUTPUT_DIR.glob("session_*")):
        if not d.is_dir() or any(d.glob("*.mp4")):
            continue
        frames = sorted(d.glob("frame_*.jpg"))
        if len(frames) >= min_frames:
            print(f"  Incomplete session {d.name} has {len(frames)} frames -- attempting recovery ...")
            video_path = d / f"{d.name}.mp4"
            assembled  = assemble_video(d, video_path, FFMPEG_FPS)
            if assembled:
                write_meta(d, video_path, total_frames=len(frames), recovered=True)
                size_mb   = video_path.stat().st_size / 1_048_576
                video_url = f"{FILE_SERVER_BASE}/{d.name}/{video_path.name}"
                if size_mb <= DISCORD_MAX_MB:
                    discord_send(f"♻️ **Recovered incomplete session** — {len(frames)} frames\n"
                                 f"📹 `{video_path.name}` ({size_mb:.1f} MB)", file_path=video_path)
                else:
                    discord_send(f"♻️ **Recovered incomplete session** — {len(frames)} frames\n"
                                 f"📹 [{video_path.name}]({video_url}) ({size_mb:.1f} MB)")
            else:
                print(f"  Recovery failed -- leaving {d.name} for manual inspection.")
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
    write_status("watching")
    _api_state = "watching"

    threading.Thread(
        target=discord_send,
        args=(f"🟢 **Timelapse monitor {VERSION} started** — {fmt_ts(now())}",),
        daemon=False,
    ).start()

    prev_frame  = None
    start_streak = 0

    try:
        while not _shutdown.is_set():
            # Check reload flag between sessions
            if check_reload_flag():
                do_reload()  # does not return

            # ── Check for manual start ────────────────────────────────────
            manual_start = _force_start.is_set()
            if manual_start:
                _force_start.clear()
                print("\n  [API] Manual session start requested.")

            frame = fetch_frame(SNAPSHOT_URL)
            if frame is None:
                if manual_start:
                    print("  [API] Cannot start -- camera unavailable.")
                if _sleep():
                    break
                continue

            if not manual_start:
                # ── Normal streak detection ───────────────────────────────
                if prev_frame is None:
                    print("  Baseline captured.")
                    prev_frame = frame
                    if _sleep():
                        break
                    continue

                diff = frame_diff(prev_frame, frame)

                if diff < START_THRESHOLD:
                    if start_streak > 0:
                        print(f"  Streak broken (delta={diff:.2f}) -- resetting.")
                        start_streak = 0
                    else:
                        print(f"  Watching ... delta={diff:.2f}")
                    prev_frame = frame
                    if _sleep():
                        return
                    continue

                start_streak += 1
                print(f"  Start streak {start_streak}/{START_CONSECUTIVE} delta={diff:.2f}")

                if start_streak < START_CONSECUTIVE:
                    if _sleep():
                        return
                    continue
            else:
                # Manual start -- use current frame, skip streak detection
                if prev_frame is None:
                    prev_frame = frame

            # ── Session start ──────────────────────────────────────────────────
            start_streak     = 0
            session_num     += 1
            session_start_dt = now()
            ts               = session_start_dt.strftime("%Y%m%d_%H%M%S")
            session_dir      = OUTPUT_DIR / f"session_{ts}"

            tag = " (manual)" if manual_start else ""
            print(f"\n[Session {session_num}] Activity detected{tag} -- {fmt_ts(session_start_dt)}")
            discord_send(f"🔥 **Laser session started{tag}** — {fmt_ts(session_start_dt)}")

            # Load reference duration if user has selected one via the index page
            expected_sec = None
            ref_file = OUTPUT_DIR / "reference.json"
            if ref_file.exists():
                try:
                    expected_sec = json.loads(ref_file.read_text()).get("active_seconds")
                except Exception:
                    pass

            write_status("recording", session_dir, expected_sec=expected_sec)
            _api_state = "recording"
            _api_session_dir = session_dir

            count, active_frames = run_session(session_dir, frame, prev_frame, idle_limit)

            _api_state = "watching"
            _api_session_dir = None

            session_end_dt = now()
            real_seconds   = count * INTERVAL_SEC
            active_seconds = active_frames * INTERVAL_SEC
            duration_str   = fmt_duration(real_seconds)
            active_str     = fmt_duration(active_seconds)
            print(f"\n[Session {session_num}] {count} total frames, {active_frames} active (~{active_str})")

            if active_frames < min_frames:
                print(f"  Too short (< {MIN_SESSION_SEC}s active) -- discarding.")
                discord_send(f"⚠️ **Session discarded** — only {active_str} of activity (minimum {fmt_duration(MIN_SESSION_SEC)})")
                shutil.rmtree(session_dir)
            else:
                video_path = session_dir / f"session_{ts}.mp4"
                assembled  = assemble_video(session_dir, video_path, FFMPEG_FPS)
                if assembled:
                    write_meta(session_dir, video_path,
                               total_frames=count, active_frames=active_frames, ended_dt=session_end_dt)
                    size_mb   = video_path.stat().st_size / 1_048_576
                    video_url = f"{FILE_SERVER_BASE}/{session_dir.name}/{video_path.name}"
                    if size_mb <= DISCORD_MAX_MB:
                        msg = (f"✅ **Laser session complete** — {active_str} active, {duration_str} total\n"
                               f"📹 `{video_path.name}` ({size_mb:.1f} MB)")
                        discord_send(msg, file_path=video_path)
                    else:
                        msg = (f"✅ **Laser session complete** — {active_str} active, {duration_str} total\n"
                               f"📹 [{video_path.name}]({video_url}) ({size_mb:.1f} MB)")
                        discord_send(msg)
                else:
                    discord_send(f"✅ **Laser session complete** — {active_str} active\n"
                                 f"⚠️ Video assembly failed — frames saved to `{session_dir.name}`")

            prev_frame = None
            write_status("watching")
            print(f"\nBack to watching ... (Ctrl+C to quit)\n")

    except (KeyboardInterrupt, SystemExit):
        pass

    _api_state = "idle"
    print("\nStopped.")
    write_status("idle")
    t = threading.Thread(target=discord_send, args=("🛑 **Timelapse monitor stopped**",), daemon=False)
    t.start()
    t.join(timeout=25)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGHUP, _handle_sigterm)
    main()
