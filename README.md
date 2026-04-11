# Lightburn Timelapse

A robust, automated timelapse capture and notification system for LightBurn-driven laser engravers (e.g., Falcon A1 diodes), designed for seamless operation on a NAS or Docker environment. It integrates with a network camera and provides live monitoring, session logging, timelapse assembly, and Discord notifications.

---

## Features

- **Automatic activity detection** via camera feed
- **Session-based timelapse**: records upon activity/start, stops on inactivity
- **Web UI** for monitoring, session progress, and live feed viewing
- **Discord notifications**: session start, progress snapshots, completion/discard
- **API** for manual session control and configuration
- **Zero-downtime hot-reload**: update Python code in-place during operation
- **Crash recovery**: auto-rebuild videos from incomplete sessions on restart

---

## Hardware/Network Setup

- **NAS:** e.g., Ubuntu server, running Docker
- **Laser:** LightBurn-controlled (tested: Falcon A1)
- **Camera:** Network MJPEG stream (e.g., Pi camera via `mjpeg-streamer`)
  - Example: `http://192.168.0.142:8080/?action=snapshot`

---

## File Structure

```
/home/andrew/lightburn_timelapse/
├── timelapse.py            # main script
├── docker-compose.yml      # orchestrates all containers
├── .env                    # Discord webhook secret
├── timelapse_sessions/     # output
│   └── session_YYYYMMDD_HHMMSS/    # session dirs
│       ├── frame_00000.jpg
│       ├── ...
│       ├── session_YYYYMMDD_HHMMSS.mp4
│       └── meta.json
└── ... (see AGENTS.md for details)
```

---

## Quick Start (Docker)

1. **Dependencies**: Docker, Docker Compose
2. **Configuration**:
    - Place your Discord webhook in `.env` as `DISCORD_WEBHOOK_URL=<url>`
    - Adjust settings in `timelapse.py` as needed (see top of script)
3. **Run**:
    ```bash
    docker compose up -d
    ```
4. **Monitor**:
    - Access web UI at: `http://nas.lan:8181`
    - Sessions served at: `http://nas.lan:8181/sessions/`

---

## Session Workflow

1. **Watching**: Monitors camera feed, waits for sustained activity.
2. **Recording**: On activity, starts new session, saves full-res frames.
3. **Idle Detection**: After sufficient inactivity, assembles frames to timelapse video (mp4).
4. **Notifications**: Discord updates at key points; video sent or linked after completion.
5. **Discard/Recovery**: Short/failed sessions are discarded or rebuilt on next launch.

---

## API

Routes proxied at `/api/` via nginx container for triggering sessions, changing config, etc.

- Start/stop session
- Enable/disable detection
- Status, config, logs

See AGENTS.md for full endpoint list and schemas.

---

## Hot Reload & Crash Recovery

- **Touch `reload.flag`** to reload code with no session loss.
- Incomplete sessions restored on restart if possible (`recovered: true` in meta.json).

---

## Configuration

Adjust parameters in `timelapse.py`, for example:
- `SNAPSHOT_URL`: Camera endpoint
- `INTERVAL_SEC`: Seconds between frames (default: 5)
- `DIFF_THRESHOLD`, `START_THRESHOLD`: Sensitivity tuning
- `MIN_SESSION_SEC`: Discards sessions below this active time
- See project notes for advanced tuning.

---

## Advanced Usage

- **Multi-version script management**: Archive current script as e.g. `timelapse_v3.py` before hot reload
- **Discord integration**: Sends live snapshots and video links; splits upload if >25MB
- **Web UI**: view live progress, set reference session, and download videos

---

## Credits

Written and maintained by Andrew. See AGENTS.md for full project history and design insights.
