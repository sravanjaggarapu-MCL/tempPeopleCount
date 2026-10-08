"""
config.py
=========

Every tunable setting, exposed as module attributes (config.X).

Values come from configs/config.yaml. Camera credentials come from environment
variables (or a local .env file) so that NO password is ever committed to git.

Other modules read settings as `config.X` at call time rather than copying them
at import time, so the CLI can override a value by simply assigning to it:

    config.INFERENCE_SIZE = 320
"""

import os
from urllib.parse import quote, urlsplit

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# src/people_count/config.py -> repo root is three levels up.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE = os.environ.get("PEOPLE_COUNT_CONFIG",
                             os.path.join(ROOT_DIR, "configs", "config.yaml"))


def _load_dotenv(path: str):
    """Minimal .env reader (KEY=VALUE per line). Real env vars always win."""
    if not os.path.exists(path):
        return
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(ROOT_DIR, ".env"))

with open(CONFIG_FILE) as _handle:
    _cfg = yaml.safe_load(_handle) or {}


def _get(section: str, key: str, default=None):
    return (_cfg.get(section) or {}).get(key, default)


def _path(value: str) -> str:
    """Resolve a config path relative to the repo root."""
    return value if os.path.isabs(value) else os.path.join(ROOT_DIR, value)


def _color(value, default):
    """Colours are written as [B, G, R] lists in YAML; OpenCV wants a tuple."""
    return tuple(int(c) for c in value) if value else default


# ---------------------------------------------------------------------------
# Camera (fallback when nothing is chosen at the prompt)
# ---------------------------------------------------------------------------
USERNAME = os.environ.get("CAMERA_USERNAME", "admin")
PASSWORD = os.environ.get("CAMERA_PASSWORD", "")
HOST = os.environ.get("CAMERA_HOST", _get("camera", "host", "192.168.1.100"))
PORT = int(os.environ.get("CAMERA_PORT", _get("camera", "port", 554)))
STREAM_PATH = os.environ.get(
    "CAMERA_STREAM_PATH",
    _get("camera", "stream_path", "/cam/realmonitor?channel=1&subtype=1"))

USE_THREADED_CAPTURE = bool(_get("camera", "use_threaded_capture", True))
MAX_RECONNECT_ATTEMPTS = int(_get("camera", "max_reconnect_attempts", 10))
RECONNECT_DELAY_SECONDS = float(_get("camera", "reconnect_delay_seconds", 2.0))
FRAME_TIMEOUT_SECONDS = float(_get("camera", "frame_timeout_seconds", 10.0))

CAMERAS_FILE = _path(_get("camera", "cameras_file", "cameras.json"))
MAX_REMEMBERED_CAMERAS = int(_get("camera", "max_remembered_cameras", 10))


def build_rtsp_url() -> str:
    """The fallback camera URL. Credentials are percent-encoded so a password
    containing '@' does not split the URL in the wrong place."""
    user = quote(USERNAME, safe="")
    password = quote(PASSWORD, safe="")
    credentials = f"{user}:{password}@" if password else (f"{user}@" if user else "")
    return f"rtsp://{credentials}{HOST}:{PORT}{STREAM_PATH}"


def mask_url(url: str) -> str:
    """Hide the password so a URL is safe to print or log."""
    parts = urlsplit(url)
    if parts.password:
        return url.replace(f":{parts.password}@", ":****@", 1)
    return url


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = _path(_get("model", "weights", "weights/yolov8n.pt"))
INFERENCE_SIZE = int(_get("model", "inference_size", 480))
PERSON_CLASS_ID = int(_get("model", "person_class_id", 0))
DETECT_CONF_THRESHOLD = float(_get("model", "detect_conf_threshold", 0.4))
TRACK_CONF_THRESHOLD = float(_get("model", "track_conf_threshold", 0.15))
TRACKER_CONFIG = _path(_get("model", "tracker_config", "configs/bytetrack.yaml"))

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
DEAD_ZONE_PX = int(_get("counting", "dead_zone_px", 12))
MIN_TRACK_AGE_FRAMES = int(_get("counting", "min_track_age_frames", 3))
MIN_LINE_LENGTH = int(_get("counting", "min_line_length", 10))
LINE_DIR = _path(_get("counting", "line_dir", "outputs/lines"))
LINE_FILE = _path(_get("counting", "legacy_line_file", "counting_line.json"))
EVENTS_DIR = _path(_get("counting", "events_dir", "outputs/events"))
SAVE_EVENTS = bool(_get("counting", "save_events", True))

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_WIDTH = int(_get("display", "window_width", 1280))
WINDOW_HEIGHT = int(_get("display", "window_height", 720))
SETUP_WINDOW_WIDTH = int(_get("display", "setup_window_width", 1280))
TRAIL_LENGTH = int(_get("display", "trail_length", 30))
EVENT_LOG_LENGTH = int(_get("display", "event_log_length", 5))

BOX_COLOR = _color(_get("display", "box_color"), (0, 255, 0))
LABEL_TEXT_COLOR = _color(_get("display", "label_text_color"), (0, 0, 0))
HUD_COLOR = _color(_get("display", "hud_color"), (0, 255, 255))
LINE_COLOR = _color(_get("display", "line_color"), (0, 0, 255))
DEAD_ZONE_COLOR = _color(_get("display", "dead_zone_color"), (0, 165, 255))
ENTRY_ARROW_COLOR = _color(_get("display", "entry_arrow_color"), (0, 255, 0))
