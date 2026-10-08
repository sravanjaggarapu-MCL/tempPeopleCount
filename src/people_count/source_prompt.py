"""
source_prompt.py
================

Asking the operator which camera to run on, and remembering the answer.

The project is pointed at several RTSP cameras, so hard-coding one in
config.py is no longer the right default. On start-up - unless --source said
otherwise - we print the URL format, list the cameras used before, and wait for
a choice.

The expected format is the standard RTSP one:

    rtsp://<username>:<password>@<host>:<port>/<stream-path>

Nothing here imports OpenCV or YOLO. It is plain stdlib, so the prompt appears
instantly instead of after PyTorch has finished waking up.

The "@" in a password
---------------------
The single most common way to get this wrong. RTSP separates the credentials
from the host with "@", so a password that CONTAINS one - "Admin@0192" - splits
the URL in the wrong place and FFmpeg reports a DNS failure for a host name
that is really half a password. Whatever is typed here goes through
normalise_url(), which percent-encodes the userinfo so the URL means what the
operator intended. See config.build_rtsp_url for the same problem in the
hard-coded path.
"""

import json
import os
import re
import sys
from urllib.parse import quote, urlsplit, urlunsplit

from . import config

# Schemes that mean "something is pushing frames at us in real time". Anything
# else typed at the prompt is treated as a path to a video file.
LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")

FORMAT_HINT = "rtsp://<username>:<password>@<host>:<port>/<stream-path>"
EXAMPLE_URL = ("rtsp://admin:Admin%400192@172.20.100.138:554"
               "/cam/realmonitor?channel=1&subtype=1")


# ---------------------------------------------------------------------------
# Parsing and tidying what was typed
# ---------------------------------------------------------------------------

def is_live_url(source: str) -> bool:
    return source.lower().startswith(LIVE_SCHEMES)


def normalise_url(url: str) -> str:
    """Percent-encode the username and password inside an RTSP URL.

    urlsplit splits the credentials off at the LAST "@", so it reads
    "rtsp://admin:Admin@0192@host:554/..." the way a human does. FFmpeg splits
    at the FIRST one and gets it wrong. Re-encoding here means both agree.

    safe="%" leaves an already-encoded password ("Admin%400192") untouched
    instead of double-encoding it into "Admin%2540192".
    """
    parts = urlsplit(url)
    if not parts.username and not parts.password:
        return url

    user = quote(parts.username or "", safe="%")
    password = quote(parts.password or "", safe="%")

    host = parts.hostname or ""
    if ":" in host:                      # bare IPv6 literal needs its brackets
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"

    credentials = user if password == "" else f"{user}:{password}"
    netloc = f"{credentials}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def mask(url: str) -> str:
    """Replace the password with **** so a URL is safe to print."""
    parts = urlsplit(url)
    if not parts.password:
        return config.mask_url(url)      # covers the config.py password too
    return url.replace(f":{parts.password}@", ":****@", 1)


def parse_source(text: str) -> str:
    """Turn what the operator typed into a source the camera module can open.

    Accepts, in this order:
      * a full URL          rtsp://admin:pass@10.0.0.5:554/stream1
      * a host or host:port  10.0.0.5  /  10.0.0.5:554
        filled out with the username, password and stream path from config.py,
        which is the common case when several identical cameras sit on one
        network
      * a path to a video file

    Raises ValueError with a specific explanation if none of those fit.
    """
    text = text.strip().strip('"').strip("'")
    if not text:
        raise ValueError("Nothing entered.")

    if is_live_url(text):
        parts = urlsplit(text)
        if not parts.hostname:
            raise ValueError(f"No host in that URL. Expected {FORMAT_HINT}")
        if not parts.path and parts.scheme.startswith("rtsp"):
            raise ValueError(
                "That URL has no stream path. Most cameras need one - for "
                "example /cam/realmonitor?channel=1&subtype=1 (Dahua/Amcrest) "
                "or /Streaming/Channels/101 (Hikvision).")
        return normalise_url(text)

    # A bare host or host:port - fill in the rest from config.py.
    if _looks_like_host(text) and not os.path.exists(text):
        host, _, port = text.partition(":")
        url = (f"rtsp://{quote(config.USERNAME, safe='')}:"
               f"{quote(config.PASSWORD, safe='')}@{host}:{port or config.PORT}"
               f"{config.STREAM_PATH}")
        print(f"  Read as {mask(url)}")
        print("  (credentials from .env, stream path from configs/config.yaml)")
        return url

    if os.path.exists(text):
        return text

    raise ValueError(
        f"Not a URL this can use, and no file at that path.\n"
        f"  Expected {FORMAT_HINT}")


def _looks_like_host(text: str) -> bool:
    """Is this a bare host or host:port rather than a typo?

    Deliberately strict. "10.0.0.5", "cam1.local" and "cam1:554" are cameras;
    a lone word like "camra" is far more likely a mistyped file name, and
    accepting it would trade a clear message here for a puzzling connection
    timeout thirty seconds later. So a bare name must carry either a dot or an
    explicit port to be read as a host.
    """
    if not re.fullmatch(r"[A-Za-z0-9.\-]+(:\d{1,5})?", text):
        return False
    host, _, port = text.partition(":")
    return bool(port) or ("." in host and not host.endswith("."))


# ---------------------------------------------------------------------------
# The remembered list of cameras
# ---------------------------------------------------------------------------

def load_cameras() -> list[str]:
    """The URLs used before, newest first. Missing or corrupt file -> []."""
    if not os.path.exists(config.CAMERAS_FILE):
        return []
    try:
        with open(config.CAMERAS_FILE) as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        print(f"Ignoring unreadable {config.CAMERAS_FILE}: {error}")
        return []
    return [url for url in data.get("cameras", []) if isinstance(url, str)]


def remember_camera(source: str):
    """Put this camera at the top of the list for next time.

    Only live URLs are remembered - a one-off video file in the list would be
    noise. The file holds passwords in clear text, exactly as config.py always
    has, so it belongs beside the code and not in a shared repository.
    """
    if not is_live_url(source):
        return

    cameras = [url for url in load_cameras() if url != source]
    cameras.insert(0, source)
    try:
        with open(config.CAMERAS_FILE, "w") as handle:
            json.dump({"cameras": cameras[:config.MAX_REMEMBERED_CAMERAS]},
                      handle, indent=2)
    except OSError as error:
        # Not being able to remember the camera is no reason to refuse to run.
        print(f"Could not save {config.CAMERAS_FILE}: {error}")


def describe(url: str) -> str:
    """A short one-line label for the menu: host, port and stream path."""
    parts = urlsplit(url)
    host = parts.hostname or url
    port = f":{parts.port}" if parts.port else ""
    path = (parts.path or "") + (f"?{parts.query}" if parts.query else "")
    user = f"{parts.username}@" if parts.username else ""
    return f"{user}{host}{port}{path}"


# ---------------------------------------------------------------------------
# The prompt itself
# ---------------------------------------------------------------------------

def prompt_for_source() -> str | None:
    """Ask which camera to run on. Returns the source, or None if cancelled.

    With no terminal to read from - launched from a scheduler, or piped - there
    is nobody to ask, so we fall back to the camera in config.py rather than
    blocking forever on stdin.
    """
    if not sys.stdin or not sys.stdin.isatty():
        default = config.build_rtsp_url()
        print(f"No terminal to prompt on - using the default camera (configs/config.yaml + .env): "
              f"{mask(default)}")
        return default

    cameras = load_cameras()

    print()
    print("=" * 68)
    print(" WHICH CAMERA?")
    print("=" * 68)
    print(f" Format:  {FORMAT_HINT}")
    print(f" Example: {EXAMPLE_URL}")
    print()
    if cameras:
        print(" Cameras used before:")
        for i, url in enumerate(cameras, start=1):
            print(f"   [{i}] {describe(url)}")
        print()
    print(" Or:  a bare IP (10.0.0.5) to reuse the login from .env")
    print("      a video file path (sample.mp4) to run on a recording")
    print(f"      [d] the default camera ({config.HOST})")
    print("      [q] quit")
    print("=" * 68)

    while True:
        try:
            answer = input("Camera > ").strip()
        except EOFError:
            # stdin closed mid-prompt - treat it as a cancel, not a crash.
            print()
            return None

        if answer.lower() in ("q", "quit", "exit"):
            return None
        if answer.lower() in ("", "d", "default"):
            return config.build_rtsp_url()

        # A number picks from the remembered list.
        if answer.isdigit() and cameras:
            index = int(answer)
            if 1 <= index <= len(cameras):
                return cameras[index - 1]
            print(f"  There are only {len(cameras)} saved cameras.")
            continue

        try:
            return parse_source(answer)
        except ValueError as error:
            print(f"  {error}")


def resolve_source(source: str | None, ask: bool = True) -> str | None:
    """Work out the source for this run: --source, the prompt, or config.py.

    Returns None only when the operator cancelled at the prompt.
    """
    if source:
        # Run the typed-on-the-command-line path through the same tidying, so
        # --source and the prompt behave identically.
        try:
            return parse_source(source)
        except ValueError as error:
            raise SystemExit(f"--source: {error}")

    if not ask:
        return config.build_rtsp_url()

    chosen = prompt_for_source()
    if chosen is not None:
        remember_camera(chosen)
    return chosen
