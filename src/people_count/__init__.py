"""
people_count
============

The pipeline package. Importing anything from it first runs this file, which is
the one reliable place to set FFmpeg's options - see below.

Modules:
    config.py         - settings from configs/config.yaml + credentials from .env
    camera.py         - opening and reading the video source
    detector.py       - YOLO detection and ByteTrack tracking
    counting_line.py  - the tripwire geometry, dead zone, save/load
    counter.py        - entry/exit tallying (the crossing state machine)
    events.py         - CSV log of every entry/exit
    line_setup.py     - click-and-drag screen for drawing the line
    source_prompt.py  - the "which camera?" prompt
    drawing.py        - all the on-screen overlays
    app.py            - the display loop that ties them together
"""

import os

# ---------------------------------------------------------------------------
# FFmpeg transport setting  (MUST run before anything imports cv2)
# ---------------------------------------------------------------------------
# RTSP can carry its video over either UDP or TCP:
#
#   UDP - the default. Lost packets are gone forever, showing up as grey
#         blocks, tearing and stutter. Many corporate networks also block the
#         random high-numbered UDP ports that RTSP negotiates.
#   TCP - every packet is acknowledged and resent if lost, so the picture stays
#         clean, and it reuses the same port 554 connection that firewalls
#         already allow. Marginally higher latency, far more reliable.
#
# OpenCV exposes FFmpeg's options only through this environment variable, in
# "key;value" form (multiple options separated by "|").
#
# The placement is the subtle part: FFmpeg reads this variable when the cv2
# module loads, so setting it afterwards does nothing at all. Putting it in the
# package's __init__ guarantees it runs before any submodule can import cv2,
# because Python always executes __init__.py first. That is why the whole
# project imports OpenCV through this package rather than directly.
#
# setdefault, not assignment: if you have exported your own value in the shell,
# yours wins.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

__version__ = "1.0.0"
