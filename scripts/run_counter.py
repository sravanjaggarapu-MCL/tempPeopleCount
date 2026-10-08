"""
scripts/run_counter.py
======================

The single entry point for the project. Run this file - nothing else.

    python scripts/run_counter.py                      # asks which camera, then counts
    python scripts/run_counter.py --redraw-line        # draw a new line over the saved one
    python scripts/run_counter.py --mode view          # just the video, no model at all
    python scripts/run_counter.py --mode detect        # boxes only, no IDs
    python scripts/run_counter.py --mode track         # boxes + stable IDs, no counting

    python scripts/run_counter.py --source rtsp://admin:pass@10.0.0.5:554/stream1
    python scripts/run_counter.py --source sample.mp4  # run against a recorded clip instead
    python scripts/run_counter.py --no-prompt          # skip the question, use the default camera
    python scripts/run_counter.py --imgsz 320          # faster, slightly less accurate
    python scripts/run_counter.py --conf 0.3           # override the confidence threshold
    python scripts/run_counter.py --dead-zone 20       # widen the ignore band at the line
    python scripts/run_counter.py --line 0,300,640,300 # set the line from the command line
    python scripts/run_counter.py --no-threaded        # read frames sequentially (debugging)

On start-up you are asked which camera to run on, in the format

    rtsp://<username>:<password>@<host>:<port>/<stream-path>

Cameras you have used before are listed to pick by number, and --source skips
the question entirely. Each camera keeps its own counting line under lines/.

Controls in the window: 'q'/Esc quits, 't' toggles trails, 'r' resets counts.
While drawing the line: drag to draw, 'f' flips the entry direction, 'r'
clears, Enter confirms.

Everything configurable lives in configs/config.yaml (camera passwords in
.env); the flags here are for trying a value without editing the file.

    python scripts/run_counter.py --headless --no-prompt   # server / no screen
"""

import argparse
import os
import sys

# Let the script run straight from a clone, without `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

# Importing the package runs people_count/__init__.py, which sets FFmpeg's
# RTSP transport option. That has to happen before anything imports cv2, which
# is why this import comes before `from people_count.app import run`.
import people_count  # noqa: E402,F401
from people_count import config  # noqa: E402
from people_count.app import run  # noqa: E402
from people_count.source_prompt import resolve_source, mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person detection, tracking and line counting on an RTSP camera.",
        # RawDescriptionHelpFormatter keeps our line breaks in the epilog
        # instead of reflowing it into one paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  view    show the camera only - the quickest way to prove the\n"
            "          stream works, since no model is loaded\n"
            "  detect  draw a box around each person, frame by frame\n"
            "  track   as detect, plus a stable ID per person (ByteTrack)\n"
            "  count   as track, plus a counting line -> entries and exits\n"
            "\n"
            "the counting line:\n"
            "  On first run you draw it by dragging across the video, and it\n"
            "  is saved under outputs/lines/ for every run after that. The\n"
            "  green arrow shows which way counts as an ENTRY - press 'f'\n"
            "  during setup if it points the wrong way.\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["view", "detect", "track", "count"], default="count",
        help="what to run (default: count)",
    )
    parser.add_argument(
        "--source", default=None,
        help="RTSP URL or path to a video file. Given, it skips the start-up "
             "question; omitted, you are asked which camera to use",
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="do not ask which camera - use the default in configs/config.yaml. For running "
             "unattended from a scheduler or a script",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="confidence threshold override; see configs/config.yaml for why detect and "
             "track use different defaults",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help=f"inference size, multiple of 32 (default: {config.INFERENCE_SIZE})",
    )
    parser.add_argument(
        "--no-trails", action="store_true",
        help="hide the motion trails in track and count modes",
    )
    parser.add_argument(
        "--redraw-line", action="store_true",
        help="draw a new counting line, replacing the saved one",
    )
    parser.add_argument(
        "--line", default=None, metavar="x1,y1,x2,y2",
        help="set the counting line from the command line instead of drawing "
             "it; not saved, since it is explicitly a one-off",
    )
    parser.add_argument(
        "--no-threaded", action="store_true",
        help="read frames on the main thread instead of a background ingestion "
             "thread. Live streams default to threaded; files always read "
             "sequentially so no frame is skipped",
    )
    parser.add_argument(
        "--dead-zone", type=int, default=None, metavar="PX",
        help=f"half-width in pixels of the ignore band around the line "
             f"(default: {config.DEAD_ZONE_PX}). Raise it if a person standing "
             f"on the line makes the counts climb",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="no window - just count and write the CSV. Needs a saved line "
             "or --line. For servers and unattended runs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Apply the overrides by writing them back into the config module. Every
    # other module reads its settings through `config.X` at call time rather
    # than copying them at import time, so assigning here reaches all of them -
    # no need to thread each option through half a dozen function signatures.
    if args.imgsz is not None:
        config.INFERENCE_SIZE = args.imgsz
    if args.conf is not None:
        config.DETECT_CONF_THRESHOLD = args.conf
        config.TRACK_CONF_THRESHOLD = args.conf
    if args.dead_zone is not None:
        config.DEAD_ZONE_PX = args.dead_zone

    # Ask which camera before anything heavy happens. --source or --no-prompt
    # skip the question; a cancelled prompt returns None and we simply stop.
    source = resolve_source(args.source, ask=not args.no_prompt)
    if source is None:
        print("No camera chosen - nothing to do.")
        return 0
    print(f"Camera: {mask(source)}")

    return run(
        mode=args.mode,
        source=source,
        show_trails=not args.no_trails,
        redraw_line=args.redraw_line,
        line_arg=args.line,
        # False forces sequential reading; None lets open_stream decide per
        # source, which is what you want almost always.
        threaded=False if args.no_threaded else None,
        headless=args.headless,
    )


if __name__ == "__main__":
    try:
        # sys.exit passes the return value out as the process exit code, so a
        # script or scheduler calling this file can tell success from failure.
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C: exit quietly rather than dumping a traceback.
        print("\nInterrupted by user.")
        sys.exit(0)
