"""
app.py
======

The display loop, shared by all four modes.

    view    - just show the camera. No model is loaded at all.
    detect  - YOLO boxes, no memory between frames.
    track   - YOLO + ByteTrack: every person carries a stable ID.
    count   - tracking plus a counting line: entry_count and exit_count.

All four run the same loop; they differ only in what `process()` does to each
frame. That is the point of the pipeline classes - a new mode is a new class,
not another copy of the loop.
"""

import time
from collections import defaultdict, deque

import cv2

from . import config
from .camera import VideoStream, ThreadedVideoStream, open_stream
from .detector import PersonDetector
from .counter import PersonCounter
from .line_setup import resolve_line
from .events import EventLogger
from . import drawing


class FpsMeter:
    """Frames per second, averaged over about one second.

    Timing a single frame is far too noisy to read off the screen, so we count
    frames until a second has passed and divide.
    """

    def __init__(self):
        self.fps = 0.0
        self._frames = 0
        self._started_at = time.time()

    def tick(self):
        self._frames += 1
        elapsed = time.time() - self._started_at
        if elapsed >= 1.0:
            self.fps = self._frames / elapsed
            self._frames = 0
            self._started_at = time.time()
        return self.fps


# ---------------------------------------------------------------------------
# Pipelines - one per mode
# ---------------------------------------------------------------------------

class ViewPipeline:
    """Show the frame untouched. Useful for proving the camera works."""

    name = "Live view"

    def process(self, frame) -> list[str]:
        """Draw on the frame in place; return the HUD lines for this frame."""
        return []


class DetectPipeline:
    """Draw a box around every person. No identity, no memory."""

    name = "Detection"

    def __init__(self, detector: PersonDetector):
        self.detector = detector

    def process(self, frame) -> list[str]:
        detections = self.detector.detect(frame)
        for detection in detections:
            drawing.draw_detection(frame, detection)

        # A PER-FRAME headcount, not a tally: with no tracking, the same person
        # in the next frame is an unrelated detection as far as this mode knows.
        return [f"People in frame: {len(detections)}"]


class TrackPipeline:
    """Draw every person with a stable ID and a motion trail."""

    name = "Tracking (ByteTrack)"

    def __init__(self, detector: PersonDetector, show_trails: bool = True):
        self.detector = detector
        self.show_trails = show_trails

        # Per-ID history of foot points. defaultdict creates the deque on first
        # access, so there is no "if id not in dict" dance. maxlen makes it
        # self-trimming - appending to a full deque drops the oldest point - so
        # memory cannot creep up over hours of video.
        self.trails = defaultdict(lambda: deque(maxlen=config.TRAIL_LENGTH))

        # Every ID ever issued. Its size doubles as a health check: if it climbs
        # much faster than people actually walk past, IDs are churning and any
        # counting built on top would over-count.
        self.seen_ids = set()

    def track_and_draw(self, frame):
        """Track, draw boxes and trails, and hand the detections back.

        Split out from process() so the counting pipeline can reuse all of it
        without inheriting the tracking HUD.
        """
        detections = self.detector.track(frame)

        live_ids = []
        for detection in detections:
            drawing.draw_detection(frame, detection)

            # A detection can arrive before the tracker has confirmed it as a
            # track; those have no ID and simply are not trailed.
            if detection.track_id is None:
                continue

            live_ids.append(detection.track_id)
            self.seen_ids.add(detection.track_id)
            self.trails[detection.track_id].append(detection.foot_point)

            if self.show_trails:
                drawing.draw_trail(frame, self.trails[detection.track_id],
                                   drawing.color_for_id(detection.track_id))

        # Drop the history of people who have left, so `trails` does not grow
        # without bound on a long run.
        for dead_id in set(self.trails) - set(live_ids):
            del self.trails[dead_id]

        return detections, live_ids

    def process(self, frame) -> list[str]:
        _, live_ids = self.track_and_draw(frame)
        return [
            f"Tracking now: {len(live_ids)}",
            f"Unique IDs so far: {len(self.seen_ids)}",
        ]

    def toggle_trails(self):
        self.show_trails = not self.show_trails


class CountPipeline(TrackPipeline):
    """Tracking, plus a counting line that tallies entries and exits."""

    name = "Counting"

    def __init__(self, detector: PersonDetector, line, show_trails: bool = True,
                 source: str | None = None):
        super().__init__(detector, show_trails=show_trails)
        self.counter = PersonCounter(line=line)
        # Every entry/exit is also appended to a daily CSV for later analysis.
        self.events = (EventLogger(source) if (config.SAVE_EVENTS and source)
                       else None)

    def process(self, frame) -> list[str]:
        # Track first, then count. The counter needs the tracked detections -
        # a detection with no ID cannot be counted, because we would have no
        # idea which side it was on a moment ago.
        detections, live_ids = self.track_and_draw(frame)
        for track_id, direction in self.counter.update(detections):
            print(f"[{time.strftime('%H:%M:%S')}] ID {track_id}: {direction.upper()}  "
                  f"(entries {self.counter.entry_count}, exits "
                  f"{self.counter.exit_count}, inside {self.counter.occupancy})")
            if self.events is not None:
                self.events.log(track_id, direction, self.counter)

        # Draw the line ON TOP of the boxes, so the tripwire is never hidden
        # behind somebody standing on it.
        self.counter.line.draw(frame)

        return [
            f"ENTRY: {self.counter.entry_count}",
            f"EXIT:  {self.counter.exit_count}",
            f"Inside: {self.counter.occupancy}    Tracking: {len(live_ids)}",
        ]

    def draw_extras(self, frame):
        """Draw the recent-crossings log down the bottom-left corner."""
        for i, event in enumerate(reversed(self.counter.recent_events)):
            y = frame.shape[0] - 12 - i * 22
            cv2.putText(frame, event, (10, y), drawing.FONT, 0.5,
                        config.HUD_COLOR, 1, cv2.LINE_AA)

    def reset(self):
        self.counter.reset()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _build_pipeline(mode: str, stream: VideoStream, show_trails: bool,
                    redraw_line: bool, line_arg: str | None, headless: bool = False):
    """Create the pipeline for a mode, doing any setup that mode requires.

    Takes the open stream because count mode needs live frames to draw the
    line on. View mode deliberately never loads YOLO - it starts instantly,
    which is what makes it a clean test of the camera by itself.
    """
    if mode == "view":
        return ViewPipeline()

    if mode == "count":
        # Ask for the line BEFORE loading the model: the setup screen only
        # needs video, and this way the operator is not left staring at a black
        # window while PyTorch wakes up.
        line = resolve_line(stream, redraw=redraw_line, line_arg=line_arg,
                            interactive=not headless)
        if line is None:
            return None          # operator cancelled
        detector = PersonDetector()
        return CountPipeline(detector, line, show_trails=show_trails,
                             source=stream.source)

    detector = PersonDetector()
    if mode == "detect":
        return DetectPipeline(detector)
    if mode == "track":
        return TrackPipeline(detector, show_trails=show_trails)

    raise ValueError(f"Unknown mode: {mode!r}. Expected view, detect, track or count.")


def run(mode: str = "count", source: str | None = None, show_trails: bool = True,
        redraw_line: bool = False, line_arg: str | None = None,
        threaded: bool | None = None, headless: bool = False) -> int:
    """Open the source, run the chosen pipeline, display until quit.

    Returns a process exit code: 0 for a clean finish, 1 if the source failed.
    """
    # None lets open_stream choose by source: threaded for live cameras,
    # sequential for files. Passing config.USE_THREADED_CAPTURE straight
    # through would force a FILE onto the threaded reader, whose reconnect
    # logic then reopens the file at its end and loops forever.
    if threaded is None and not config.USE_THREADED_CAPTURE:
        threaded = False
    pipeline = stream = None

    try:
        # open_stream picks the threaded reader for live cameras and the
        # sequential one for files. The `with` block releases either on the way
        # out, including when the body raises - so a crash never leaves the
        # camera holding a dead stream slot open.
        with open_stream(source, threaded=threaded) as stream:
            pipeline = _build_pipeline(mode, stream, show_trails, redraw_line,
                                       line_arg, headless=headless)
            if pipeline is None:
                return 0     # cancelled during line setup; not an error

            window_name = f"Person counter - {pipeline.name}"
            if not headless:
                # WINDOW_NORMAL makes the window resizable; the default
                # WINDOW_AUTOSIZE locks it to the frame size and ignores drags.
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

            keys = ["'q'/Esc quits"]
            if mode in ("track", "count"):
                keys.append("'t' toggles trails")
            if mode == "count":
                keys.append("'r' resets counts")
            if headless:
                print(f"Running [{mode}] headless - Ctrl+C stops.")
            else:
                print(f"Running [{mode}]. " + ", ".join(keys) + ".")

            fps_meter = FpsMeter()

            for frame in stream:
                # The pipeline draws its overlays onto `frame` in place and
                # hands back whatever it wants shown in the corner.
                hud_lines = pipeline.process(frame)
                if headless:
                    # No window: nothing to draw or poll. Ctrl+C ends the run.
                    continue

                fps = fps_meter.tick()
                # FPS is the honest measure of whether the pipeline keeps up
                # with the camera. If it sits far below, drop INFERENCE_SIZE.
                status = [f"FPS: {fps:.1f}"]

                if isinstance(stream, ThreadedVideoStream):
                    # Lag is what the threading actually buys, so show it: how
                    # old the frame being processed is. Sequential reading makes
                    # this climb without limit; here it should sit at roughly
                    # one inference pass and stay there.
                    status.append(
                        f"Lag: {stream.last_frame_age * 1000:.0f} ms   "
                        f"Dropped: {stream.dropped} ({stream.drop_rate:.0f}%)")

                drawing.draw_hud(frame, hud_lines + status)

                if hasattr(pipeline, "draw_extras"):
                    pipeline.draw_extras(frame)

                cv2.imshow(window_name, frame)

                # waitKey does two jobs: it hands a millisecond to OpenCV's GUI
                # event loop so the window can actually repaint (skip it and
                # you get a frozen grey rectangle), and it returns any key
                # pressed in that window. The & 0xFF masks off high bits some
                # platforms set, leaving a plain ASCII code.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):        # 'q' or Esc
                    print("Quit requested.")
                    break
                if key == ord("t") and hasattr(pipeline, "toggle_trails"):
                    pipeline.toggle_trails()
                if key == ord("r") and hasattr(pipeline, "reset"):
                    pipeline.reset()

                # Catch the window's X button. Without this the window would
                # vanish while the loop ran on in the background, holding the
                # camera open with no way to stop it.
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed.")
                    break

    except RuntimeError as error:
        # VideoStream raises this with a troubleshooting checklist when the
        # source will not open. No traceback needed - the message is the point.
        print(error)
        return 1
    finally:
        if not headless:
            cv2.destroyAllWindows()

    _print_summary(pipeline, stream)
    return 0


def _print_summary(pipeline, stream=None):
    """Leave the totals in the terminal, where they outlive the closed window."""
    if isinstance(pipeline, CountPipeline):
        counter = pipeline.counter
        print(f"\nFinal counts - entries: {counter.entry_count}, "
              f"exits: {counter.exit_count}, still inside: {counter.occupancy}")
    elif isinstance(pipeline, TrackPipeline):
        print(f"Done. {len(pipeline.seen_ids)} unique people tracked this session.")
