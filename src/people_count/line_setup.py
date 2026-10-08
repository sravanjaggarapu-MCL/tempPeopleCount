"""
line_setup.py
=============

The interactive screen where the operator drags out the counting line.

It runs on live video, so you can watch people actually walk through the scene
while you place the tripwire - far easier than guessing coordinates and far
easier than placing it on a frozen frame that happens to be empty.

Controls
--------
    click and drag   draw the line
    f                flip which side counts as ENTRY (watch the green arrow)
    r                start over
    Enter / c        confirm and begin counting
    q / Esc          cancel

A note on mouse coordinates
---------------------------
OpenCV reports mouse positions in the coordinates of the DISPLAYED image, which
is not the same thing as the frame's own pixels once a window has been scaled.
Getting this wrong puts the line somewhere other than where it was drawn, and
the error is proportional to the zoom - so it looks almost right at first
glance, which is the worst kind of bug.

Two things keep it honest here:

  * the window is WINDOW_AUTOSIZE, so its size always equals the image we hand
    it - no user resizing can introduce a scale factor behind our back;
  * we do the scaling ourselves, by a factor we chose, and divide it back out
    of every mouse coordinate.

So the line is always stored in the frame's own pixel coordinates, whatever
size the setup window happens to be.
"""

import os

import cv2

from . import config
from .counting_line import CountingLine, line_path_for

WINDOW_NAME = "Draw the counting line"


class _DragState:
    """Mouse state, owned by the callback and read by the draw loop."""

    def __init__(self, scale: float):
        self.scale = scale
        self.start: tuple[int, int] | None = None   # frame coords
        self.end: tuple[int, int] | None = None     # frame coords
        self.dragging = False

    def to_frame_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert a displayed-window position into frame pixels."""
        return int(x / self.scale), int(y / self.scale)

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            # A fresh press always starts a new line - no need to press 'r'
            # first, which is what an operator expects from a drawing tool.
            self.start = self.to_frame_coords(x, y)
            self.end = self.start
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = self.to_frame_coords(x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = self.to_frame_coords(x, y)
            self.dragging = False


def _draw_instructions(display, lines: list[str]):
    """Draw the help text on a translucent panel so it stays readable."""
    panel_height = 24 * len(lines) + 12

    # Blend a dark rectangle under the text rather than drawing it solid: the
    # video stays visible through it, which matters when the operator is trying
    # to line the tripwire up with something underneath.
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    for i, text in enumerate(lines):
        cv2.putText(display, text, (12, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_line_interactively(stream, existing: CountingLine | None = None) -> CountingLine | None:
    """Show live video and let the operator drag out a line.

    Returns the CountingLine, or None if they cancelled.
    """
    # Scale the video up to a comfortable working size. Our sub-stream is only
    # 640x480, and placing a line accurately in a small window is fiddly.
    scale = config.SETUP_WINDOW_WIDTH / max(stream.width, 1)
    state = _DragState(scale)

    # Pre-load an existing line so "adjust the line slightly" does not mean
    # redrawing it from nothing.
    line = existing
    if existing is not None:
        state.start = (existing.x1, existing.y1)
        state.end = (existing.x2, existing.y2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, state.on_mouse)

    print("Draw the counting line: click and drag. "
          "'f' flips entry direction, 'r' resets, Enter confirms, Esc cancels.")

    try:
        for frame in stream:
            display = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)

            # Rebuild the line object from the current drag, keeping whatever
            # entry_side has been chosen so far.
            if state.start and state.end:
                length_ok = (abs(state.end[0] - state.start[0]) > config.MIN_LINE_LENGTH
                             or abs(state.end[1] - state.start[1]) > config.MIN_LINE_LENGTH)
                if length_ok:
                    line = CountingLine(
                        state.start[0], state.start[1],
                        state.end[0], state.end[1],
                        entry_side=line.entry_side if line else 1,
                    )
                else:
                    # Too short to be meaningful - a stray click rather than a
                    # deliberate drag. The normal of a near-zero-length line is
                    # numerically meaningless, so we refuse to build one.
                    line = None

            # Draw the line onto the SCALED display. The line lives in frame
            # coordinates, so we scale a copy for drawing rather than storing
            # display coordinates anywhere.
            if line is not None:
                scaled = CountingLine(
                    int(line.x1 * scale), int(line.y1 * scale),
                    int(line.x2 * scale), int(line.y2 * scale),
                    line.entry_side,
                )
                scaled.draw(display)

            if line is None:
                help_lines = ["Click and drag to draw the counting line",
                              "Esc cancels"]
            else:
                help_lines = [
                    "Drag again to redraw  |  'f' flips the ENTRY arrow",
                    "Enter or 'c' to confirm  |  'r' clears  |  Esc cancels",
                ]
            _draw_instructions(display, help_lines)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):                    # q or Esc
                print("Line setup cancelled.")
                return None
            if key in (13, 10, ord("c")) and line is not None:   # Enter / c
                return line
            if key == ord("r"):
                state.start = state.end = None
                line = None
            if key == ord("f") and line is not None:
                line.flip()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Line setup window closed.")
                return None
    finally:
        # Always take the setup window down, including on the cancel path -
        # otherwise it lingers as a dead frame beside the counting window.
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)   # let the GUI process the destroy before we move on

    return None


def resolve_line(stream, redraw: bool = False, line_arg: str | None = None,
                 interactive: bool = True) -> CountingLine | None:
    """Get the counting line: from the flag, from disk, or by asking.

    Order of preference:
      1. --line x1,y1,x2,y2 passed on the command line (never saved, since it
         is explicitly a one-off)
      2. the line saved for THIS camera, unless --redraw-line was given
      3. the interactive drawing screen

    The line is stored per camera - see line_path_for - so switching cameras
    asks for a new line instead of reusing one drawn somewhere else.
    """
    if line_arg:
        try:
            x1, y1, x2, y2 = (int(value) for value in line_arg.split(","))
        except ValueError:
            raise SystemExit(f"--line needs four integers: x1,y1,x2,y2 (got {line_arg!r})")
        print(f"Using line from --line: ({x1},{y1}) -> ({x2},{y2})")
        return CountingLine(x1, y1, x2, y2)

    frame_size = (stream.width, stream.height)
    path = line_path_for(stream.source)

    if not redraw:
        saved = CountingLine.load(path, frame_size=frame_size)
        if saved is None:
            saved = _load_legacy_line(frame_size, path)
        if saved is not None:
            print(f"Loaded saved line from {path} "
                  f"({saved.x1},{saved.y1}) -> ({saved.x2},{saved.y2}). "
                  "Use --redraw-line to change it.")
            return saved
        print(f"No line saved for this camera yet ({path}) - draw one.")

    if not interactive:
        raise SystemExit(
            "Headless mode cannot draw a line. Run once with a window to draw "
            "and save it, or pass --line x1,y1,x2,y2.")

    line = draw_line_interactively(stream)
    if line is not None:
        line.save(path, frame_size=frame_size)
    return line


def _load_legacy_line(frame_size, new_path: str) -> CountingLine | None:
    """Adopt the old single-file line, ONCE, if it fits this stream.

    Before the per-camera split every run shared counting_line.json, so a line
    drawn then belongs to exactly one camera - we just cannot tell which. The
    first camera to run is the best guess, and it is checked against the saved
    resolution as load() has always done.

    The file is then renamed, and that rename is the important half: without it
    the next camera up would adopt the same line, which is precisely the
    silent "camera A's doorway over camera B's car park" this split exists to
    prevent. One adoption, then the old file is out of the way for good.
    """
    if not os.path.exists(config.LINE_FILE):
        return None

    line = CountingLine.load(config.LINE_FILE, frame_size=frame_size)
    if line is None:
        return None

    line.save(new_path, frame_size=frame_size)

    retired = config.LINE_FILE + ".migrated"
    try:
        os.replace(config.LINE_FILE, retired)
    except OSError as error:
        print(f"Could not rename {config.LINE_FILE}: {error}")

    print(f"Adopted the line from {config.LINE_FILE} for this camera "
          f"(the old file is now {retired}).")
    print("If that line was drawn for a different camera, run again with "
          "--redraw-line.")
    return line
