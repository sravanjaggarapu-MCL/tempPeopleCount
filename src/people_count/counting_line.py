"""
counting_line.py
================

The virtual tripwire: the geometry of the line, which side of it a point is on,
and how far away it is.

The maths in one paragraph
--------------------------
A line is defined by two points, A and B. For any third point P, the 2D cross
product

    cross = (B - A) x (P - A)
          = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)

is positive on one side of the line, negative on the other, and exactly zero on
it. That sign is the whole "which side" test. Dividing by the line's length
turns the raw cross product into the perpendicular DISTANCE from the line in
pixels, which is what makes a dead zone measurable in pixels rather than in
arbitrary units.

So one number - the signed distance - answers both questions at once: its sign
says which side, its magnitude says how far. Everything else here is bookkeeping
around that.

Which side is "inside"?
-----------------------
The maths cannot know. It depends entirely on how the operator dragged the line
and where the door is. So the object carries an `entry_side` (+1 or -1) saying
which side counts as INSIDE, the setup screen draws an arrow showing it, and
the operator flips it with a keypress if it points the wrong way.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2

from . import config


@dataclass
class CountingLine:
    """A tripwire between two points, with a notion of which side is inside."""

    x1: int
    y1: int
    x2: int
    y2: int

    # Which side of the line is INSIDE. +1 is the side the cross product calls
    # positive; -1 is the other one. Crossing INTO this side is an ENTRY.
    entry_side: int = 1

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def normal(self) -> tuple[float, float]:
        """Unit vector pointing from the line towards the POSITIVE side.

        Derivation: with d = B - A, take the candidate point P = A + t*(-dy, dx).
        Substituting into the cross product gives t * (dx^2 + dy^2), which is
        positive whenever t is. So (-dy, dx), normalised, is the direction of
        the positive side. (In image coordinates y points down, so this is a
        clockwise perpendicular rather than the anticlockwise one you would get
        on graph paper - which is exactly why we let the operator flip it by eye
        instead of reasoning about it.)
        """
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = self.length or 1.0          # guard against a zero-length line
        return -dy / length, dx / length

    def signed_distance(self, point: tuple[int, int]) -> float:
        """Perpendicular distance from the line, in pixels, carrying a sign.

        Sign = which side. Magnitude = how far. Zero = exactly on the line.
        """
        px, py = point
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        cross = dx * (py - self.y1) - dy * (px - self.x1)
        return cross / (self.length or 1.0)

    def projection_t(self, point: tuple[int, int]) -> float:
        """How far along the line a point falls: 0 at A, 1 at B.

        This is the point projected onto the line, expressed as a fraction of
        the line's length. Values outside 0..1 mean the point is off the end of
        the drawn segment - beside the tripwire rather than on it.
        """
        px, py = point
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return 0.0
        return ((px - self.x1) * dx + (py - self.y1) * dy) / length_squared

    def covers(self, point: tuple[int, int]) -> bool:
        """Is this point within the span of the drawn line, rather than past
        one of its ends?

        The signed distance treats the line as INFINITE - it happily reports
        which side of the extended line a point is on, however far off the end
        of the drawn segment it sits. That is wrong for a tripwire: an operator
        who draws a line across a doorway means that doorway, not the invisible
        continuation of it across the rest of the scene. Without this check,
        somebody walking past at the far side of the frame crosses the
        extension and books a count for a door they never went near.
        """
        return 0.0 <= self.projection_t(point) <= 1.0

    def side_of(self, point: tuple[int, int], dead_zone: float | None = None) -> int:
        """Which side a point is on: +1, -1, or 0 for "no opinion".

        Two distinct things produce a 0, and both mean "change nothing":

        1. The point is inside the dead zone. This is the point of the whole
           module. A person standing on the line produces a foot point that
           wobbles by a few pixels every frame as the box jitters - without a
           dead zone that wobble reads as a stream of genuine crossings, and
           the counter runs away.
        2. The point is off the end of the drawn segment - see covers().
        """
        if not self.covers(point):
            return 0

        if dead_zone is None:
            dead_zone = config.DEAD_ZONE_PX
        distance = self.signed_distance(point)
        if abs(distance) < dead_zone:
            return 0
        return 1 if distance > 0 else -1

    def flip(self):
        """Swap which side counts as inside."""
        self.entry_side = -self.entry_side

    # -----------------------------------------------------------------------
    # Persistence - so the operator draws the line once, not once per run
    # -----------------------------------------------------------------------

    def to_dict(self, frame_size: tuple[int, int] | None = None) -> dict:
        data = {
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "entry_side": self.entry_side,
        }
        # Store the resolution it was drawn at. A line saved against a 640x480
        # sub-stream would sit in the wrong place on a 1920x1080 main stream,
        # and silently counting the wrong doorway is worse than asking again.
        if frame_size:
            data["frame_width"], data["frame_height"] = frame_size
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CountingLine":
        return cls(data["x1"], data["y1"], data["x2"], data["y2"],
                   data.get("entry_side", 1))

    def save(self, path: str | None = None, frame_size: tuple[int, int] | None = None):
        path = path or config.LINE_FILE
        # The per-camera directory will not exist on the very first run.
        # dirname is "" for a bare filename, which makedirs would reject.
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(frame_size), handle, indent=2)
        print(f"Counting line saved to {path}")

    @classmethod
    def load(cls, path: str | None = None, frame_size: tuple[int, int] | None = None):
        """Load a saved line, or return None if there isn't a usable one."""
        path = path or config.LINE_FILE
        if not os.path.exists(path):
            return None

        try:
            with open(path) as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, KeyError, OSError) as error:
            print(f"Ignoring unreadable {path}: {error}")
            return None

        saved_size = (data.get("frame_width"), data.get("frame_height"))
        if frame_size and None not in saved_size and tuple(saved_size) != frame_size:
            print(f"Saved line was drawn for {saved_size[0]}x{saved_size[1]} but this "
                  f"stream is {frame_size[0]}x{frame_size[1]} - redraw it.")
            return None

        return cls.from_dict(data)

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def draw(self, frame, show_dead_zone: bool = True):
        """Draw the tripwire, its dead-zone band, and the entry arrow."""
        if show_dead_zone and config.DEAD_ZONE_PX > 0:
            # The band is the line shifted along its normal in both directions.
            # Drawing it makes the tolerance visible: a person has to clear the
            # whole band before anything is counted, and seeing its width is the
            # easiest way to judge whether DEAD_ZONE_PX needs tuning.
            nx, ny = self.normal
            offset_x = nx * config.DEAD_ZONE_PX
            offset_y = ny * config.DEAD_ZONE_PX
            for sign in (1, -1):
                cv2.line(
                    frame,
                    (int(self.x1 + sign * offset_x), int(self.y1 + sign * offset_y)),
                    (int(self.x2 + sign * offset_x), int(self.y2 + sign * offset_y)),
                    config.DEAD_ZONE_COLOR, 1, cv2.LINE_AA,
                )

        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2),
                 config.LINE_COLOR, 2, cv2.LINE_AA)

        # Small circles mark the endpoints, so a line dragged almost off-screen
        # is still obviously a line the operator placed.
        for point in ((self.x1, self.y1), (self.x2, self.y2)):
            cv2.circle(frame, point, 5, config.LINE_COLOR, -1)

        self._draw_entry_arrow(frame)

    def _draw_entry_arrow(self, frame):
        """Arrow from the line's midpoint pointing the way an ENTRY goes."""
        nx, ny = self.normal
        mid_x, mid_y = self.midpoint
        # Point the arrow towards whichever side is currently "inside".
        tip_x = int(mid_x + nx * self.entry_side * 45)
        tip_y = int(mid_y + ny * self.entry_side * 45)

        cv2.arrowedLine(frame, (mid_x, mid_y), (tip_x, tip_y),
                        config.ENTRY_ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(frame, "ENTRY", (tip_x + 5, tip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.ENTRY_ARROW_COLOR, 1)


# ---------------------------------------------------------------------------
# One saved line per camera
# ---------------------------------------------------------------------------

def line_path_for(source: str) -> str:
    """Where this camera's counting line is stored.

    A line is a position in one camera's picture. Pointed at a second camera,
    the same coordinates land on whatever happens to be there - so each source
    gets its own file, named after the camera rather than numbered, so the
    directory stays readable:

        lines/172-20-100-138-554-cam-realmonitor-channel-1-subtype-1.json
        lines/sample-mp4.json

    The stream path is part of the name on purpose: channel 1 and channel 2 of
    one NVR are different cameras pointing at different places.
    """
    parts = urlsplit(source)
    if parts.hostname:
        key = f"{parts.hostname}-{parts.port or ''}-{parts.path}-{parts.query}"
    else:
        # A video file: its name is enough, and the full path would make for an
        # unreadable file name.
        key = os.path.basename(source) or source

    # Anything that is not a letter or digit becomes a hyphen - dots, slashes,
    # colons and "?" are variously illegal or awkward in a Windows file name.
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-").lower()[:80]
    return os.path.join(config.LINE_DIR, f"{slug or 'camera'}.json")
