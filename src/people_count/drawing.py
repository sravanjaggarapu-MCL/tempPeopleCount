"""
drawing.py
==========

Everything drawn on top of a frame: boxes, labels, trails and the heads-up
display. Keeping it in one module means the display loop in app.py stays about
the pipeline and not about pixel arithmetic.

A reminder that bites everyone at least once: OpenCV colours are (B, G, R),
not (R, G, B), and image coordinates put (0, 0) at the TOP-left with y growing
downwards.
"""

import cv2
import numpy as np

from . import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Give each track ID its own stable, well-separated colour.

    Deterministic, so a person keeps one colour for their whole time on screen
    and you can follow them by eye. Hues are stepped by 47, which shares no
    factor with 180, so consecutive IDs land far apart on the colour wheel
    instead of looking like near-identical shades.

    OpenCV's hue channel runs 0-179 rather than the usual 0-359 - it has to fit
    in a single byte.
    """
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])            # vivid and bright
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])   # cv2 wants plain Python ints


def draw_label(frame, text: str, x: int, y: int, color):
    """Draw text on a filled background block, anchored above the point (x, y).

    The background is what keeps a label readable over a busy scene - plain
    text disappears against anything light.
    """
    # getTextSize returns ((width, height), baseline). The baseline is the few
    # pixels descenders like 'p' hang below the line; we include it as padding
    # so the block never clips them.
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.5, 1)

    # Sit the label above the point, unless it would fall off the top of the
    # frame - then tuck it just below instead so it stays on screen.
    bottom = y if y - text_h - baseline > 0 else y + text_h + baseline

    cv2.rectangle(frame, (x, bottom - text_h - baseline), (x + text_w, bottom),
                  color, -1)  # -1 thickness means filled
    cv2.putText(frame, text, (x, bottom - baseline), FONT, 0.5,
                config.LABEL_TEXT_COLOR, 1)


def draw_detection(frame, detection):
    """Draw one detection: box, confidence label, and the foot point.

    Used by both modes. A detection with a track_id gets its own colour and an
    "ID n" prefix; one without gets plain green.
    """
    color = (config.BOX_COLOR if detection.track_id is None
             else color_for_id(detection.track_id))

    cv2.rectangle(frame, (detection.x1, detection.y1),
                  (detection.x2, detection.y2), color, 2)

    label = (f"person {detection.confidence:.2f}" if detection.track_id is None
             else f"ID {detection.track_id}  {detection.confidence:.2f}")
    draw_label(frame, label, detection.x1, detection.y1, color)

    # The dot where the person meets the floor - see Detection.foot_point for
    # why this, and not the box centre, is the point that will be counted.
    cv2.circle(frame, detection.foot_point, 4, color, -1)


def draw_trail(frame, points, color):
    """Join a person's remembered foot points into a fading polyline.

    Older segments are drawn thinner, so the trail reads as a direction of
    travel at a glance - which is precisely the question entry/exit counting
    has to answer.
    """
    for i in range(1, len(points)):
        thickness = max(1, int(3 * i / len(points)))
        cv2.line(frame, points[i - 1], points[i], color, thickness)


def draw_hud(frame, lines: list[str]):
    """Draw the status lines down the top-left corner of the frame."""
    for i, line in enumerate(lines):
        y = 30 + i * 30
        cv2.putText(frame, line, (10, y), FONT, 0.8, config.HUD_COLOR, 2)
