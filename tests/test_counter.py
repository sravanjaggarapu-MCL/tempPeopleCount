"""
Synthetic tests for the counting logic - known answers, no camera, no model.

    pytest -q
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest  # noqa: E402

from people_count import config  # noqa: E402
from people_count.counter import PersonCounter  # noqa: E402
from people_count.counting_line import CountingLine  # noqa: E402


class FakeDetection:
    """Just enough of detector.Detection for the counter."""

    def __init__(self, track_id, x, y):
        self.track_id = track_id
        self.foot_point = (int(x), int(y))


@pytest.fixture(autouse=True)
def fixed_config(monkeypatch):
    monkeypatch.setattr(config, "DEAD_ZONE_PX", 12)
    monkeypatch.setattr(config, "MIN_TRACK_AGE_FRAMES", 3)


def horizontal_line():
    # y = 240 across the frame. Dragged left->right, positive side is y > 240
    # (downwards). entry_side=+1 -> walking DOWN the image is an ENTRY.
    return CountingLine(0, 240, 640, 240, entry_side=1)


def walk(counter, track_id, points):
    for x, y in points:
        counter.update([FakeDetection(track_id, x, y)])


def test_clean_entry():
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, y) for y in range(150, 330, 10)])
    assert (counter.entry_count, counter.exit_count) == (1, 0)


def test_clean_exit():
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, y) for y in range(330, 150, -10)])
    assert (counter.entry_count, counter.exit_count) == (0, 1)


def test_flipped_line_swaps_direction():
    line = horizontal_line()
    line.flip()
    counter = PersonCounter(line)
    walk(counter, 1, [(320, y) for y in range(150, 330, 10)])
    assert (counter.entry_count, counter.exit_count) == (0, 1)


def test_in_and_out_again():
    counter = PersonCounter(horizontal_line())
    path = [(320, y) for y in range(150, 330, 10)] + \
           [(320, y) for y in range(330, 150, -10)]
    walk(counter, 1, path)
    assert (counter.entry_count, counter.exit_count) == (1, 1)
    assert counter.occupancy == 0


def test_loitering_on_line_with_jitter_never_counts():
    random.seed(0)
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, 200)] * 5)  # clearly outside first
    walk(counter, 1, [(320, 240 + random.randint(-8, 8)) for _ in range(300)])
    assert counter.entry_count + counter.exit_count == 0


def test_without_dead_zone_jitter_over_counts(monkeypatch):
    """Proves why the dead zone exists."""
    monkeypatch.setattr(config, "DEAD_ZONE_PX", 0)
    random.seed(0)
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, 200)] * 5)
    walk(counter, 1, [(320, 240 + random.choice([-8, 8])) for _ in range(300)])
    assert counter.entry_count + counter.exit_count > 50


def test_hesitate_then_retreat_is_not_counted():
    counter = PersonCounter(horizontal_line())
    path = [(320, y) for y in range(150, 245, 5)] + \
           [(320, y) for y in range(245, 150, -5)]
    walk(counter, 1, path)
    assert counter.entry_count + counter.exit_count == 0


def test_crossing_past_line_end_is_ignored():
    line = CountingLine(200, 240, 440, 240, entry_side=1)   # a doorway
    counter = PersonCounter(line)
    walk(counter, 1, [(600, y) for y in range(150, 330, 10)])  # beside it
    assert counter.entry_count + counter.exit_count == 0


def test_brand_new_track_cannot_count():
    counter = PersonCounter(horizontal_line())
    # Born on one side, on the other side on its 2nd frame (< MIN_TRACK_AGE).
    walk(counter, 1, [(320, 200), (320, 280)])
    assert counter.entry_count == 0


def test_two_people_at_once():
    counter = PersonCounter(horizontal_line())
    for step in range(18):
        counter.update([
            FakeDetection(1, 200, 150 + step * 10),   # going in
            FakeDetection(2, 400, 330 - step * 10),   # going out
        ])
    assert (counter.entry_count, counter.exit_count) == (1, 1)


def test_departed_tracks_are_forgotten():
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, 150)] * 3)
    counter.update([])
    assert counter.states == {}


def test_reset_keeps_sides():
    counter = PersonCounter(horizontal_line())
    walk(counter, 1, [(320, y) for y in range(150, 330, 10)])
    counter.reset()
    walk(counter, 1, [(320, 330)] * 3)   # still standing inside
    assert counter.entry_count == 0
