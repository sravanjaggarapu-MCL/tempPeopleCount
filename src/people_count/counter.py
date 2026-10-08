"""
counter.py
==========

Turns tracked people into entry_count and exit_count.

The rule, stated plainly
------------------------
For each tracked person we remember the last side of the line they were seen
CLEARLY on - "clearly" meaning fully outside the dead-zone band. When they are
next seen clearly on the OTHER side, that is one crossing, and its direction
tells us whether it was an entry or an exit.

Why it is written as a confirmed-side state machine
---------------------------------------------------
The obvious implementation compares each frame's side with the previous
frame's side and counts whenever they differ. That implementation is broken,
and it breaks precisely where it matters most - at the line.

ByteTrack is resilient, but the underlying YOLO box still breathes by a few
pixels every frame: an arm swings, a leg occludes, the box edge moves, and the
foot point moves with it. A person pausing on the tripwire therefore produces a
foot point that flickers across it - left, right, left, right - and a
frame-to-frame comparison books a count for every flicker. One person standing
in a doorway can add dozens of entries.

The fix is to make the sides sticky, with a band of "no opinion" between them:

    |<-- side -1 -->|<-- dead zone -->|<-- side +1 -->|
                     no opinion here

  * Outside the band, we record the side as CONFIRMED.
  * Inside the band, we return "no opinion" and change nothing at all - the
    confirmed side simply stays whatever it last was.
  * A count happens only when a new CONFIRMED side differs from the stored one.

So jitter within the band is structurally incapable of producing a count, no
matter how violent it is. To register, a person must travel completely clear of
the band on the far side - which is exactly what walking through a doorway
does, and exactly what standing on the threshold does not.

The second guard: track age
---------------------------
A track that has existed for only a frame or two is not yet trustworthy - it
may be a flickering false positive, or a real person whose ID was just reissued
mid-stride after an occlusion. Either way its first "crossing" is an artefact.
So a track must survive MIN_TRACK_AGE_FRAMES before its crossings count. Note
that we still update its confirmed side during that time, so it starts counting
from the correct side once it has earned the right to.
"""

from dataclasses import dataclass, field

from . import config
from .counting_line import CountingLine


@dataclass
class TrackState:
    """What we remember about one tracked person, for counting purposes."""

    # The last side this person was seen CLEARLY on: +1, -1, or None if they
    # have only ever been seen inside the dead zone. Crucially this is NOT
    # "which side they were on last frame" - it only changes when they are well
    # clear of the line, which is what makes jitter harmless.
    confirmed_side: int | None = None

    # Frames this track has been alive, for the minimum-age guard.
    age: int = 0

    # Crossings this person has been credited with. Kept for debugging: a
    # single ID racking up many crossings means the dead zone is too narrow for
    # how much the boxes jitter at that spot.
    crossings: int = 0


@dataclass
class PersonCounter:
    """Counts entries and exits of people crossing a CountingLine."""

    line: CountingLine
    entry_count: int = 0
    exit_count: int = 0

    # Per-track memory, keyed by ByteTrack's ID.
    states: dict[int, TrackState] = field(default_factory=dict)

    # The most recent crossings, for the on-screen event log. A plain list we
    # trim, since it is only ever a handful of entries.
    recent_events: list[str] = field(default_factory=list)

    def update(self, detections) -> list[tuple[int, str]]:
        """Feed one frame's tracked detections in; get this frame's crossings.

        Returns a list of (track_id, "entry"/"exit") for anything that crossed
        on this frame - usually empty, occasionally one item.
        """
        events = []
        live_ids = set()

        for detection in detections:
            # Detections the tracker has not confirmed into a track carry no ID
            # and cannot be counted - we have no way to know where they were
            # before, which is the entire basis of a directional check.
            if detection.track_id is None:
                continue

            live_ids.add(detection.track_id)
            state = self.states.setdefault(detection.track_id, TrackState())
            state.age += 1

            # THE directional test. We use the foot point rather than the box
            # centroid: on a bullet camera mounted above head height the body
            # leans into the frame, so the centroid reaches the line while the
            # person is still short of the doorway. The feet are where the
            # person actually is.
            side = self.line.side_of(detection.foot_point)

            # 0 means inside the dead zone: no opinion, change nothing. This
            # single early return is what makes the whole scheme jitter-proof.
            if side == 0:
                continue

            # First clear sighting: record where they came from, count nothing.
            # Someone who walks into view already inside has not entered.
            if state.confirmed_side is None:
                state.confirmed_side = side
                continue

            if side == state.confirmed_side:
                continue  # still on the same side, nothing to do

            # They are now clearly on the other side: a real crossing.
            crossed_into_entry_side = (side == self.line.entry_side)

            if state.age >= config.MIN_TRACK_AGE_FRAMES:
                direction = "entry" if crossed_into_entry_side else "exit"
                if crossed_into_entry_side:
                    self.entry_count += 1
                else:
                    self.exit_count += 1

                state.crossings += 1
                events.append((detection.track_id, direction))
                self._log(f"ID {detection.track_id}: {direction.upper()}")
            else:
                # Too young to trust. We deliberately still fall through to
                # update confirmed_side below, so that once this track matures
                # it is measuring from the right side.
                self._log(f"ID {detection.track_id}: crossing ignored (new track)")

            state.confirmed_side = side

        # Forget people who have left, so `states` cannot grow without bound
        # over a long run. Their counts are already banked in the totals.
        for dead_id in set(self.states) - live_ids:
            del self.states[dead_id]

        return events

    def _log(self, message: str):
        self.recent_events.append(message)
        del self.recent_events[:-config.EVENT_LOG_LENGTH]

    @property
    def occupancy(self) -> int:
        """People currently inside, assuming everyone was outside at startup.

        Treat this as indicative, not authoritative: anyone already inside when
        the script started was never counted in, so the figure can legitimately
        go negative. Reset the counts when the space is known to be empty and
        it becomes meaningful.
        """
        return self.entry_count - self.exit_count

    def reset(self):
        """Zero the totals, keeping the line and the live tracks as they are."""
        self.entry_count = 0
        self.exit_count = 0
        self.recent_events.clear()
        # Each track's confirmed_side is intentionally kept: those people are
        # still standing where they are, and forgetting their side would let
        # the next frame look like a fresh crossing.
        for state in self.states.values():
            state.crossings = 0
        print("Counts reset.")
