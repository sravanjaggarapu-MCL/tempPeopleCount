"""
events.py
=========

Writes every ENTRY / EXIT to a CSV file, so store footfall can be analysed
after the run (hourly traffic, peak times, daily totals).

One file per camera per day:

    outputs/events/<camera-slug>_2026-10-08.csv

    timestamp,camera,track_id,direction,entry_total,exit_total,inside
    2026-10-08T10:15:02,172-20-100-138-554-...,12,entry,1,0,1
"""

import csv
import os
from datetime import datetime

from . import config
from .counting_line import line_path_for

FIELDS = ["timestamp", "camera", "track_id", "direction",
          "entry_total", "exit_total", "inside"]


class EventLogger:
    """Appends crossing events to a daily CSV for one camera."""

    def __init__(self, source: str, directory: str | None = None):
        # Reuse the per-camera slug the line files already use - it carries
        # no password and is readable.
        self.camera = os.path.splitext(os.path.basename(line_path_for(source)))[0]
        self.directory = directory or config.EVENTS_DIR
        os.makedirs(self.directory, exist_ok=True)

    def _path_for(self, when: datetime) -> str:
        return os.path.join(self.directory, f"{self.camera}_{when:%Y-%m-%d}.csv")

    def log(self, track_id: int, direction: str, counter):
        now = datetime.now()
        path = self._path_for(now)
        new_file = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(FIELDS)
                writer.writerow([now.isoformat(timespec="seconds"), self.camera,
                                 track_id, direction, counter.entry_count,
                                 counter.exit_count, counter.occupancy])
        except OSError as error:
            # Failing to log must never stop the counting.
            print(f"Could not write event to {path}: {error}")
