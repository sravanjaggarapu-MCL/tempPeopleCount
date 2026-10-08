"""
detector.py
===========

The YOLO model, in both of its modes:

    detect()  - find people in one frame, with no memory of any other frame
    track()   - find people AND give each a stable ID across frames (ByteTrack)

Why tracking is a separate thing from detection
-----------------------------------------------
Detection is memoryless: each frame is analysed from scratch, so nothing
connects the box in frame 12 to the box in frame 11. That is fine for a
headcount, but useless for entry/exit counting - to say "this person crossed
the line downwards" you must first be able to say "this person" across time.

How ByteTrack supplies that
---------------------------
Each frame it predicts where every existing track should have moved (a Kalman
filter assuming roughly constant velocity), then matches those predictions
against the new detections by box overlap (IoU).

Its namesake trick is a two-round match. Most trackers discard low-confidence
detections outright; ByteTrack does not:

    Round 1 - match existing tracks against HIGH confidence detections.
    Round 2 - take the tracks still unmatched, try them against the LOW
              confidence leftovers.

A person half-hidden behind someone else produces exactly such a weak
detection. Round 2 is what keeps their track alive through the occlusion
instead of killing it and reissuing a fresh ID - and every ID change like that
would become a double count once we are counting crossings.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    """One person found in one frame.

    A small plain-data class rather than raw tensors, so the rest of the code
    never has to know what shape ultralytics returns things in. If we swap YOLO
    for something else later, this is the only boundary that has to change.
    """

    x1: int          # left edge, pixels
    y1: int          # top edge (y grows DOWNWARDS in image coordinates)
    x2: int          # right edge
    y2: int          # bottom edge
    confidence: float
    track_id: int | None = None   # None in detect mode; an integer in track mode

    @property
    def foot_point(self) -> tuple[int, int]:
        """Bottom-centre of the box: where the person meets the floor.

        Entry/exit counting will test THIS point against the counting line
        rather than the box centre. On a bullet camera mounted above head
        height the body leans into the frame, so the centroid reaches the line
        while the person is still short of the doorway. The feet are where the
        person actually is.
        """
        return (self.x1 + self.x2) // 2, self.y2

    @property
    def centroid(self) -> tuple[int, int]:
        """Middle of the box. Handy for drawing; not what we count on."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


class PersonDetector:
    """Wraps the YOLO model and turns its output into Detection objects."""

    def __init__(self, model_path: str | None = None, inference_size: int | None = None):
        self.model_path = model_path or config.MODEL_PATH
        self.inference_size = inference_size or config.INFERENCE_SIZE

        # On the very first run this downloads the weights (~6 MB for nano) and
        # caches them next to the project, so later runs work offline. The
        # weights are pretrained on COCO - we are NOT training anything.
        print(f"Loading {self.model_path} ...")
        self.model = YOLO(self.model_path)
        self._warm_up()
        print(f"Model ready (imgsz={self.inference_size}).")

    def _warm_up(self):
        """Run one throwaway inference on a blank image.

        PyTorch's first predict() call is dramatically slower than the rest -
        it allocates buffers, picks algorithms and fills caches on the way
        through. Measured here: ~2-3 s for the first call, ~0.09 s for every
        one after it. Doing it now, before the window opens, keeps that stall
        out of the display loop where it looks like the video has frozen.
        """
        blank = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=self.inference_size, verbose=False)

    def detect(self, frame, conf: float | None = None) -> list[Detection]:
        """Find people in one frame. No IDs, no memory of previous frames."""
        results = self.model.predict(
            frame,
            classes=[config.PERSON_CLASS_ID],   # people only - see config.py
            conf=conf if conf is not None else config.DETECT_CONF_THRESHOLD,
            imgsz=self.inference_size,
            verbose=False,                      # silence per-frame console spam
        )
        # predict() returns one result per input image; we passed a single
        # frame, so results[0] is ours.
        return self._to_detections(results[0].boxes, with_ids=False)

    def track(self, frame, conf: float | None = None) -> list[Detection]:
        """Find people and give each a stable ID that persists across frames."""
        results = self.model.track(
            frame,
            # persist=True is the critical argument: it tells ultralytics this
            # frame continues the same video as the last call, so tracker state
            # carries over. Omit it and the tracker is rebuilt every frame -
            # everyone would be "ID 1", newly born, forever.
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[config.PERSON_CLASS_ID],
            conf=conf if conf is not None else config.TRACK_CONF_THRESHOLD,
            imgsz=self.inference_size,
            verbose=False,
        )
        return self._to_detections(results[0].boxes, with_ids=True)

    @staticmethod
    def _to_detections(boxes, with_ids: bool) -> list[Detection]:
        """Convert ultralytics' Boxes object into our plain Detection list."""
        if boxes is None or len(boxes) == 0:
            return []

        # boxes.id is None when the tracker has no confirmed tracks yet - an
        # empty scene, or the first frames while tracks are still being
        # confirmed. Indexing it unguarded would crash the loop.
        has_ids = with_ids and boxes.id is not None

        # Pull the parallel arrays off the tensor in one go; cheaper than
        # touching each box individually. .xyxy is already in ORIGINAL FRAME
        # pixels - ultralytics undoes the imgsz letterboxing for us, so there
        # is no rescaling to do here.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if has_ids else [None] * len(xyxy)

        return [
            Detection(int(x1), int(y1), int(x2), int(y2),
                      float(conf), int(tid) if tid is not None else None)
            for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids)
        ]
