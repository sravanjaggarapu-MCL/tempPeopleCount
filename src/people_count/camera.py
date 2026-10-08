"""
camera.py
=========

Opening the video source and reading frames from it.

Two classes, for two different jobs:

    VideoStream          reads frames one at a time, on the calling thread.
                         Correct for FILES, where every frame matters.
    ThreadedVideoStream  reads frames in a background thread that keeps only
                         the newest one. Correct for LIVE cameras, where being
                         current matters more than seeing every frame.

Use open_stream() at the bottom to pick the right one automatically.

Why a live camera needs the threaded version
--------------------------------------------
An RTSP camera pushes frames on its own schedule and will not wait for you.
Measured on this camera: it delivers a frame every 38 ms (~25 FPS), while one
YOLO pass takes ~90 ms (~11 FPS). Read them sequentially and you consume less
than half of what arrives.

The surplus does not evaporate - it queues in FFmpeg's buffer. That queue is
bounded, so the behaviour has two stages: the lag grows until the buffer is
full, then frames start being discarded AT THE SOURCE, outside your control.
Measured by draining it, sequential reading settles about 36 frames - 1.4
seconds - behind real time, and drops frames from then on. Late video plus
missing frames is exactly what a slow consumer looks like from the outside.

The threaded reader measures 2 frames - 0.1 s - behind instead.

Note that CAP_PROP_BUFFERSIZE does not fix this. It is a hint, honoured
inconsistently across backends, and on FFmpeg/RTSP it generally is not.

The fix is to decouple the two rates. A background thread reads at the
camera's pace and keeps exactly ONE frame - the newest - overwriting whatever
was there. Inference then always starts on the most recent frame available,
and everything that arrived while it was busy is discarded rather than queued.
Dropping frames is the point: for counting people, a frame from two seconds ago
has no value, and the tracker does better with a current frame than a stale one.
"""

import threading
import time

import cv2

from . import config


class VideoStream:
    """A video source (RTSP URL or file) that cleans up after itself.

    Use it as a context manager:

        with VideoStream() as stream:
            for frame in stream:
                ...

    Iterating stops by itself when the stream ends or drops.
    """

    def __init__(self, source: str | None = None):
        # No source given -> the camera configured in config.py.
        self.source = source or config.build_rtsp_url()
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> "VideoStream":
        """Connect to the source. Raises RuntimeError if it cannot."""
        print(f"Opening {config.mask_url(self.source)}")

        # cv2.CAP_FFMPEG explicitly selects the FFmpeg backend. Without it,
        # OpenCV probes its backends in order and on Windows can land on MSMF,
        # which does not handle RTSP URLs. Being explicit removes the guesswork.
        #
        # This call runs the whole RTSP handshake (DESCRIBE -> SETUP -> PLAY),
        # so it blocks for a second or two before returning.
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        # Keep only the newest frame in the internal queue. Without this, frames
        # pile up while the detector is busy and every read() hands back an
        # older one - the displayed video then drifts further behind real time
        # the longer it runs. (Not every backend honours the hint, but asking
        # costs nothing.)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.capture.isOpened():
            # Tailor the advice: a missing file and an unreachable camera fail
            # identically here, but have nothing in common to check.
            if self.source.startswith("rtsp://"):
                raise RuntimeError(
                    "Could not open the camera stream. Check that:\n"
                    f"  1. This machine can reach it:  ping {config.HOST}\n"
                    "  2. The username/password (.env) are correct "
                    "(case-sensitive).\n"
                    "  3. The stream path matches this camera's vendor.\n"
                    "  `python scripts/run_counter.py --mode view` tests the "
                    "stream on its own."
                )
            raise RuntimeError(
                f"Could not open '{self.source}'. Check the path exists and is "
                "a video format OpenCV can read (mp4, avi, mkv...)."
            )

        print(f"Connected: {self.width}x{self.height}")
        return self

    @property
    def width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return the next frame, or None when the stream has ended.

        A frame is a NumPy array of shape (height, width, 3) in BGR channel
        order - the array you hand straight to YOLO.
        """
        ok, frame = self.capture.read()
        # ok is False when the camera reboots, the network hiccups, or another
        # client takes the last stream slot (cameras cap simultaneous viewers,
        # often at 3-5).
        return frame if ok else None

    def __iter__(self):
        """Yield frames until the source runs dry, so callers can just loop."""
        while True:
            frame = self.read()
            if frame is None:
                print("Frame read failed - stream ended or dropped.")
                return
            yield frame

    def release(self):
        """Tear down the RTSP session so the camera frees the stream slot."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # The two methods that make `with VideoStream() as s:` work. __exit__ runs
    # even if the body raised, which is the whole point - the camera is released
    # on the error path too, not just the happy one.
    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False  # never swallow an exception; let it propagate


class ThreadedVideoStream(VideoStream):
    """A live stream read by a background thread that keeps only the newest frame.

    The ingestion thread runs flat out at the camera's own pace. It holds a
    single slot, which it overwrites on every frame - so whatever is in there is
    always the most recent thing the camera has sent. read() takes from that
    slot, and blocks only until something NEW arrives.

    Everything the camera sent while inference was busy is simply discarded. The
    `dropped` count says how many, and that number is the point rather than a
    fault: those are the frames that would otherwise have queued up in front of
    the live picture, each one making what you finally see older than the last.
    """

    def __init__(self, source: str | None = None, reconnect: bool = True):
        super().__init__(source)

        # A Condition is a lock plus a way to wait on it. The consumer sleeps
        # inside wait() - burning nothing - until the ingestion thread calls
        # notify_all() to say a new frame has landed. A polling loop would spin
        # the CPU competing with the very inference we are trying to speed up.
        self._condition = threading.Condition()

        self._frame = None          # the single slot - guarded by _condition
        self._seq = 0               # incremented once per frame ingested
        self._consumed_seq = 0      # the last sequence number read() handed out
        self._frame_time = 0.0      # when the slot's frame was captured
        self._running = False
        self._thread: threading.Thread | None = None

        self.reconnect = reconnect
        self.frames_ingested = 0
        self.frames_consumed = 0
        self.last_frame_age = 0.0   # seconds between capture and consumption

    def open(self) -> "ThreadedVideoStream":
        super().open()
        self._running = True

        # daemon=True so a forgotten stream can never keep the process alive
        # after the main thread has finished.
        self._thread = threading.Thread(target=self._ingest, name="ingest", daemon=True)
        self._thread.start()
        print("Ingestion thread started - keeping only the newest frame.")
        return self

    def _ingest(self):
        """Background loop: read as fast as the camera sends, keep the latest."""
        failures = 0
        while self._running:
            ok, frame = self.capture.read()

            if not ok:
                failures += 1
                if not self.reconnect or failures > config.MAX_RECONNECT_ATTEMPTS:
                    break
                # A live camera drops out for all sorts of transient reasons: a
                # network blip, a reboot, another client taking the last stream
                # slot. Sequential code could only give up; a background thread
                # can quietly rebuild the connection while the display loop
                # carries on showing the last good frame.
                print(f"Stream dropped, reconnecting "
                      f"({failures}/{config.MAX_RECONNECT_ATTEMPTS})...")
                time.sleep(config.RECONNECT_DELAY_SECONDS)
                self.capture.release()
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            failures = 0

            # The copy is NOT optional, and it is the subtle part of this class.
            #
            # OpenCV recycles a small pool of internal buffers - measured on
            # this camera, consecutive read() calls alternate between just TWO
            # addresses. So the array handed to the inference thread would be
            # overwritten underneath it two frames later, corrupting a detection
            # already in progress. Copying gives the slot an array of its own
            # that nothing else will touch.
            #
            # It is cheap: 0.03 ms for a 640x480 frame against ~90 ms of
            # inference. We copy outside the lock, so the consumer is never held
            # up waiting for a memcpy.
            frame = frame.copy()

            with self._condition:
                self._frame = frame
                self._seq += 1
                self.frames_ingested += 1
                self._frame_time = time.time()
                self._condition.notify_all()   # wake the consumer if waiting

        # Whatever ended the loop, tell any waiting consumer that no more frames
        # are coming - otherwise read() would block until its timeout.
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def read(self, timeout: float | None = None):
        """Return the newest unread frame, or None if the stream has finished.

        Blocks until a frame NEWER than the last one handed out arrives. That
        sequence check matters: without it, a consumer faster than the camera
        would be handed the same frame repeatedly and would waste whole
        inference passes re-detecting people who have not moved.
        """
        if timeout is None:
            timeout = config.FRAME_TIMEOUT_SECONDS
        deadline = time.time() + timeout

        with self._condition:
            while self._seq == self._consumed_seq and self._running:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(f"No new frame for {timeout:.0f}s - giving up.")
                    return None
                self._condition.wait(remaining)

            # Woken with nothing new means the ingestion thread has stopped.
            if self._seq == self._consumed_seq:
                return None

            self._consumed_seq = self._seq
            self.frames_consumed += 1
            self.last_frame_age = time.time() - self._frame_time
            return self._frame

    @property
    def dropped(self) -> int:
        """Frames the camera sent that inference never saw."""
        return max(0, self.frames_ingested - self.frames_consumed)

    @property
    def drop_rate(self) -> float:
        """Dropped frames as a percentage of everything the camera sent."""
        if not self.frames_ingested:
            return 0.0
        return 100.0 * self.dropped / self.frames_ingested

    def release(self):
        """Stop the ingestion thread, then close the capture."""
        self._running = False
        with self._condition:
            self._condition.notify_all()     # wake the thread if it is waiting

        if self._thread is not None and self._thread.is_alive():
            # The thread may be blocked inside capture.read() waiting on the
            # network, so we cannot wait for it indefinitely. It is a daemon and
            # the release() below makes its next read() fail, so a bounded join
            # is enough.
            self._thread.join(timeout=2.0)
        self._thread = None

        super().release()


def open_stream(source: str | None = None, threaded: bool | None = None) -> VideoStream:
    """Open the right kind of stream for the source.

    Live streams get the threaded reader; files get the sequential one.

    Using the threaded reader on a FILE would be actively wrong. A file has no
    clock of its own - it is read as fast as the disk allows - so the ingestion
    thread would race to the end while inference was still on the opening
    seconds, and almost every frame would be discarded unseen. Files should be
    processed frame by frame, which is what the sequential reader does.
    """
    resolved = source or config.build_rtsp_url()

    if threaded is None:
        # A URL scheme means something is pushing frames at us in real time.
        threaded = resolved.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://"))

    return ThreadedVideoStream(resolved) if threaded else VideoStream(resolved)
