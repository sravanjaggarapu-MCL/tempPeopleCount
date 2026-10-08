import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from people_count import config  # noqa: E402
from people_count.counter import PersonCounter  # noqa: E402
from people_count.counting_line import CountingLine, line_path_for  # noqa: E402
from people_count.events import EventLogger  # noqa: E402


def test_signed_distance_and_sides():
    line = CountingLine(0, 100, 200, 100)
    assert line.signed_distance((50, 130)) == 30
    assert line.signed_distance((50, 70)) == -30
    assert line.side_of((50, 130), dead_zone=12) == 1
    assert line.side_of((50, 105), dead_zone=12) == 0


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "line.json"
    CountingLine(1, 2, 300, 4, entry_side=-1).save(str(path), frame_size=(640, 480))
    loaded = CountingLine.load(str(path), frame_size=(640, 480))
    assert (loaded.x1, loaded.y2, loaded.entry_side) == (1, 4, -1)
    # A different resolution must force a redraw.
    assert CountingLine.load(str(path), frame_size=(1920, 1080)) is None


def test_line_file_name_has_no_password():
    url = "rtsp://admin:Secret%40123@10.0.0.5:554/cam/realmonitor?channel=1&subtype=1"
    name = os.path.basename(line_path_for(url))
    assert "secret" not in name.lower()
    assert name.startswith("10-0-0-5-554")


def test_event_csv(tmp_path):
    logger = EventLogger("rtsp://u:p@10.0.0.5:554/stream1", directory=str(tmp_path))
    counter = PersonCounter(CountingLine(0, 0, 10, 0))
    counter.entry_count = 1
    logger.log(7, "entry", counter)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    rows = list(csv.DictReader(open(files[0])))
    assert rows[0]["track_id"] == "7" and rows[0]["direction"] == "entry"
    assert rows[0]["inside"] == "1"


def test_rtsp_url_encodes_at_sign(monkeypatch):
    monkeypatch.setattr(config, "USERNAME", "admin")
    monkeypatch.setattr(config, "PASSWORD", "Admin@0192")
    url = config.build_rtsp_url()
    assert "Admin%400192@" in url
    assert "****" in config.mask_url(url)


def test_files_read_sequentially_cameras_threaded():
    """Regression: a file on the threaded reader loops forever at its end."""
    from people_count.camera import open_stream, ThreadedVideoStream
    assert not isinstance(open_stream("clip.mp4"), ThreadedVideoStream)
    assert isinstance(open_stream("rtsp://u:p@10.0.0.5:554/s"), ThreadedVideoStream)
