# People Count — Store Entry / Exit Counter

Counts people **entering** and **exiting** a store from an RTSP camera (or a
video file) using **YOLOv8** detection, **ByteTrack** tracking and a virtual
tripwire line drawn across the doorway.

```
camera ──► YOLOv8 (person boxes) ──► ByteTrack (stable IDs) ──► line crossing ──► ENTRY / EXIT
                                                                     │
                                                         outputs/events/*.csv
```

## Repository layout

```
people-count/
├── configs/
│   ├── config.yaml          all settings (camera, model, counting, display)
│   └── bytetrack.yaml       tracker settings
├── src/people_count/
│   ├── __init__.py          sets FFmpeg RTSP-over-TCP (must run before cv2 loads)
│   ├── config.py            loads config.yaml + credentials from .env
│   ├── camera.py            stream reading (threaded for live, sequential for files)
│   ├── detector.py          YOLOv8 detection + ByteTrack tracking
│   ├── counting_line.py     tripwire geometry, dead zone, save/load per camera
│   ├── counter.py           entry/exit state machine
│   ├── events.py            daily CSV log of every crossing
│   ├── line_setup.py        click-and-drag screen to draw the line
│   ├── source_prompt.py     "which camera?" prompt
│   ├── drawing.py           boxes, trails, HUD
│   └── app.py               main loop + one pipeline class per mode
├── scripts/
│   └── run_counter.py       ← the entry point
├── tests/                   pytest suite (no camera or model needed)
├── weights/                 yolov8n.pt is downloaded here on first run
├── outputs/
│   ├── lines/               saved counting line per camera
│   └── events/              CSV of entries / exits per camera per day
├── .env.example             copy to .env and add the camera password
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put the camera username/password in .env
```

Set the camera host and stream path in `configs/config.yaml`. The password goes
only in `.env`, which is git-ignored and never pushed.

## Run

```bash
python scripts/run_counter.py                       # asks which camera, then counts
python scripts/run_counter.py --mode view           # test the camera only (no model)
python scripts/run_counter.py --redraw-line         # draw a new door line
python scripts/run_counter.py --source sample.mp4   # run on a recording
python scripts/run_counter.py --headless --no-prompt  # server: no window, CSV only
python scripts/run_counter.py --help                # all options
```

| Mode | What it does |
|---|---|
| `count` (default) | Draw a line, then count ENTRY / EXIT |
| `view` | Video only, no model. Quickest camera test |
| `detect` | Person boxes, no IDs |
| `track` | Boxes + stable IDs, no counting |

Window keys: `q`/Esc quit, `t` toggle trails, `r` reset counts.

## Drawing the door line

On the first run you draw the line on live video:

| Key | Action |
|---|---|
| click and drag | draw the line across the doorway |
| `f` | flip which side is ENTRY (watch the green arrow) |
| `r` | clear and redraw |
| Enter / `c` | confirm |
| `q` / Esc | cancel |

**Make sure the green ENTRY arrow points into the store.** If it does not,
press `f`. The line is saved per camera in `outputs/lines/` and reused on later
runs.

## Output: entry / exit log

Every crossing is written to `outputs/events/<camera>_<YYYY-MM-DD>.csv`:

```
timestamp,camera,track_id,direction,entry_total,exit_total,inside
2026-10-08T10:15:02,172-20-100-138-554-cam-realmonitor-channel-1-subtype-1,12,entry,1,0,1
2026-10-08T10:15:40,172-20-100-138-554-cam-realmonitor-channel-1-subtype-1,15,exit,1,1,0
```

`inside` = entries − exits. It assumes the store was empty when counting
started, so press `r` to reset when it is.

## How counting avoids false counts

1. **Signed distance.** For each person's foot point, the cross product with
   the line gives which side they are on and how far away they are, in pixels.
2. **Dead zone (`dead_zone_px`).** A band around the line where the side is not
   updated. Someone standing on the threshold jitters across the line but is
   never counted. They must clear the band on the far side. In synthetic tests,
   300 frames of ±8 px jitter on the line give **0** counts with the dead zone
   and over 50 without it.
3. **Segment bound.** Crossings beyond the ends of the drawn line are ignored,
   so people walking past the door are not counted.
4. **Foot point, not box centre.** This matches where the person actually stands.
5. **Minimum track age.** A brand-new ID cannot count, which blocks counts from
   flicker and re-issued IDs.

## Tuning (`configs/config.yaml`)

| Setting | Default | Notes |
|---|---|---|
| `model.inference_size` | 480 | 320 = faster, 640 = more accurate |
| `model.track_conf_threshold` | 0.15 | Keep low: ByteTrack recovers occluded people from weak detections |
| `counting.dead_zone_px` | 12 | Tuned for 640×480; increase for 1080p |
| `counting.min_track_age_frames` | 3 | |
| `bytetrack.yaml: track_buffer` | 30 | Increase if people get new IDs after occlusion (double counts) |

Prefer the camera **sub-stream** (Dahua `subtype=1`, Hikvision `/Streaming/Channels/102`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

These cover clean entries and exits, loitering with jitter, hesitating and
turning back, crossings off the line ends, new-track guard, two people at
once, CSV logging, and password-safe URLs.

## Credits

Counting logic is adapted from
[sravanjaggarapu-MCL/peopleCount](https://github.com/sravanjaggarapu-MCL/peopleCount).
