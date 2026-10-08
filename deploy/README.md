# Deploying the head-detection model to edge / continuous systems

Author: Sathish (sathish.c@magellanic-cloud.com) — Magellanic Cloud, TrainX

`edge_client.py` runs a TrainX-exported ONNX head model against a camera
(RTSP) or a file — on a Raspberry Pi, a store PC, or a server — and reports
**how many people are in frame**, as clean occupancy changes rather than a
per-frame number stream. It needs only `onnxruntime`, `opencv-python`,
`numpy` — no TrainX install.

## 1. Get the artifacts
From the completed run's Artifacts panel download `best.int8.onnx` (edge
default) and `model_package.json` (carries image size + class names, in the
id order the model was trained with).

The manifest is **not optional** — it is how a class id becomes the word
`head`, and it carries the input size the ONNX graph was exported at. If you
only have the ONNX, pass the names by hand in training id order instead:
`--names 'head'`, plus `--imgsz` matching the export.

## 2. Install the runtime (per device)
    # Raspberry Pi OS 64-bit / Debian / Ubuntu
    sudo apt-get update && sudo apt-get install -y python3-pip python3-opencv
    pip3 install onnxruntime numpy

## 3. Run it
    python3 edge_client.py --model best.int8.onnx \
      --manifest model_package.json --source rtsp://CAMERA_IP:554/stream

It prints what it resolved before the first frame, so a misconfiguration is
obvious immediately:

    model    : best.int8.onnx (input 640x640)
    classes  : ['head']
    counting : 'head' -> class id(s) ['0:head']
    smoothing: median of 5 frames, held 3 frame(s), conf>0.25, iou 0.45, every 1 frame(s)
    [CHANGE] frame 812: 0 -> 4 head(s) (+4), peak 4
    [CHANGE] frame 934: 4 -> 7 head(s) (+3), peak 7
    [CHANGE] frame 1190: 7 -> 0 head(s) (-7), peak 7

If the class list does not contain a head label it exits loudly rather than
counting nothing.

## 4. Wire in your action
Edit **`on_count_event()`** in `edge_client.py`. It is called **once per
confirmed occupancy change**, not once per frame:

* `event == "count_changed"` — the smoothed count settled on a new value and
  held it for `--hold` processed frames. `count == 0` means the frame is empty.
* `event == "heartbeat"` — `--report-every` seconds elapsed, so a dashboard
  gets a reading even while nothing changes. Off by default.

Both get `count`, `previous`, `peak`, and the detections as
`(class_name, confidence, [x1,y1,x2,y2])` in original frame coordinates.
Four hooks are already wired for you and need no code change:

    --save-dir /var/log/head-counts     # annotated JPEG snapshot per change
    --csv      /var/log/headcount.csv   # append-only event log
    --webhook  https://your-backend/api/occupancy   # JSON POST per event
    --max-occupancy 25                  # log an alert above this count

The webhook payload:

    {"event": "count_changed", "frame": 934, "timestamp": 1757404800.1,
     "device": "store-pi-04", "count": 7, "previous": 4, "peak": 7,
     "heads": [{"conf": 0.7134, "box": [412.0, 233.5, 448.2, 291.7]}, ...]}

A dead backend never takes the camera loop down — POST failures are logged and
swallowed.

## 5. Tuning knobs

| Flag | Default | When to change it |
| --- | --- | --- |
| `--conf` | 0.25 | Raise it if the count runs high on empty scenes (headrests, posters, lamps). Lower it if people at the back of the frame are missed — then check the snapshots to see what it is actually firing on. **Lowering this cannot recover heads the model never learned**; that is a training fix, not an edge fix. |
| `--iou` | 0.45 | NMS overlap. **Raise it (0.6–0.7) on crowded scenes** — at 0.45 two genuinely adjacent heads can suppress each other and two people are counted as one. |
| `--smooth` | 5 | Median window over processed frames. Raise it on a noisy camera where the count jitters by one or two; lower it to 1–3 if the count must react instantly. |
| `--hold` | 3 | Processed frames a new count must hold before it is published. Raise it where people cut across the edge of frame and you only want settled occupancy. |
| `--report-every` | 0 (off) | Seconds between heartbeat events. Set it (e.g. 60) when a dashboard needs a periodic reading rather than only deltas. |
| `--every` | 1 | Process every Nth frame. **The main Raspberry Pi lever** — `--every 5` on a 25fps camera cuts CPU ~5x. Note `--smooth` and `--hold` count *processed* frames, so raising this also lengthens the real-time debounce. |
| `--max-occupancy` | 0 (off) | Logs an alert when the count exceeds the limit — a capacity or queue-length rule. |
| `--target` | `head` | The class name to count. Use the exact CVAT label if it was renamed. |
| `--loop` | off | Restart a **file** source at EOF. RTSP/HTTP streams always reconnect; a file otherwise ends and the client exits 0. |

### What this client does and does not do
It reports **how many heads are in frame right now** — occupancy. It does not
track identities across frames and does not do directional line-crossing
("entered" vs "left"), because both need a tracker and a calibrated line,
which are deployment decisions rather than model ones. The detections handed
to `on_count_event()` are everything a tracker would need as input.

## 6. Make it continuous (systemd)
    # /etc/systemd/system/head-count.service
    [Unit]
    Description=Head detection / occupancy count
    After=network-online.target
    [Service]
    ExecStart=/usr/bin/python3 /opt/trainx/edge_client.py \
      --model /opt/trainx/best.int8.onnx \
      --manifest /opt/trainx/model_package.json \
      --source rtsp://CAMERA_IP:554/stream \
      --csv /var/log/headcount.csv \
      --report-every 60 \
      --every 3
    Restart=always
    RestartSec=5
    [Install]
    WantedBy=multi-user.target

    sudo systemctl daemon-reload && sudo systemctl enable --now head-count
    journalctl -u head-count -f

## Updating the model later
Re-export from the new run, copy the new `best.int8.onnx` +
`model_package.json` onto the device, `sudo systemctl restart head-count`.
No code change — the manifest carries the image size and class names, and the
client re-resolves `head` by name, so even a CVAT label rename is safe.

## Other targets
- Intel NUC/mini-PC: use the `best_openvino_model/` artifact + `pip install openvino`.
- Google Coral / TFLite boards: use `best-fp16.tflite` + `tflite-runtime`.
- x86 server (Docker): the same TrainX inference image serves any model via
  `model_package.json` — see the dashboard User Guide, "Deploying to Edge &
  Continuous Systems".
