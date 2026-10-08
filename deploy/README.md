# Deploying the mobile-in-frame model to edge / continuous systems

Author: Sathish (sathish.c@magellanic-cloud.com) — Magellanic Cloud, TrainX

`edge_client.py` runs a TrainX-exported ONNX model against a camera (RTSP) or
a file — on a Raspberry Pi, a store PC, or a server — and reports **two clean
transitions**: a phone entered the frame, and the frame is clear again. It
needs only `onnxruntime`, `opencv-python`, `numpy` — no TrainX install.

## 1. Get the artifacts
From the completed run's Artifacts panel download `best.int8.onnx` (edge
default) and `model_package.json` (carries image size + class names, in the
id order the model was trained with).

The manifest is **not optional** — it is how a class id becomes the word
`mobile`. If you only have the ONNX, pass the names by hand in training id
order instead: `--names 'person,mobile'`.

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
    classes  : ['person', 'mobile']
    watching : 'mobile' -> class id(s) ['1:mobile']
    debounce : 3 frames on / 15 frames off, conf>0.25, every 1 frame(s)
    [ALERT] frame 812: MOBILE IN FRAME - 1 phone(s), top conf 0.71, 2 person(s)
    [CLEAR] frame 934: no mobile in frame (2 person(s))

If the class list does not contain a phone label it exits loudly rather than
watching for nothing.

## 4. Wire in your action
Edit **`on_mobile_event()`** in `edge_client.py`. It is called **once per
confirmed transition**, not once per frame:

* `event == "mobile_detected"` — a phone has been in view for `--on-frames`
  consecutive processed frames
* `event == "clear"` — no phone for `--off-frames` consecutive frames

Both get the detections as `(class_name, confidence, [x1,y1,x2,y2])` in
original frame coordinates. Two hooks are already wired for you and need no
code change:

    --save-dir /var/log/mobile-alerts    # annotated JPEG snapshot per alert
    --webhook  https://your-backend/api/mobile-events   # JSON POST per event

The webhook payload:

    {"event": "mobile_detected", "frame": 812, "timestamp": 1757404800.1,
     "device": "store-pi-04", "persons": 2,
     "mobiles": [{"conf": 0.7134, "box": [412.0, 233.5, 448.2, 291.7]}]}

A dead backend never takes the camera loop down — POST failures are logged and
swallowed.

## 5. Tuning knobs

| Flag | Default | When to change it |
| --- | --- | --- |
| `--conf` | 0.25 | Raise it if you get false phones (wallets, badges, remotes). Lower it if phones are missed — then check the alert snapshots to see what it is actually firing on. |
| `--on-frames` | 3 | Raise it on a noisy or busy camera to kill flicker alerts. Lower to 1 for the fastest possible alert. |
| `--off-frames` | 15 | How long the phone must be gone before `clear`. Raise it so a phone briefly hidden by a hand does not produce alert/clear/alert churn. |
| `--every` | 1 | Process every Nth frame. **The main Raspberry Pi lever** — `--every 5` on a 25fps camera cuts CPU ~5x and still reacts in well under a second. Note `--on-frames` counts *processed* frames. |
| `--iou` | 0.45 | NMS overlap. Rarely needs touching. |
| `--target` | `mobile` | The class name to watch. Use the exact CVAT label if it was renamed. |
| `--loop` | off | Restart a **file** source at EOF. RTSP/HTTP streams always reconnect; a file otherwise ends and the client exits 0. |

## 6. Make it continuous (systemd)
    # /etc/systemd/system/mobile-watch.service
    [Unit]
    Description=Mobile-in-frame detection
    After=network-online.target
    [Service]
    ExecStart=/usr/bin/python3 /opt/trainx/edge_client.py \
      --model /opt/trainx/best.int8.onnx \
      --manifest /opt/trainx/model_package.json \
      --source rtsp://CAMERA_IP:554/stream \
      --save-dir /var/log/mobile-alerts \
      --every 3
    Restart=always
    RestartSec=5
    [Install]
    WantedBy=multi-user.target

    sudo systemctl daemon-reload && sudo systemctl enable --now mobile-watch
    journalctl -u mobile-watch -f

## Updating the model later
Re-export from the new run, copy the new `best.int8.onnx` +
`model_package.json` onto the device, `sudo systemctl restart mobile-watch`.
No code change — the manifest carries the image size and class names, and the
client re-resolves `mobile` by name, so even a CVAT label reorder is safe.

## Other targets
- Intel NUC/mini-PC: use the `best_openvino_model/` artifact + `pip install openvino`.
- Google Coral / TFLite boards: use `best-fp16.tflite` + `tflite-runtime`.
- x86 server (Docker): the same TrainX inference image serves any model via
  `model_package.json` — see the dashboard User Guide, "Deploying to Edge &
  Continuous Systems".
