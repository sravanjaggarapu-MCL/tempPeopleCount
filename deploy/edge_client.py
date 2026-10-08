#!/usr/bin/env python3
# Author: Sathish (sathish.c@magellanic-cloud.com) - Magellanic Cloud, TrainX
# Customized for the HEAD DETECTION project (person's head; CVAT label 'head').
#
# Edge client: pull frames from a camera (RTSP) or a video/image, run the
# TrainX-exported ONNX head model, and answer the one question this project
# exists to answer - HOW MANY PEOPLE ARE IN FRAME RIGHT NOW?
#
# Heads are counted instead of bodies on purpose: in a queue, a doorway or a
# packed room, bodies occlude each other almost completely while heads stay
# visible, so a head count degrades far more gracefully with crowding than a
# person count does.
#
# Continuous by design - run it under systemd (see deploy/README.md) so a
# device recovers automatically. Depends only on: onnxruntime, opencv-python,
# numpy (+ the standard library for the optional webhook).
#
# WHY THE EXTRA MACHINERY vs. a plain "print every detection" loop:
#
#   * class ids are resolved BY NAME from model_package.json, never
#     hardcoded. The id of 'head' is whatever position it holds in the
#     project's CVAT label list, and that list is regenerated on every
#     training run - a rename or reorder in CVAT would silently turn a head
#     count into a count of some other class if this file assumed id 0.
#   * NMS. The raw ONNX head emits thousands of overlapping candidates; one
#     head becomes 20 "detections" without it, which makes the count - the
#     entire output of this client - meaningless.
#   * SMOOTHING. A raw per-frame count on a 25fps stream flickers by one or
#     two every few frames as borderline heads cross the confidence
#     threshold. The reported count is the MEDIAN over a short window, and a
#     new value must hold for --hold processed frames before it is published,
#     so downstream systems get occupancy changes instead of a noise stream.
#
# Usage:
#   python3 edge_client.py --model best.int8.onnx \
#       --manifest model_package.json --source rtsp://CAMERA_IP:554/stream
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

# Names this project's head class might carry in CVAT. Matching is
# case-insensitive and ignores spaces/underscores/hyphens, so "Head",
# "person_head" and "Human-Head" all resolve. Only used to FIND the class in
# the manifest's list - the id itself always comes from that list.
HEAD_ALIASES = ("head", "personhead", "humanhead", "peoplehead", "headperson",
                "headcount", "face", "personface", "humanface")

# Class-id offset used to make NMS per-class (boxes of different classes must
# not suppress each other). Larger than any sane frame.
NMS_CLASS_STRIDE = 8192.0


def letterbox(img, size, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = size - nw, size - nh
    left, top = pw // 2, ph // 2
    padded = cv2.copyMakeBorder(resized, top, ph - top, left, pw - left,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def decode(pred, conf_thres):
    # (1, A, B) -> drop batch. Orientation tells the layout apart: real anchor
    # counts are in the thousands, so rows < cols (with cols >= 300) is the
    # transposed ultralytics layout (no objectness); otherwise classic yolov5.
    p = pred[0] if pred.ndim == 3 else pred
    ultra = p.shape[0] < p.shape[1] and p.shape[1] >= 300
    if ultra:
        p = p.T
        cls = p[:, 4:]
        ids = np.argmax(cls, axis=1)
        conf = cls[np.arange(cls.shape[0]), ids]
    else:
        obj = p[:, 4]
        cls = p[:, 5:]
        ids = np.argmax(cls, axis=1)
        conf = obj * cls[np.arange(cls.shape[0]), ids]
    keep = conf > conf_thres
    xywh = p[keep, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy, conf[keep], ids[keep]


def nms(boxes, scores, ids, conf_thres, iou_thres):
    """Per-class non-maximum suppression via OpenCV (already a dependency, so
    no torch/torchvision on the device). Boxes of different classes are pushed
    into disjoint coordinate space so they cannot suppress one another.

    The IoU threshold matters more here than in a presence detector: in a
    crowd, adjacent heads genuinely overlap, and too low an --iou merges two
    people into one head. Raise it on dense scenes - see deploy/README.md."""
    if len(boxes) == 0:
        return boxes, scores, ids
    offset = ids.astype(np.float32) * NMS_CLASS_STRIDE
    rects = np.stack([boxes[:, 0] + offset,
                      boxes[:, 1] + offset,
                      boxes[:, 2] - boxes[:, 0],
                      boxes[:, 3] - boxes[:, 1]], axis=1)
    keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.astype(np.float32).tolist(),
                            float(conf_thres), float(iou_thres))
    if keep is None or len(keep) == 0:
        return boxes[:0], scores[:0], ids[:0]
    idx = np.asarray(keep).flatten().astype(int)
    return boxes[idx], scores[idx], ids[idx]


def scale(boxes, r, pad, shape):
    left, top = pad
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return boxes


def canonical(name):
    return "".join(str(name).lower().split()).replace("_", "").replace("-", "")


def resolve_target_ids(names, wanted):
    """Class ids in `names` that mean the thing we are counting.

    Exact (normalized) match first; then the alias list; then a substring
    sweep, so an unexpected CVAT label like "head-top-view" is still picked up
    rather than silently counting nothing."""
    norm = [canonical(n) for n in names]
    want = canonical(wanted)
    ids = [i for i, n in enumerate(norm) if n == want]
    if ids:
        return ids
    ids = [i for i, n in enumerate(norm) if n in HEAD_ALIASES]
    if ids:
        return ids
    return [i for i, n in enumerate(norm)
            if want in n or any(a in n for a in HEAD_ALIASES)]


def median(values):
    """Plain median of a short list - the smoothing window is a handful of
    frames, so sorting it per frame costs nothing and numpy is not needed."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2:
        return s[n // 2]
    return int(round((s[n // 2 - 1] + s[n // 2]) / 2.0))


def draw(frame, dets, count, target_names):
    """Annotate a copy of the frame - used only for snapshots."""
    out = frame.copy()
    for name, conf, (x1, y1, x2, y2) in dets:
        color = (0, 200, 0) if name in target_names else (160, 160, 160)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{conf:.2f}", (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def post_webhook(url, payload):
    """Fire-and-forget JSON POST. A dead backend must never take the camera
    loop down with it, so every failure is logged and swallowed."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"webhook POST to {url} failed: {e}", flush=True)


def on_count_event(event, frame_idx, count, previous, peak,
                   head_dets, all_dets, frame, state, args):
    """---- YOUR ACTION HOOK -------------------------------------------------

    Called on a CONFIRMED occupancy change, not once per frame:

      event == "count_changed"  the smoothed head count settled on a new
                                value and held it for --hold processed
                                frames. count == 0 means the frame is empty.
      event == "heartbeat"      --report-every seconds elapsed; carries the
                                current count and the peak since the last
                                event, so a dashboard still gets a reading
                                while nothing changes.

    count/previous/peak are integers; head_dets and all_dets are lists of
    (class_name, confidence, [x1,y1,x2,y2]) in ORIGINAL frame coordinates.
    Replace the body below with the real thing: push to an occupancy
    dashboard, trip a queue-length alert, write to a store PoS system.
    """
    if event == "heartbeat":
        print(f"[COUNT] frame {frame_idx}: {count} head(s) in frame "
              f"(peak {peak} since last report)", flush=True)
    else:
        delta = count - previous
        print(f"[CHANGE] frame {frame_idx}: {previous} -> {count} head(s) "
              f"({delta:+d}), peak {peak}", flush=True)

    if args.max_occupancy and count > args.max_occupancy and event == "count_changed":
        print(f"[ALERT] frame {frame_idx}: occupancy {count} exceeds the "
              f"--max-occupancy limit of {args.max_occupancy}", flush=True)

    if args.save_dir and event == "count_changed":
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(
            args.save_dir, time.strftime("heads-%Y%m%d-%H%M%S-n") + f"{count}.jpg")
        cv2.imwrite(path, draw(frame, all_dets, count, state["target_names"]))
        print(f"         snapshot: {path}", flush=True)

    if args.csv:
        # Append-only so a restart never truncates a day's history.
        new = not os.path.isfile(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write("timestamp,device,event,frame,count,peak\n")
            f.write(f"{time.time():.3f},{args.device_id},{event},"
                    f"{frame_idx},{count},{peak}\n")

    if args.webhook:
        post_webhook(args.webhook, {
            "event": event,
            "frame": frame_idx,
            "timestamp": time.time(),
            "device": args.device_id,
            "count": count,
            "previous": previous,
            "peak": peak,
            "heads": [{"conf": round(c, 4), "box": [round(v, 1) for v in b]}
                      for _, c, b in head_dets],
        })
    # ---------------------------------------------------------------------


def is_stream(source):
    """RTSP/HTTP cameras are reconnected forever; a plain file ends."""
    return str(source).split("://", 1)[0].lower() in (
        "rtsp", "rtmp", "http", "https", "udp", "tcp")


def load_manifest(args):
    """img_size + class names. The manifest is how id -> 'head' is resolved,
    so a missing one is only survivable when --names is given explicitly.

    img_size is NOT a free choice: the ONNX graph was exported at a fixed
    input size and the letterbox below has to match it exactly."""
    size, names = args.imgsz, []
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        size = int(manifest.get("img_size") or manifest.get("imgsz") or size)
        names = manifest.get("names") or manifest.get("class_names") or []
        if isinstance(names, dict):
            names = [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    return size, [str(n) for n in names]


def main():
    ap = argparse.ArgumentParser(
        description="Head-count edge inference client (TrainX ONNX model)")
    ap.add_argument("--model", required=True, help="best.int8.onnx (or best.onnx)")
    ap.add_argument("--manifest", help="model_package.json (img_size + class names)")
    ap.add_argument("--names", help="comma-separated class names, in training id "
                                    "order - overrides the manifest, e.g. 'head'")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="model input size when no manifest is given (must match export)")
    ap.add_argument("--source", required=True, help="rtsp://... , a video file, or an image")
    ap.add_argument("--target", default="head",
                    help="class name to count (default: head)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU threshold - raise it on crowded scenes where "
                         "adjacent heads genuinely overlap")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (raise it on a Raspberry Pi)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="median-filter the per-frame count over this many "
                         "processed frames")
    ap.add_argument("--hold", type=int, default=3,
                    help="processed frames a new count must hold before it is reported")
    ap.add_argument("--report-every", type=float, default=0.0, dest="report_every",
                    help="also emit a heartbeat with the current count every N "
                         "seconds (0 = only on change)")
    ap.add_argument("--max-occupancy", type=int, default=0, dest="max_occupancy",
                    help="log an alert when the count exceeds this (0 = off)")
    ap.add_argument("--save-dir", help="write an annotated snapshot on each change")
    ap.add_argument("--csv", help="append every event to this CSV file")
    ap.add_argument("--webhook", help="POST each event as JSON to this URL")
    ap.add_argument("--device-id", default=os.uname().nodename if hasattr(os, "uname") else "edge",
                    help="identifier included in webhook/CSV payloads")
    ap.add_argument("--loop", action="store_true",
                    help="restart a FILE source at EOF (streams always reconnect)")
    args = ap.parse_args()

    size, names = load_manifest(args)
    if not names:
        raise SystemExit(
            "no class names available: pass --manifest model_package.json "
            "(from the run's Artifacts panel) or --names 'head' in training "
            "id order. Without them a class id cannot be mapped to 'head'.")
    target_ids = resolve_target_ids(names, args.target)
    if not target_ids:
        raise SystemExit(
            f"class {args.target!r} is not in this model's class list {names}. "
            f"The names come from the project's CVAT label list - pass --target "
            f"with the exact label this model was trained on.")
    print(f"model    : {args.model} (input {size}x{size})", flush=True)
    print(f"classes  : {names}", flush=True)
    print(f"counting : {args.target!r} -> class id(s) "
          f"{[f'{i}:{names[i]}' for i in target_ids]}", flush=True)
    print(f"smoothing: median of {args.smooth} frames, held {args.hold} frame(s), "
          f"conf>{args.conf}, iou {args.iou}, every {args.every} frame(s)", flush=True)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(args.source)  # transparently handles rtsp OR a file
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    streaming = is_stream(args.source)

    # Handed to the action hook so snapshots can be coloured without globals.
    state = {"names": names,
             "target_names": {names[i] for i in target_ids if i < len(names)}}

    idx = 0
    window = deque(maxlen=max(1, args.smooth))
    reported = 0           # last published count
    candidate = None       # a new smoothed value waiting out --hold
    candidate_frames = 0
    peak = 0               # peak since the last published event
    last_report = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if streaming or args.loop:
                # an RTSP camera dropped, or a file we were told to loop
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(args.source)
                continue
            break  # a file ended and we were not asked to loop - exit cleanly

        idx += 1
        if args.every > 1 and idx % args.every:
            continue

        padded, r, pad = letterbox(frame, size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        out = sess.run(None, {inp: blob})[0]
        boxes, conf, ids = decode(out, args.conf)
        boxes, conf, ids = nms(boxes, conf, ids, args.conf, args.iou)
        if len(boxes):
            boxes = scale(boxes, r, pad, frame.shape)
        dets = [(names[int(c)] if int(c) < len(names) else str(int(c)),
                 float(cf), b.tolist()) for b, cf, c in zip(boxes, conf, ids)]
        heads = [d for d, c in zip(dets, ids) if int(c) in target_ids]

        # Smooth, then debounce: the median kills single-frame flicker, and
        # the hold counter stops a real but momentary change (someone cutting
        # across the edge of frame) from publishing two events a second apart.
        window.append(len(heads))
        smoothed = median(window)
        peak = max(peak, smoothed)
        if smoothed == reported:
            candidate, candidate_frames = None, 0
        else:
            if smoothed == candidate:
                candidate_frames += 1
            else:
                candidate, candidate_frames = smoothed, 1
            if candidate_frames >= args.hold:
                previous, reported = reported, smoothed
                candidate, candidate_frames = None, 0
                on_count_event("count_changed", idx, reported, previous, peak,
                               heads, dets, frame, state, args)
                peak = reported
                last_report = time.time()

        if args.report_every and time.time() - last_report >= args.report_every:
            on_count_event("heartbeat", idx, reported, reported, peak,
                           heads, dets, frame, state, args)
            peak = reported
            last_report = time.time()

    cap.release()
    print(f"source ended after {idx} frame(s)", flush=True)


if __name__ == "__main__":
    main()
