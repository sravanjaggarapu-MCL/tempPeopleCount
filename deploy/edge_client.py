#!/usr/bin/env python3
# Author: Sathish (sathish.c@magellanic-cloud.com) - Magellanic Cloud, TrainX
# Customized for the mobile-in-frame detection project (labels: person, mobile).
#
# Edge client: pull frames from a camera (RTSP) or a video/image, run the
# TrainX-exported ONNX model, and answer the one question this project exists
# to answer - IS THERE A MOBILE PHONE IN THE FRAME RIGHT NOW?
#
# Continuous by design - run it under systemd (see deploy/README.md) so a
# device recovers automatically. Depends only on: onnxruntime, opencv-python,
# numpy (+ the standard library for the optional webhook).
#
# WHY THE EXTRA MACHINERY vs. a plain "print every detection" loop:
#
#   * class ids are resolved BY NAME from model_package.json, never
#     hardcoded. The id of 'mobile' is whatever position it holds in the
#     project's CVAT label list, and that list is regenerated on every
#     training run - a rename or reorder in CVAT would silently turn
#     "mobile detected" into "person detected" if this file assumed id 1.
#   * NMS. The raw ONNX head emits thousands of overlapping candidates; one
#     phone becomes 20 "detections" without it, which makes every count and
#     every alert threshold meaningless.
#   * debounce. A single-frame flicker on a 25fps stream is noise, not an
#     event. A phone must be seen in --on-frames consecutive processed frames
#     before MOBILE_DETECTED fires, and be gone for --off-frames before CLEAR
#     does, so downstream systems get two clean transitions instead of a
#     thousand duplicate alerts per minute.
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

import cv2
import numpy as np
import onnxruntime as ort

# Names this project's 'mobile' class might carry in CVAT. Matching is
# case-insensitive and ignores spaces/underscores, so "Cell Phone",
# "mobile_phone" and "MobilePhone" all resolve. Only used to FIND the class in
# the manifest's list - the id itself always comes from that list.
MOBILE_ALIASES = ("mobile", "phone", "cellphone", "mobilephone",
                  "smartphone", "handphone", "telephone")

# Class-id offset used to make NMS per-class (a person box and a phone box
# that overlap must not suppress each other). Larger than any sane frame.
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
    into disjoint coordinate space so they cannot suppress one another."""
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
    """Class ids in `names` that mean the thing we are watching for.

    Exact (normalized) match first; then the alias list; then a substring
    sweep, so an unexpected CVAT label like "mobile-phone-in-hand" is still
    picked up rather than silently watching nothing."""
    norm = [canonical(n) for n in names]
    want = canonical(wanted)
    ids = [i for i, n in enumerate(norm) if n == want]
    if ids:
        return ids
    ids = [i for i, n in enumerate(norm) if n in MOBILE_ALIASES]
    if ids:
        return ids
    return [i for i, n in enumerate(norm) if want in n or any(a in n for a in MOBILE_ALIASES)]


def draw(frame, dets):
    """Annotate a copy of the frame - used only for alert snapshots."""
    out = frame.copy()
    for name, conf, (x1, y1, x2, y2) in dets:
        color = (0, 0, 255) if canonical(name) in MOBILE_ALIASES else (0, 200, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{name} {conf:.2f}", (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
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


def on_mobile_event(event, frame_idx, mobile_dets, all_dets, frame, args):
    """---- YOUR ACTION HOOK -------------------------------------------------

    Called ONCE per confirmed transition, not once per frame:

      event == "mobile_detected"  a phone has been in view for --on-frames
                                  consecutive processed frames
      event == "clear"            no phone for --off-frames consecutive frames

    mobile_dets / all_dets are lists of (class_name, confidence, [x1,y1,x2,y2])
    in ORIGINAL frame coordinates. Replace the body below with the real thing:
    raise an alert, flip a relay, write to a store dashboard, POST to your
    backend, start recording, whatever this deployment needs.
    """
    people = sum(1 for n, _, _ in all_dets if canonical(n).startswith("person"))
    if event == "mobile_detected":
        best = max((c for _, c, _ in mobile_dets), default=0.0)
        print(f"[ALERT] frame {frame_idx}: MOBILE IN FRAME - "
              f"{len(mobile_dets)} phone(s), top conf {best:.2f}, "
              f"{people} person(s)", flush=True)
    else:
        print(f"[CLEAR] frame {frame_idx}: no mobile in frame "
              f"({people} person(s))", flush=True)

    if args.save_dir and event == "mobile_detected":
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir,
                            time.strftime("mobile-%Y%m%d-%H%M%S.jpg"))
        cv2.imwrite(path, draw(frame, all_dets))
        print(f"        snapshot: {path}", flush=True)

    if args.webhook:
        post_webhook(args.webhook, {
            "event": event,
            "frame": frame_idx,
            "timestamp": time.time(),
            "device": args.device_id,
            "mobiles": [{"conf": round(c, 4), "box": [round(v, 1) for v in b]}
                        for _, c, b in mobile_dets],
            "persons": people,
        })
    # ---------------------------------------------------------------------


def is_stream(source):
    """RTSP/HTTP cameras are reconnected forever; a plain file ends."""
    return str(source).split("://", 1)[0].lower() in ("rtsp", "rtmp", "http", "https", "udp", "tcp")


def load_manifest(args):
    """img_size + class names. The manifest is how id -> 'mobile' is resolved,
    so a missing one is only survivable when --names is given explicitly."""
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
        description="Mobile-in-frame edge inference client (TrainX ONNX model)")
    ap.add_argument("--model", required=True, help="best.int8.onnx (or best.onnx)")
    ap.add_argument("--manifest", help="model_package.json (img_size + class names)")
    ap.add_argument("--names", help="comma-separated class names, in training id "
                                    "order - overrides the manifest, e.g. 'person,mobile'")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="model input size when no manifest is given (must match export)")
    ap.add_argument("--source", required=True, help="rtsp://... , a video file, or an image")
    ap.add_argument("--target", default="mobile",
                    help="class name to watch for (default: mobile)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (raise it on a Raspberry Pi)")
    ap.add_argument("--on-frames", type=int, default=3,
                    help="consecutive positive frames before MOBILE_DETECTED fires")
    ap.add_argument("--off-frames", type=int, default=15,
                    help="consecutive negative frames before CLEAR fires")
    ap.add_argument("--save-dir", help="write an annotated snapshot on each alert")
    ap.add_argument("--webhook", help="POST each event as JSON to this URL")
    ap.add_argument("--device-id", default=os.uname().nodename if hasattr(os, "uname") else "edge",
                    help="identifier included in webhook payloads")
    ap.add_argument("--loop", action="store_true",
                    help="restart a FILE source at EOF (streams always reconnect)")
    args = ap.parse_args()

    size, names = load_manifest(args)
    if not names:
        raise SystemExit(
            "no class names available: pass --manifest model_package.json "
            "(from the run's Artifacts panel) or --names 'person,mobile' in "
            "training id order. Without them a class id cannot be mapped to "
            "'mobile'.")
    target_ids = resolve_target_ids(names, args.target)
    if not target_ids:
        raise SystemExit(
            f"class {args.target!r} is not in this model's class list {names}. "
            f"The names come from the project's CVAT label list - pass --target "
            f"with the exact label this model was trained on.")
    print(f"model    : {args.model} (input {size}x{size})", flush=True)
    print(f"classes  : {names}", flush=True)
    print(f"watching : {args.target!r} -> class id(s) "
          f"{[f'{i}:{names[i]}' for i in target_ids]}", flush=True)
    print(f"debounce : {args.on_frames} frames on / {args.off_frames} frames off, "
          f"conf>{args.conf}, every {args.every} frame(s)", flush=True)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(args.source)  # transparently handles rtsp OR a file
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    streaming = is_stream(args.source)

    idx = 0
    present = False        # current debounced state: is a mobile in frame?
    hits = misses = 0      # consecutive positive / negative processed frames
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
        mobiles = [d for d, c in zip(dets, ids) if int(c) in target_ids]

        # Debounce: only the two state TRANSITIONS reach the action hook.
        if mobiles:
            hits, misses = hits + 1, 0
        else:
            hits, misses = 0, misses + 1
        if not present and hits >= args.on_frames:
            present = True
            on_mobile_event("mobile_detected", idx, mobiles, dets, frame, args)
        elif present and misses >= args.off_frames:
            present = False
            on_mobile_event("clear", idx, mobiles, dets, frame, args)

    cap.release()
    print(f"source ended after {idx} frame(s)", flush=True)


if __name__ == "__main__":
    main()
