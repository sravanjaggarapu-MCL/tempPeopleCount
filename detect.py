"""
detect.py - inference entry point for the HEAD DETECTION project.

WHY THIS FILE EXISTS: the post-training evaluation stage runs
eval_visual_sample.py, which looks for detect.py at the repo root and renders
the annotated "what did the model actually find?" sample shown next to the
run's metrics. Without it the stage logs, non-fatally:

    eval_visual_sample.py failed (non-fatal, continuing without visual
    sample): detect.py not found at '<repo>/detect.py'

and the run has numbers but no picture - which for a small-object project
like this one is exactly the artifact you want, because heads the model
missed in the back of a crowd, or boxes it hallucinated on shoulders and
headrests, are far easier to SEE than to read off an mAP.

Same architecture dispatch as train.py/export.py: the family comes from the
checkpoint itself ('models.yolo' -> yolov5, 'ultralytics.nn' -> ultralytics),
with the basename convention as the fallback.

  yolov5      -> DELEGATE: re-exec $YOLOV5_ROOT/detect.py with this argv,
                 verbatim. Delegation matters more here than anywhere else:
                 the evaluation stage's exact flags are its own business, and
                 YOLOv5's detect.py already accepts all of them.
  ultralytics -> NATIVE: YOLO(weights).predict(..., save=True)

Unrecognized flags are passed through to the yolov5 delegate untouched and
ignored on the ultralytics path, so a change to the eval stage's argv cannot
break this file.
"""
# Author: Sathish (sathish.c@magellanic-cloud.com) - Magellanic Cloud, TrainX
# Customized for the HEAD DETECTION project (person's head; CVAT label 'head').
import argparse
import os
import sys
import zipfile

YOLOV5_ROOT_DEFAULT = "/home/ivis/projects/Human-Detection/code"
ULTRALYTICS_PREFIXES = ("yolov8", "yolo11", "yolo26")


def parse_args():
    """parse_known_args, deliberately: this script's job is to run whatever
    the evaluation stage asks for, not to police its flags."""
    p = argparse.ArgumentParser(
        description="TrainX multi-architecture inference entry point")
    p.add_argument("--weights", required=True, help="trained checkpoint (.pt)")
    p.add_argument("--source", required=True, help="image / directory / video")
    p.add_argument("--img", "--imgsz", "--img-size", type=int, default=320,
                   dest="img", help="inference image size")
    p.add_argument("--conf-thres", "--conf", type=float, default=0.25,
                   dest="conf", help="confidence threshold")
    p.add_argument("--project", help="parent dir for output")
    p.add_argument("--name", help="output directory name")
    p.add_argument("--exist-ok", action="store_true", help="reuse output dir")
    p.add_argument("--save-txt", action="store_true", help="also write .txt labels")
    args, extra = p.parse_known_args()
    return args, extra


def log(msg):
    print(f"[detect.py] {msg}", flush=True)


def fail(msg):
    print(f"[detect.py] ERROR: {msg}", file=sys.stderr, flush=True)
    return 1


def sniff_checkpoint_family(weights):
    """'yolov5' | 'ultralytics' | None. 'models.yolo' is tested FIRST - see
    export.py's twin for why the reverse order breaks locally-trained YOLOv5
    checkpoints (their pickled git remote URL contains 'ultralytics')."""
    try:
        with zipfile.ZipFile(weights) as zf:
            pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            raw = zf.read(pkl) if pkl else b""
    except Exception:
        return None
    if b"models.yolo" in raw:
        return "yolov5"
    if b"ultralytics.nn" in raw or b"ultralytics" in raw:
        return "ultralytics"
    return None


def weights_family(weights):
    family = sniff_checkpoint_family(weights)
    if family:
        return family, "checkpoint contents"
    base = os.path.basename(str(weights)).lower()
    if base.startswith("yolov5"):
        return "yolov5", f"basename {base!r}"
    if base.startswith(ULTRALYTICS_PREFIXES):
        return "ultralytics", f"basename {base!r}"
    return None, f"neither checkpoint contents nor basename {base!r}"


def detect_yolov5_delegate(argv):
    root = os.environ.get("YOLOV5_ROOT") or YOLOV5_ROOT_DEFAULT
    target = os.path.join(root, "detect.py")
    if not os.path.isfile(target):
        if os.environ.get("YOLOV5_ROOT"):
            via = f"YOLOV5_ROOT={os.environ['YOLOV5_ROOT']}"
        else:
            via = f"YOLOV5_ROOT is unset; tried the default {YOLOV5_ROOT_DEFAULT}"
        return fail(
            "yolov5 checkpoints delegate inference to the shared YOLOv5 "
            f"checkout, but {target} does not exist ({via}). Set the "
            "YOLOV5_ROOT environment variable to a YOLOv5 checkout (the "
            "TrainX training host exports it)."
        )
    log(f"delegating to {target}")
    os.execv(sys.executable, [sys.executable, target] + argv)


def detect_ultralytics(args):
    try:
        from ultralytics import YOLO  # lazy: only this family needs it
    except ImportError as e:
        return fail(
            "this checkpoint needs the 'ultralytics' package, which is not "
            f"importable here ({e}). Install it into the project venv: "
            "pip install ultralytics"
        )
    kwargs = dict(source=args.source, imgsz=args.img, conf=args.conf,
                  save=True, save_txt=args.save_txt, exist_ok=args.exist_ok)
    if args.project:
        kwargs["project"] = args.project
    if args.name:
        kwargs["name"] = args.name
    YOLO(args.weights).predict(**kwargs)
    return 0


def main():
    args, extra = parse_args()
    if extra:
        log(f"passing through unrecognized args: {extra}")
    if not os.path.isfile(args.weights):
        return fail(f"--weights {args.weights!r} does not exist")
    family, how = weights_family(args.weights)
    log(f"weights={args.weights} -> family={family or 'UNKNOWN'} ({how})")
    if family == "yolov5":
        return detect_yolov5_delegate(sys.argv[1:])
    if family == "ultralytics":
        return detect_ultralytics(args)
    return fail(
        f"cannot determine the model family of --weights {args.weights!r}. "
        "Supported: YOLOv5 checkpoints (delegated to the YOLOV5_ROOT "
        "checkout's detect.py) and ultralytics yolov8/yolo11/yolo26 "
        "checkpoints (via the ultralytics package)."
    )


if __name__ == "__main__":
    sys.exit(main())
