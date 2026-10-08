"""
export.py - TrainX export entry point for the MOBILE-IN-FRAME detection
project (2 CVAT labels: person, mobile).

The post-training export/quantize stage (export_and_quantize.sh Step 2) runs,
verbatim:

    <project-venv>/bin/python export.py --weights <run_dir>/weights/best.pt \
        --img <size> --include onnx --data <dataset.yaml>

and REQUIRES best.onnx to appear NEXT TO the --weights file. It may also be
re-invoked with --include openvino / --include tflite (best-effort formats -
a failure there is logged by the stage, not fatal). INT8 quantization of the
ONNX graph is done by the pipeline itself (onnxruntime), never here.

ARCHITECTURE DISPATCH: at export time --weights is a trained best.pt, so the
basename says nothing - the family is sniffed from the CHECKPOINT ITSELF
(a torch .pt is a zip archive whose pickle payload names the model's modules:
'ultralytics.*' vs 'models.yolo'), with the basename convention
(yolov5* / yolov8* / yolo11* / yolo26*) as the fallback for bare pretrained
files.

  yolov5      -> DELEGATE: re-exec $YOLOV5_ROOT/export.py with this same argv
                 (the TrainX host exports YOLOV5_ROOT; default
                 /home/ivis/projects/Human-Detection/code)
  ultralytics -> NATIVE: YOLO(weights).export(format=..., imgsz=...) - the
                 package writes best.onnx next to the weights, exactly where
                 the pipeline looks for it

TWO PROJECT-SPECIFIC ADDITIONS on the ultralytics path:

  * onnx is FATAL, everything else is BEST-EFFORT. A single --include list
    such as "onnx openvino" used to fail the whole stage when openvino blew
    up even though the load-bearing best.onnx had already been written.
  * a model_package.json is written next to the weights when one is not
    already there, carrying img_size + the class names read from --data. That
    is the file deploy/edge_client.py reads to map class id -> "mobile", so a
    locally exported model can be shipped to a device without hand-writing it.
    The platform's own Artifacts-panel model_package.json is authoritative -
    this one only fills the gap and never overwrites.
"""
# Author: Sathish (sathish.c@magellanic-cloud.com) - Magellanic Cloud, TrainX
# Customized for the mobile-in-frame detection project (labels: person, mobile).
import argparse
import json
import os
import sys
import zipfile

# Duplicated from train.py ON PURPOSE: each generated file must stand alone
# (the export stage runs export.py in isolation; importing train.py would
# couple the two files' lifetimes for three tiny helpers).
YOLOV5_ROOT_DEFAULT = "/home/ivis/projects/Human-Detection/code"
ULTRALYTICS_PREFIXES = ("yolov8", "yolo11", "yolo26")

# The one format the pipeline actually gates on.
REQUIRED_FORMAT = "onnx"


def parse_args():
    # EXACT contract with export_and_quantize.sh Step 2 (and its best-effort
    # openvino/tflite re-invocations) - never rename or drop a flag here.
    p = argparse.ArgumentParser(
        description="TrainX multi-architecture export entry point")
    p.add_argument("--weights", required=True, help="trained checkpoint (.pt)")
    p.add_argument("--img", "--imgsz", type=int, default=320, help="export image size")
    p.add_argument("--include", nargs="+", default=["onnx"],
                   help="formats: onnx/openvino/tflite")
    p.add_argument("--data", help="dataset.yaml for shape inference")
    return p.parse_args()


def log(msg):
    print(f"[export.py] {msg}", flush=True)


def warn(msg):
    print(f"[export.py] WARNING: {msg}", flush=True)


def fail(msg):
    print(f"[export.py] ERROR: {msg}", file=sys.stderr, flush=True)
    return 1


def sniff_checkpoint_family(weights):
    """'yolov5' | 'ultralytics' | None by reading the torch zip's pickle
    payload - see train.py's twin of this function for the full rationale.

    ORDER IS LOAD-BEARING, and getting it wrong broke a real run. A YOLOv5
    checkpoint trained from a git clone of ultralytics/yolov5 pickles
    check_git_info() alongside the model, and that dict carries the remote
    URL 'https://github.com/ultralytics/yolov5' - so the bare substring
    'ultralytics' is present in a YOLOv5 checkpoint too, and testing for it
    first mislabels every locally-trained YOLOv5 best.pt as an ultralytics
    model. The export then hands it to YOLO() and dies with 'appears to be an
    Ultralytics YOLOv5 model ... NOT forwards compatible with YOLOv8'.

    'models.yolo' is checked FIRST because it is unambiguous: only YOLOv5
    pickles its model class from the top-level models.yolo module. Then
    'ultralytics.nn' - the module a real yolov8/yolo11/yolo26 checkpoint
    pickles - which, unlike bare 'ultralytics', cannot match a URL."""
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
    """(family, how). Checkpoint sniff FIRST - unlike train.py, the weights
    file always exists here (export runs after training) and its basename is
    just 'best.pt' - then the basename convention for bare pretrained files."""
    family = sniff_checkpoint_family(weights)
    if family:
        return family, "checkpoint contents"
    base = os.path.basename(str(weights)).lower()
    if base.startswith("yolov5"):
        return "yolov5", f"basename {base!r}"
    if base.startswith(ULTRALYTICS_PREFIXES):
        return "ultralytics", f"basename {base!r}"
    return None, f"neither checkpoint contents nor basename {base!r}"


def export_yolov5_delegate(argv):
    """Re-exec the shared YOLOv5 checkout's export.py with OUR argv, verbatim
    (its export.py takes --weights/--img/--include/--data as-is and writes
    best.onnx next to the weights). os.execv keeps exit code and output
    honest - see train.py's delegate for the rationale."""
    root = os.environ.get("YOLOV5_ROOT") or YOLOV5_ROOT_DEFAULT
    target = os.path.join(root, "export.py")
    if not os.path.isfile(target):
        if os.environ.get("YOLOV5_ROOT"):
            via = f"YOLOV5_ROOT={os.environ['YOLOV5_ROOT']}"
        else:
            via = f"YOLOV5_ROOT is unset; tried the default {YOLOV5_ROOT_DEFAULT}"
        return fail(
            "yolov5 checkpoints delegate export to the shared YOLOv5 checkout, "
            f"but {target} does not exist ({via}). Set the YOLOV5_ROOT "
            "environment variable to a YOLOv5 checkout (the TrainX training "
            "host exports it) or vendor a YOLOv5 fork into this repo."
        )
    log(f"delegating to {target}")
    os.execv(sys.executable, [sys.executable, target] + argv)


def class_names_from_data(data_yaml):
    """The class list for the deployment manifest, read from the GENERATED
    dataset yaml - i.e. straight from CVAT, in the id order the model was
    trained with. Never hardcode ['person', 'mobile'] here: the ORDER is
    whatever the project's CVAT label list says, and the edge client maps
    id -> name through exactly this list."""
    if not data_yaml or not os.path.isfile(data_yaml):
        return []
    try:
        import yaml
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        warn(f"could not read class names from {data_yaml} ({e})")
        return []
    raw = cfg.get("names")
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda k: int(k))]
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    return []


def write_model_package(weights, img, names):
    """Best-effort deployment manifest next to the weights, for
    deploy/edge_client.py. Never overwrites: the platform's Artifacts-panel
    model_package.json is the authoritative one."""
    path = os.path.join(os.path.dirname(os.path.abspath(weights)),
                        "model_package.json")
    if os.path.isfile(path):
        return
    if not names:
        warn("no class names available (--data missing or unreadable); "
             "skipping model_package.json - the edge client needs one, so "
             "take it from the run's Artifacts panel instead")
        return
    try:
        with open(path, "w") as f:
            json.dump({"img_size": img, "names": names,
                       "source_weights": os.path.basename(weights)}, f, indent=2)
    except OSError as e:
        warn(f"could not write {path} ({e})")
        return
    log(f"wrote deployment manifest: {path}  (img_size={img}, names={names})")


def export_ultralytics(args):
    """Native export via the ultralytics package. The stage's format tokens
    (onnx/openvino/tflite) are ultralytics' own format names, so --include is
    forwarded one-for-one; the package writes each artifact next to the
    weights file, which is exactly the contract.

    onnx is the only format whose failure fails this script - openvino and
    tflite are explicitly best-effort in the pipeline, so a broken exporter
    for them must not throw away a good best.onnx."""
    try:
        from ultralytics import YOLO  # lazy: only this family needs it
    except ImportError as e:
        return fail(
            "this checkpoint needs the 'ultralytics' package, which is not "
            f"importable here ({e}). Install it into the project venv: "
            "pip install ultralytics  (coordinate with the MLOps engineer - "
            "the training venv is shared)."
        )
    model = YOLO(args.weights)
    for fmt in args.include:
        try:
            model.export(format=fmt, imgsz=args.img)
        except Exception as e:
            if fmt == REQUIRED_FORMAT:
                return fail(f"ONNX export failed ({e}) - the pipeline requires "
                            "best.onnx next to --weights")
            warn(f"best-effort '{fmt}' export failed ({e}); continuing")
    if REQUIRED_FORMAT in args.include:
        # The load-bearing contract: export_and_quantize.sh dies unless
        # best.onnx exists next to --weights after this script exits 0.
        expected = os.path.splitext(args.weights)[0] + ".onnx"
        if not os.path.isfile(expected):
            return fail(f"ultralytics export finished but {expected} does not "
                        "exist - the pipeline requires the .onnx next to --weights")
        log(f"ONNX written: {expected}")
    write_model_package(args.weights, args.img, class_names_from_data(args.data))
    return 0


def main():
    args = parse_args()
    if not os.path.isfile(args.weights):
        return fail(
            f"--weights {args.weights!r} does not exist - export needs the "
            "trained checkpoint the training stage wrote "
            "(<run_dir>/weights/best.pt)."
        )
    family, how = weights_family(args.weights)
    log(f"weights={args.weights} -> family={family or 'UNKNOWN'} ({how})")
    if family == "yolov5":
        return export_yolov5_delegate(sys.argv[1:])
    if family == "ultralytics":
        return export_ultralytics(args)
    return fail(
        f"cannot determine the model family of --weights {args.weights!r}. "
        "Supported: YOLOv5 checkpoints (delegated to the YOLOV5_ROOT "
        "checkout's export.py) and ultralytics yolov8/yolo11/yolo26 "
        "checkpoints (exported via the ultralytics package)."
    )


if __name__ == "__main__":
    sys.exit(main())
