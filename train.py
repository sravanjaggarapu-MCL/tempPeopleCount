"""
train.py - TrainX training entry point for the MOBILE-IN-FRAME detection
project (2 CVAT labels: person, mobile).

CONTRACT FIRST: this file keeps the orchestrator's argv and output contract
byte-for-byte. It accepts exactly the flags run_training.sh Step 3 passes and
writes results.csv + weights/best.pt into <--project>/<--name>/. It dispatches
on the family of --weights:

  yolov5*  -> DELEGATE: re-exec the shared YOLOv5 checkout's train.py with
      this same argv, verbatim. The checkout is named by the YOLOV5_ROOT
      environment variable, which the TrainX training host exports (default
      /home/ivis/projects/Human-Detection/code). YOLOv5's own train.py
      accepts this argv as-is, so nothing is translated.

  yolov8* / yolo11* / yolo26* -> NATIVE: train with the `ultralytics`
      package, then rewrite the run's results.csv into YOLOv5 column names
      (metrics/precision, metrics/recall, metrics/mAP_0.5,
      metrics/mAP_0.5:0.95) - the TrainX acceptance gate reads exactly those
      columns from the last row - and ensure weights/best.pt exists.

  anything else (e.g. a warm-start best.pt from an earlier run) -> the
      family is sniffed from the checkpoint file itself: a torch .pt is a
      zip archive whose pickle payload names the model's modules as plain
      strings ('ultralytics.*' vs 'models.yolo').

PROJECT-SPECIFIC PART - the dataset preflight (see preflight_dataset below).
Before a single epoch runs it parses the generated --data yaml and prints
what the run is ACTUALLY about to train on: nc/names straight from CVAT, the
image count per split, and a per-class box tally over the train split. It
warns - never fails - on the three things that quietly ruin a two-class
person/mobile run:

  * 'mobile' present in names but with ZERO boxes in the train split (the
    CVAT label exists but nobody drew the small boxes yet - the run will
    happily finish with mAP that looks fine because 'person' carries it)
  * label .txt files missing for images (the images/ -> labels/ rewrite did
    not resolve, so those images silently train as backgrounds)
  * a class id in the labels that is >= nc (the annotations and the CVAT
    label list are out of sync)

NEVER hardcode nc/names here. The class list is regenerated from the
project's CVAT labels on EVERY run; this file only reports what it was given.
Set TRAINX_SKIP_PREFLIGHT=1 to skip the scan entirely.

INVOCATION (what the orchestrator actually runs - see run_training.sh):
    <project-venv>/bin/python train.py \
        --img 640 --batch 8 --epochs 100 \
        --data /trainx/workspaces/<project>/<dataset_id>/dataset.yaml \
        --weights <repo>/yolov5s.pt \
        --project /trainx/runs/<project_id> --name run-<ts>-<suffix> \
        --seed 0 --exist-ok

INPUT: --data is a GENERATED YOLO dataset yaml (works for both families).
train:/val: are newline-delimited lists of absolute image paths (approved
dataset version under /trainx/datasets/<project>/master/versions/<id>/);
labels resolve the standard YOLO way (images/ -> labels/).

REQUIRED OUTPUTS in <--project>/<--name>/ (the pipeline fails without them):
    results.csv       per-epoch metrics, YOLOv5 column convention - the last
                      row's metrics/precision, metrics/recall, metrics/mAP_0.5,
                      metrics/mAP_0.5:0.95 feed the human acceptance gate
    weights/best.pt   the trained checkpoint (exported, quantized, published,
                      and the warm-start source for later runs)

Exit 0 on success. MLflow logging is done FOR you afterwards
(log_run_to_mlflow.py) - do not add your own.
"""
# Author: Sathish (sathish.c@magellanic-cloud.com) - Magellanic Cloud, TrainX
# Customized for the mobile-in-frame detection project (labels: person, mobile).
import argparse
import csv
import os
import shutil
import sys
import zipfile

# Where the shared YOLOv5 checkout lives when YOLOV5_ROOT is not exported.
# The TrainX training host exports YOLOV5_ROOT (run_training.sh and
# export_and_quantize.sh both do); the default below is that host's real
# checkout so this file also works from a bare shell on the box.
YOLOV5_ROOT_DEFAULT = "/home/ivis/projects/Human-Detection/code"

# Weight basenames that select the ultralytics package natively.
ULTRALYTICS_PREFIXES = ("yolov8", "yolo11", "yolo26")

# ultralytics writes results.csv with (B)-suffixed detection-metric columns;
# the TrainX acceptance gate reads YOLOv5's names. This is the whole mapping -
# every other column passes through untouched.
RESULTS_CSV_COLUMN_MAP = {
    "metrics/precision(B)": "metrics/precision",
    "metrics/recall(B)": "metrics/recall",
    "metrics/mAP50(B)": "metrics/mAP_0.5",
    "metrics/mAP50-95(B)": "metrics/mAP_0.5:0.95",
}

# The four columns the human Model Acceptance Gate reads from the LAST row.
GATE_COLUMNS = ("metrics/precision", "metrics/recall",
                "metrics/mAP_0.5", "metrics/mAP_0.5:0.95")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Scanning every label file of a large dataset is pointless - a few thousand
# are more than enough to tell "the mobile class has no boxes" from "the
# mobile class is just rarer than person".
LABEL_SCAN_LIMIT = 10000

# A detector needs boxes, not images. An autolabel export can easily be 1000
# images of which only a hundred carry a box - which trains almost nothing
# while looking like a healthy 1000-image dataset in the dashboard. These two
# thresholds are what turn that into a warning instead of a wasted run.
LOW_INSTANCE_WARN = 200      # boxes per class below which the class is too thin
HIGH_BACKGROUND_WARN = 0.60  # fraction of label-less images that is suspicious


def parse_args():
    # EXACT contract with run_training.sh Step 3 - never rename or drop a
    # flag here; the orchestrator passes every one of them.
    p = argparse.ArgumentParser(
        description="TrainX multi-architecture training entry point")
    p.add_argument("--img", "--imgsz", type=int, default=320, help="image size")
    p.add_argument("--batch", "--batch-size", type=int, default=8, help="batch size")
    p.add_argument("--epochs", type=int, default=2, help="epochs")
    p.add_argument("--data", required=True, help="generated dataset.yaml (see module docstring)")
    p.add_argument("--weights", required=True,
                   help="starting checkpoint (.pt) - its family selects the architecture")
    p.add_argument("--project", required=True, help="parent dir for run output")
    p.add_argument("--name", required=True, help="run directory name")
    p.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    p.add_argument("--exist-ok", action="store_true", help="reuse existing run dir")
    return p.parse_args()


def log(msg):
    print(f"[train.py] {msg}", flush=True)


def warn(msg):
    """Loud but NON-fatal. The orchestrator shows stdout verbatim in the stage
    log, so a preflight warning is the cheapest place to catch a dataset
    problem - but it must never be the reason a run does not start."""
    print(f"[train.py] WARNING: {msg}", flush=True)


def fail(msg):
    """One honest, greppable error line on stderr; returns the exit code so
    callers can `return fail(...)`. The orchestrator's stage log shows stderr
    verbatim, so the message must say exactly what to fix."""
    print(f"[train.py] ERROR: {msg}", file=sys.stderr, flush=True)
    return 1


# --------------------------------------------------------------------------
# Dataset preflight - project-specific, advisory only
# --------------------------------------------------------------------------

def normalize_names(raw, nc):
    """CVAT-generated yamls carry names as a list; ultralytics also accepts a
    {id: name} dict. Return an id-indexed list either way, padded to nc so a
    tally can still be printed when the two disagree."""
    if isinstance(raw, dict):
        names = [str(raw[k]) for k in sorted(raw, key=lambda k: int(k))]
    elif isinstance(raw, (list, tuple)):
        names = [str(n) for n in raw]
    else:
        names = []
    if isinstance(nc, int) and nc > len(names):
        names += ["<class %d>" % i for i in range(len(names), nc)]
    return names


def resolve_split(entry, base_dir):
    """Turn a dataset yaml train:/val: entry into (image_paths, unresolved).

    The TrainX pipeline writes newline-delimited absolute image-path lists and
    points train:/val: at those .txt files, but plain YOLO directories and
    lists of either are equally legal, so all three are handled."""
    entries = entry if isinstance(entry, (list, tuple)) else [entry]
    images, unresolved = [], []
    for raw in entries:
        p = str(raw)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base_dir, p))
        if os.path.isfile(p) and p.lower().endswith(".txt"):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        images.append(line)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                images += [os.path.join(root, fn) for fn in sorted(files)
                           if fn.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            unresolved.append(p)
    return images, unresolved


def label_path_for(image_path):
    """The standard YOLO rule: replace the LAST '/images/' segment with
    '/labels/' and swap the extension for .txt. This mirrors YOLOv5's own
    img2label_paths, so a mismatch here is a real mismatch at training time."""
    p = image_path.replace("\\", "/")
    head, sep, tail = p.rpartition("/images/")
    if sep:
        p = head + "/labels/" + tail
    return os.path.splitext(p)[0] + ".txt"


def scan_labels(images, nc):
    """Tally boxes per class id over (a sample of) a split's label files.

    Returns (counts, stats). counts maps class id -> box count, including ids
    >= nc, which is exactly the out-of-sync case worth shouting about."""
    counts, stats = {}, {"scanned": 0, "missing": 0, "empty": 0, "malformed": 0}
    for img in images[:LABEL_SCAN_LIMIT]:
        lbl = label_path_for(img)
        if not os.path.isfile(lbl):
            stats["missing"] += 1
            continue
        stats["scanned"] += 1
        boxes = 0
        try:
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        stats["malformed"] += 1
                        continue
                    counts[cid] = counts.get(cid, 0) + 1
                    boxes += 1
        except OSError:
            stats["malformed"] += 1
            continue
        if boxes == 0:
            stats["empty"] += 1
    return counts, stats


def format_tally(counts, names, nc):
    """Every declared class (so a zero shows up loudly) plus any undeclared id
    that actually appeared - but nothing in between, which would just be a row
    of noisy zeroes when the labels use an out-of-range id."""
    parts = []
    for i in sorted(set(range(nc)) | set(counts)):
        name = names[i] if i < len(names) else "<undeclared class %d>" % i
        parts.append(f"{name}={counts.get(i, 0)}")
    return "  ".join(parts)


def preflight_dataset(data_yaml):
    """Print - and sanity-check - the dataset this run was handed. Advisory
    only: every problem is a warning, never an exit code, because the yaml is
    generated by the pipeline and a false positive here must not block a run."""
    if os.environ.get("TRAINX_SKIP_PREFLIGHT"):
        log("dataset preflight skipped (TRAINX_SKIP_PREFLIGHT is set)")
        return
    log("---- dataset preflight -------------------------------------------")
    log(f"data yaml : {data_yaml}")
    if not os.path.isfile(data_yaml):
        warn(f"{data_yaml} does not exist - training will fail on it in a moment")
        return
    try:
        import yaml  # PyYAML ships with the YOLOv5 requirements in the shared venv
    except ImportError as e:
        warn(f"PyYAML not importable ({e}); skipping the preflight - training continues")
        return
    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        warn(f"could not parse {data_yaml} ({e}); skipping the preflight")
        return

    nc = cfg.get("nc")
    names = normalize_names(cfg.get("names"), nc)
    if not isinstance(nc, int):
        nc = len(names)
    log(f"classes   : nc={nc} names={names}")
    if nc != 2:
        warn(f"this project is annotated with TWO CVAT labels (person, mobile) "
             f"but the generated yaml says nc={nc}. Check the project's label "
             f"list in CVAT - nc/names are regenerated from it every run.")

    # 'path:' is optional in YOLO yamls; relative train:/val: resolve against it.
    base_dir = str(cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml)))
    resolved = {}
    for split in ("train", "val"):
        entry = cfg.get(split)
        if entry is None:
            warn(f"the yaml has no '{split}:' key")
            continue
        images, unresolved = resolve_split(entry, base_dir)
        resolved[split] = images
        log(f"{split:<10}: {len(images)} images  (from {entry})")
        if unresolved:
            warn(f"'{split}:' entries that are neither a list file nor a "
                 f"directory: {unresolved}")
        if not images:
            warn(f"the {split} split resolved to ZERO images - training cannot "
                 f"learn from it")

    # Leakage: the pipeline normally writes two disjoint split lists, but a
    # legacy/unpromoted dataset version can leave train: and val: pointing at
    # the same images. Training still "works" and the reported mAP is then
    # measured on the training data, so it says nothing about held-out
    # performance - the single most misleading thing a run can do.
    train_images = resolved.get("train") or []
    val_images = resolved.get("val") or []
    if train_images and val_images:
        overlap = len(set(train_images) & set(val_images))
        if overlap:
            pct = 100.0 * overlap / len(val_images)
            same = overlap == len(val_images) == len(train_images)
            warn(f"{overlap} image(s) ({pct:.0f}% of val) appear in BOTH the "
                 f"train and val splits"
                 + (" - they are the SAME image set." if same else ".")
                 + " Validation metrics will NOT reflect held-out performance; "
                   "mAP is being measured on training data. Re-promote this "
                   "dataset version so it gets a real persisted split.")

    if not train_images:
        log("------------------------------------------------------------------")
        return

    counts, stats = scan_labels(train_images, nc)
    sampled = min(len(train_images), LABEL_SCAN_LIMIT)
    suffix = (f" (sample of {sampled}/{len(train_images)})"
              if sampled < len(train_images) else "")
    log(f"box tally : {format_tally(counts, names, nc)}{suffix}")
    log(f"label files: {stats['scanned']} read, {stats['missing']} missing, "
        f"{stats['empty']} empty (background), {stats['malformed']} malformed")

    if stats["missing"]:
        warn(f"{stats['missing']} image(s) in the train split have NO label .txt "
             f"next to them (images/ -> labels/ did not resolve). Those images "
             f"train as pure background - if that was not intended, re-check the "
             f"dataset version's labels/ tree.")

    # A pile of images is not a pile of boxes. An autolabel export of 1000
    # frames in which the labeller only found something in 98 of them trains
    # on 98 images, and the dashboard still says "1000 images".
    blank = stats["missing"] + stats["empty"]
    total_files = stats["scanned"] + stats["missing"]
    if total_files and blank / total_files >= HIGH_BACKGROUND_WARN:
        warn(f"{blank}/{total_files} train images ({100.0 * blank / total_files:.0f}%) "
             f"contain NO boxes at all - only {total_files - blank} image(s) carry "
             f"any annotation. Some background frames help precision, but at this "
             f"ratio the model is training on almost nothing. If this is an "
             f"autolabel export, the auto-labeller found objects in very few "
             f"frames: review/correct it in CVAT and re-promote before spending "
             f"another run.")
    total_boxes = sum(counts.values())
    if total_boxes and total_boxes < LOW_INSTANCE_WARN * max(nc, 1):
        log(f"note      : {total_boxes} boxes total across {nc} class(es)")
    for cid in range(nc):
        n_boxes = counts.get(cid, 0)
        cname = names[cid] if cid < len(names) else "<class %d>" % cid
        if 0 < n_boxes < LOW_INSTANCE_WARN:
            warn(f"class '{cname}' (id {cid}) has only {n_boxes} boxes in the "
                 f"train split. A detector needs hundreds to low thousands of "
                 f"instances per class; expect near-zero precision and unstable "
                 f"mAP below that. Annotate more frames containing '{cname}' "
                 f"rather than tuning epochs or learning rate.")
    for cid in range(nc):
        if counts.get(cid, 0) == 0:
            cname = names[cid] if cid < len(names) else "<class %d>" % cid
            warn(f"class '{cname}' (id {cid}) has ZERO boxes in the train split. "
                 f"The run will still finish and report mAP, but the model can "
                 f"never detect it. For this project that usually means the "
                 f"'mobile' boxes have not been drawn/approved in CVAT yet.")
    stray = sorted(c for c in counts if c >= nc)
    if stray:
        warn(f"label files contain class id(s) {stray} but nc={nc}. The "
             f"annotations and the CVAT label list are out of sync - training "
             f"will error or mislabel. Fix the label list before rerunning.")
    log("------------------------------------------------------------------")


# --------------------------------------------------------------------------
# Architecture dispatch
# --------------------------------------------------------------------------

def sniff_checkpoint_family(weights):
    """Family of an EXISTING .pt by looking inside it: a torch checkpoint is
    a zip archive whose pickle payload names the model classes' modules as
    plain byte strings, so b'ultralytics' / b'models.yolo' identify the
    framework without loading - or even having installed - either one.
    Returns 'yolov5' | 'ultralytics' | None.

    ORDER IS LOAD-BEARING. A YOLOv5 checkpoint trained from a git clone of
    ultralytics/yolov5 pickles check_git_info() alongside the model, and that
    dict carries the remote URL 'https://github.com/ultralytics/yolov5' - so
    the bare substring 'ultralytics' appears in YOLOv5 checkpoints too.
    Testing for it first mislabels every locally-trained YOLOv5 best.pt as an
    ultralytics model, which breaks warm starts (and, in export.py's twin of
    this function, broke a real export stage).

    'models.yolo' is checked FIRST because it is unambiguous: only YOLOv5
    pickles its model class from the top-level models.yolo module. Then
    'ultralytics.nn' - what a real yolov8/yolo11/yolo26 checkpoint pickles -
    which, unlike bare 'ultralytics', cannot match a URL."""
    try:
        with zipfile.ZipFile(weights) as zf:
            pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            raw = zf.read(pkl) if pkl else b""
    except Exception:
        # Not a readable torch zip (corrupt, or not a checkpoint at all) -
        # the caller falls back to its other evidence.
        return None
    if b"models.yolo" in raw:
        return "yolov5"
    if b"ultralytics.nn" in raw or b"ultralytics" in raw:
        return "ultralytics"
    return None


def weights_family(weights):
    """(family, how) for the --weights argument. Basename convention first -
    it works for a COLD start whose checkpoint file does not exist yet and
    will be auto-downloaded by the framework - then the checkpoint sniff,
    which handles a WARM start named best.pt."""
    base = os.path.basename(str(weights)).lower()
    if base.startswith("yolov5"):
        return "yolov5", f"basename {base!r}"
    if base.startswith(ULTRALYTICS_PREFIXES):
        return "ultralytics", f"basename {base!r}"
    if os.path.isfile(weights):
        family = sniff_checkpoint_family(weights)
        if family:
            return family, f"checkpoint contents of {base!r}"
    return None, f"basename {base!r}"


def train_yolov5_delegate(argv):
    """Re-exec the shared YOLOv5 checkout's train.py with OUR argv, verbatim.
    os.execv REPLACES this process, so YOLOv5's exit code IS this script's
    exit code and its output streams straight into the orchestrator's stage
    log - no wrapping, no re-parsing, nothing to drift."""
    root = os.environ.get("YOLOV5_ROOT") or YOLOV5_ROOT_DEFAULT
    target = os.path.join(root, "train.py")
    if not os.path.isfile(target):
        if os.environ.get("YOLOV5_ROOT"):
            via = f"YOLOV5_ROOT={os.environ['YOLOV5_ROOT']}"
        else:
            via = f"YOLOV5_ROOT is unset; tried the default {YOLOV5_ROOT_DEFAULT}"
        return fail(
            "yolov5 weights delegate training to the shared YOLOv5 checkout, "
            f"but {target} does not exist ({via}). Set the YOLOV5_ROOT "
            "environment variable to a YOLOv5 checkout (the TrainX training "
            "host exports it) or vendor a YOLOv5 fork into this repo."
        )
    log(f"delegating to {target}")
    os.execv(sys.executable, [sys.executable, target] + argv)


def report_gate_metrics(header, body):
    """Echo the exact four numbers the human acceptance gate will read, so the
    stage log answers 'was this run any good?' without opening results.csv."""
    if not body:
        warn("results.csv has a header but no data rows - the acceptance gate "
             "will see no metrics")
        return
    last = dict(zip(header, body[-1]))
    missing = [c for c in GATE_COLUMNS if c not in last]
    if missing:
        warn(f"results.csv is missing the acceptance-gate column(s) {missing} - "
             f"the gate reads {list(GATE_COLUMNS)} from the last row")
        return
    log("acceptance-gate metrics (last row): "
        + "  ".join(f"{c.split('/')[-1]}={last[c]}" for c in GATE_COLUMNS))


def rewrite_results_csv(run_dir):
    """Rewrite ultralytics' results.csv header into YOLOv5 column names, in
    place. Without this the acceptance gate - which reads the LAST row's
    metrics/precision, metrics/recall, metrics/mAP_0.5, metrics/mAP_0.5:0.95 -
    sees no metrics at all, because ultralytics names them metrics/mAP50(B)
    etc. Cells are stripped because both frameworks pad columns with spaces."""
    path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(path):
        return fail(f"training finished but {path} was not written - "
                    "the pipeline requires it")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return fail(f"{path} is empty - the pipeline requires per-epoch metrics in it")
    header = [RESULTS_CSV_COLUMN_MAP.get(c.strip(), c.strip()) for c in rows[0]]
    body = [[c.strip() for c in r] for r in rows[1:]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    log(f"rewrote {path} header into YOLOv5 column names")
    report_gate_metrics(header, body)
    return 0


def ensure_best_pt(run_dir):
    """weights/best.pt is what the export stage loads and what later runs
    warm-start from. ultralytics writes it on every normal run; in edge cases
    (e.g. validation disabled) only last.pt appears, and copying it is the
    honest equivalent - it IS the final checkpoint."""
    weights_dir = os.path.join(run_dir, "weights")
    best = os.path.join(weights_dir, "best.pt")
    last = os.path.join(weights_dir, "last.pt")
    if os.path.isfile(best):
        return 0
    if os.path.isfile(last):
        shutil.copy2(last, best)
        log(f"best.pt was missing; copied last.pt -> {best}")
        return 0
    return fail(f"training finished but neither best.pt nor last.pt exists "
                f"under {weights_dir} - the pipeline requires weights/best.pt")


def train_ultralytics(args):
    """Native yolov8/yolo11/yolo26 training via the ultralytics package, then
    the two platform-contract fixups: results.csv column names and
    weights/best.pt."""
    try:
        from ultralytics import YOLO  # lazy: only this family needs it
    except ImportError as e:
        return fail(
            "this weights family needs the 'ultralytics' package, which is "
            f"not importable here ({e}). Install it into the project venv: "
            "pip install ultralytics  (coordinate with the MLOps engineer - "
            "the training venv is shared)."
        )
    # Cold start: the platform passes <repo>/<variant>.pt which may not exist
    # yet - hand ultralytics the bare name so it auto-downloads, exactly like
    # YOLOv5 does for its own variants.
    weights = args.weights if os.path.isfile(args.weights) else os.path.basename(args.weights)
    model = YOLO(weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.img, batch=args.batch,
        project=args.project, name=args.name, seed=args.seed,
        exist_ok=args.exist_ok,
    )
    # ultralytics honors project/name the same way YOLOv5 does; still prefer
    # its own record of where it actually wrote, when available.
    run_dir = os.path.join(args.project, args.name)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        run_dir = str(save_dir)
    rc = rewrite_results_csv(run_dir)
    if rc:
        return rc
    return ensure_best_pt(run_dir)


def main():
    args = parse_args()
    family, how = weights_family(args.weights)
    log(f"weights={args.weights} -> family={family or 'UNKNOWN'} ({how})")
    log(f"img={args.img} batch={args.batch} epochs={args.epochs} seed={args.seed}")
    log(f"run dir   : {os.path.join(args.project, args.name)}")
    if args.img < 640:
        warn(f"--img {args.img}: a mobile phone is a SMALL object, often under "
             f"5% of the frame height. 640 detects phones far more reliably "
             f"than 320 - raise 'Image size' in the wizard unless the edge "
             f"device cannot afford it.")
    # Advisory - runs for BOTH families, and before the yolov5 delegate's
    # execv replaces this process.
    preflight_dataset(args.data)
    if family == "yolov5":
        return train_yolov5_delegate(sys.argv[1:])
    if family == "ultralytics":
        return train_ultralytics(args)
    return fail(
        f"cannot determine the model family from --weights {args.weights!r}. "
        "Supported: yolov5* (delegates to the YOLOV5_ROOT checkout) and "
        "yolov8*/yolo11*/yolo26* (trained via the ultralytics package); a "
        "warm-start best.pt is recognized by reading the checkpoint itself, "
        "which did not identify either family here."
    )


if __name__ == "__main__":
    sys.exit(main())
