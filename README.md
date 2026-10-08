# Head detection — TrainX training repository

Detects a **person's head** in a frame. One CVAT label: `head`.

Head boxes rather than full-body `person` boxes are the point of this
project: in a queue, a doorway or a packed room, bodies occlude one another
almost completely while heads stay visible, so a head detector degrades far
more gracefully with crowding — which is what makes it the right base for
counting people.

Based on the TrainX model-repository skeleton
(`GET /api/orchestrator/meta/model-repo-skeleton.zip`).
Author of the skeleton: Sathish (sathish.c@magellanic-cloud.com) — Magellanic Cloud, TrainX

This repo is the **code half** of a TrainX pipeline. It only trains and
exports. Dataset resolution, MLflow logging, DVC/MinIO publishing and lineage
commit-back are all done **for** it by the orchestrator.

---

## Use it

1. `git add . && git commit -m "head detection training repo"`
2. Push to your git remote (Azure DevOps / GitHub / GitLab).
3. Paste the repo URL into the New Pipeline wizard's **Repository** step
   (*Check & Continue* runs a real `git ls-remote`).
4. The repo is cloned at the **first training run**, not at wizard time. The
   stage log names the exact commit trained on.

Nothing else needs editing before the first run. `train.py` and `export.py`
keep the orchestrator's argv and output contract byte-for-byte: train →
`results.csv` → `weights/best.pt` → `best.onnx` → `model_package.json` → edge
inference.

---

## Wizard settings for this project

| Field | Use | Why |
| --- | --- | --- |
| **Image size** | **640** minimum, **960** for wide or crowded views | A head is a *small object* — in a room, corridor or entrance view it is typically 3–8% of frame height, i.e. 10–25 px at 320, which is below what a stride-32 detector can resolve at all. `train.py` warns below 640 **and prints the real median head size of your dataset in pixels at the chosen size** — set this from that number, not from a rule of thumb. |
| **Epochs** | **100–150** | The skeleton default of 2 is a smoke test, not training. |
| **Batch size** | 8–16 | 8 is safe at 640 on a modest GPU; raise it if memory allows. Lower it before lowering Image size — image size is the lever that matters here. |
| **Weights variant** | **`yolov5s`** for the first run | Zero extra dependencies — it delegates to the shared `$YOLOV5_ROOT` checkout, which the training venv already provides. `yolov5n` is faster but weakest on small objects, which is exactly where this project lives. |
| | `yolo11s` / `yolov8s` as an upgrade | Usually better on small objects, **but** requires `pip install ultralytics` on the *shared* training venv — coordinate with the MLOps engineer **before** the run, or it fails immediately with that exact message. |
| **Seed** | 0 | Forwarded verbatim and logged to MLflow. |

Both weight families work — `train.py` dispatches on the `--weights`
basename, and a warm-start `best.pt` is identified by reading the checkpoint
itself.

---

## The class list is NOT in this repo

`nc` and `names` are regenerated from the project's **CVAT label list on every
run** and reach `train.py` only through the generated `--data` yaml. So the id
of `head` is whatever position CVAT gives it, and it can change. Nothing
here hardcodes it:

* `train.py` reads `nc`/`names` to **report** them and to recognize *which*
  label is the head class (see the preflight below) — never to assume an id
* `export.py` copies them into `model_package.json` **in that same id order**
* `deploy/edge_client.py` resolves `head` **by name** through that list

Rename the label to `Person Head` or add a second class in CVAT and the whole
chain still works. See `dataset.yaml.example`.

---

## The dataset preflight

Before epoch 1, `train.py` prints exactly what the run is about to train on:

```
[train.py] ---- dataset preflight -------------------------------------------
[train.py] data yaml : /trainx/workspaces/<project>/<dataset_id>/dataset.yaml
[train.py] classes   : nc=1 names=['head']
[train.py] head class: id 0 = 'head'
[train.py] train     : 4820 images  (from /trainx/workspaces/.../train_split.txt)
[train.py] val       : 1204 images  (from /trainx/workspaces/.../val_split.txt)
[train.py] box tally : head=38412
[train.py] label files: 4820 read, 0 missing, 118 empty (background), 0 malformed
[train.py] box size  : 'head' median height 4.8% of frame = 31px at --img 640  (smallest decile 19px)
[train.py] density   : 8.0 head(s) per annotated train image
[train.py] ------------------------------------------------------------------
```

The two lines that are specific to head detection are the last two, and they
are the ones worth reading first:

* **`box size`** — the median head height expressed **in pixels at this run's
  `--img`**. Below ~16 px there is nothing left after the stride-8/16/32
  downsampling and recall stays low no matter how many epochs are spent, so
  the preflight warns and suggests a concrete Image size.
* **`density`** — heads per annotated image. It tells a sparse office camera
  apart from a packed platform view, and above ~50 it warns that the
  per-image detection cap (`max_det`, default 300) and NMS on touching heads
  will make counts saturate.

It **warns, never fails** — the yaml is generated by the pipeline, so a false
positive here must not block a run:

| Warning | What it actually means |
| --- | --- |
| `none of the CVAT labels [...] looks like a head class` | The pipeline is pointed at the wrong project / label list. Everything head-specific below is then skipped. |
| `class 'head' (id 0) has ZERO boxes in the train split` | The CVAT label exists but nobody drew the boxes yet. **This is the failure to watch for.** |
| `the median 'head' box is only Npx tall at --img M` | Raise Image size (the warning names a value) or annotate from closer cameras. No epoch count fixes this. |
| `N image(s) ... have NO label .txt next to them` | The `images/` → `labels/` rewrite did not resolve. Those images train as pure background. |
| `label files contain class id(s) [5] but nc=1` | Annotations and the CVAT label list are out of sync. |
| `N heads per image is a dense crowd` | Expect counts to saturate; prefer a larger Image size and validate on a held-out dense clip. |

`TRAINX_SKIP_PREFLIGHT=1` skips the scan. It samples at most 10,000 label
files, which is plenty for a stable box-size median and class tally.

After training, the four numbers the human acceptance gate reads are echoed
straight into the stage log:

```
[train.py] acceptance-gate metrics (last row): precision=0.887  recall=0.741  mAP_0.5=0.812  mAP_0.5:0.95=0.503
```

---

## Reading the results for THIS project

**`recall` is the metric that matters.** A missed head is a person who was
never counted, and no downstream smoothing recovers them. Precision errors
are visible and fixable by raising the confidence threshold at the edge;
recall errors are invisible.

`mAP_0.5:0.95` will look low compared to a full-body `person` model and that
is expected, not a defect: a head box is small, so a few pixels of box offset
move the IoU far more than they would on a person-sized box. Judge the model
on `mAP_0.5` and `recall`, and confirm it on the evaluation stage's visual
sample.

If recall is weak:

* raise **Image size** to 640, then 960 — by far the biggest lever on small
  objects, and the preflight's `box size` line tells you how far to go;
* add frames at the distances and angles the **deployment camera** actually
  sees. A model trained on close-up heads will not find heads at the back of
  a room;
* add the hard cases: heads from behind and from above, heads in hats,
  helmets and hoods, heads partly cut by the frame edge, motion-blurred heads
  on a moving subject, heads under harsh backlight;
* add **hard negatives** — headrests, mannequins, posters and photographs of
  people, balls, round bags and lamps. Without them anything head-sized and
  round becomes a head;
* annotate **every** head in a frame, including the small ones in the
  background. A half-annotated crowd teaches the model that distant heads are
  background, which is precisely the recall you are trying to buy.

---

## Repo layout

| Path | Purpose |
| --- | --- |
| `train.py` | **Required.** Training entry point at the repo root. Accepts the orchestrator's exact argv, dispatches on the `--weights` family, runs the dataset preflight, writes `results.csv` + `weights/best.pt`. |
| `export.py` | **Required.** Post-training export. Sniffs the family from the trained checkpoint, writes `best.onnx` next to the weights, plus a `model_package.json` when one is not already there. |
| `detect.py` | Inference entry point. The evaluation stage's `eval_visual_sample.py` looks for it here to render the annotated "what did the model actually find?" sample; without it that stage logs `detect.py not found` and the run has numbers but no picture. |
| `requirements.txt` | Dependencies per architecture family. **Not** pip-installed automatically — the training venv is shared. |
| `dataset.yaml.example` | Reference copy of the per-run generated dataset config. Never ship a live `dataset.yaml`. |
| `deploy/edge_client.py` | Runnable edge client: RTSP/file → ONNX → *"how many heads are in frame?"* → your action hook. |
| `deploy/README.md` | Install + systemd steps for shipping this model to a device. |
| `.gitignore` | Keeps weights/exports/runs out of git — binaries are versioned by DVC in MinIO (house rule). |

---

## The platform contract (unchanged from the skeleton)

**Invocation** — `<project-venv>/bin/python <repo-clone>/train.py` with:
`--img`, `--batch`, `--epochs`, `--data`, `--weights`, `--project`, `--name`,
`--seed`, `--exist-ok`. Never rename or drop one of these flags; the
orchestrator passes every one.

**Required outputs** in `<--project>/<--name>/`:

* `results.csv` — per-epoch metrics in **YOLOv5 column names**. The last row's
  `metrics/precision`, `metrics/recall`, `metrics/mAP_0.5`,
  `metrics/mAP_0.5:0.95` feed the acceptance gate. The ultralytics path
  rewrites its native `(B)`-suffixed columns into these automatically.
* `weights/best.pt` — the trained checkpoint. Exported, quantized,
  regression-tested, published via DVC to MinIO, and offered as the warm-start
  source for later runs.
* `weights/last.pt` — optional, not read by the pipeline.

**Export stage** runs
`python export.py --weights <best.pt> --img <size> --include onnx --data <yaml>`
and fails unless `best.onnx` appears next to the weights. `--include
openvino` / `tflite` may follow and are best-effort. Export at the size the
model was **trained** at — the ONNX input is fixed, and a graph exported at
320 cannot be run at 640 later to win back small-head recall.

**Architecture dispatch** — `yolov5*` re-execs `$YOLOV5_ROOT/train.py`
(default `/home/ivis/projects/Human-Detection/code`, exported by the training
host); `yolov8*`/`yolo11*`/`yolo26*` train natively via `ultralytics`.

**Dataset** — CVAT annotations exported as "YOLO 1.1", promoted into an
approved immutable dataset version under
`/trainx/datasets/<project_id>/master/versions/<dataset_id>/`. The pipeline
writes the split lists and points the generated yaml at them.

**MLflow** — logged **for** the run by `log_run_to_mlflow.py` after `train.py`
exits. This repo needs no MLflow code of its own; it only has to produce
`results.csv` and `weights/best.pt`.

**MinIO / DVC** — weights and datasets are **never** committed to git; git
holds code and `.dvc` pointers only.

---

## Troubleshooting

| Stage log says | Fix |
| --- | --- |
| `needs the 'ultralytics' package, which is not importable` | You picked a `yolov8`/`yolo11`/`yolo26` variant. Ask the MLOps engineer to `pip install ultralytics` on the shared training venv, or switch to `yolov5s`. |
| `yolov5 weights delegate training to the shared YOLOv5 checkout, but .../train.py does not exist` | `YOLOV5_ROOT` is unset or wrong on the host. The training host normally exports it. |
| `class 'head' (id 0) has ZERO boxes in the train split` | Draw and approve the head boxes in CVAT, promote a new dataset version, rerun. |
| `none of the CVAT labels ... looks like a head class` | The project's CVAT label list has no head label. Check the wizard's project, or rename the label. The run continues but every head-specific check is skipped. |
| `the median 'head' box is only 11px tall at --img 320` | Raise Image size to the value the warning names and rerun. Do not spend epochs on this. |
| `cannot determine the model family from --weights` | The variant is neither `yolov5*` nor `yolov8*`/`yolo11*`/`yolo26*`, and a warm-start checkpoint did not identify itself. Pick a supported variant. |
| Good mAP, but people missed in the field | Read the visual sample, then raise Image size, and check that the training frames were shot at the deployment camera's distance and angle. See *Reading the results* above. |
| `best.pt appears to be an Ultralytics YOLOv5 model ... NOT forwards compatible with YOLOv8` in the export stage | The family sniff misread a YOLOv5 checkpoint as an ultralytics one. A YOLOv5 `best.pt` trained from a git clone pickles the remote URL `github.com/ultralytics/yolov5`, so the bare substring `ultralytics` is present in YOLOv5 checkpoints too. Fixed here by testing `models.yolo` **first**; if you see this again, the repo the pipeline cloned is older than this commit. |
| No `[train.py] ---- dataset preflight` block in the stage log | The pipeline cloned a commit **without** this code. Check the wizard's Repository URL and that your work is pushed to *that* repo, then rerun with 'use latest code'. |
| `N/1000 train images (91%) contain NO boxes at all` | An autolabel export where the auto-labeller found objects in very few frames. Review and correct it in CVAT and re-promote — no epoch count or learning rate fixes a dataset with ~100 annotated images. |
| `images appear in BOTH the train and val splits` | The dataset version has no persisted split, so mAP is being measured on training data and means nothing. Re-promote the dataset version (any mode) to get a real leakage-free split. |
| `_ARRAY_API not found` / `compiled using NumPy 1.x cannot be run in NumPy 2.x` | Environment, not repo: the training venv has numpy 2.x while pandas/pyarrow were built against 1.x. Non-fatal (training and MLflow logging still complete) — hand it to the MLOps engineer to pin `numpy<2` or rebuild those wheels. |
