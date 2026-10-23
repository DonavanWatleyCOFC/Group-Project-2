# Iteration 1 report — work in progress

## Current status

Windows setup began on 2026-10-22. The code now targets the supplied 30-class Roboflow
version rather than the obsolete ten-class scaffold. No real export has been imported
yet, and no model has been downloaded, trained, or evaluated.
The workflow uses Python scripts directly; no notebook or Jupyter dependency is required.
Technical owner: Donavan. Teammates help collect approximately 120 refrigerator photos
in total; annotation assistance and second-person review must be recorded as completed.

## Goal and method

Fine-tune pretrained YOLO11n for all 30 classes from the exact Roboflow export, retaining
the exported label order and spelling (including `eggs`). Bell pepper, broccoli, and
yogurt are not supported by this baseline. Train locally on Windows with an RTX 3060 Ti.
Initial settings: 640 pixels, 50 epochs, patience 10, seed 42, batch size 8, workers=0,
and standard training augmentation.
These are starting choices, not experimentally established best settings.

See [decision log](decision-log.md) for evidence, tradeoffs, AI assistance, and limitations.

## Dataset record — pending

Record before training:

- Roboflow project/version/export date and retrieval instructions.
- Source URLs, license checks, and approval for public upload.
- Image and object counts by source, class, and split.
- Annotation rules, ambiguous examples, exclusions, and second-person review status.
- Session/refrigerator grouping and frozen split manifest.
- Approximate 70/15/15 custom-photo split; public images training-only.
- Auto-orientation, 640×640 resize, no offline augmentation, and any actual deviations.
- Entirely held-out refrigerator if available; otherwise disclose its absence.

No dataset statistics, actual 30-name list, attribution files, or review completion are
currently established because the raw version-1 export is absent. The expected image
counts are 2,994 train / 103 validation / 51 test; these are claims to verify, not results.

## Reproducibility record — Windows setup verified

Observed: Python 3.11.9; NVIDIA GeForce RTX 3060 Ti; NVIDIA driver 581.15 (reports CUDA
13.0); torch 2.11.0+cu128; torchvision 0.26.0+cu128; Ultralytics 8.4.184; PyYAML 6.0.3;
and PyTorch CUDA build 12.8. `torch.cuda.is_available()` is true and a small CUDA tensor
calculation passed. Install command: `python -m pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu128`, then `python -m pip install -r requirements.txt`.

Still record the dataset version, training command, actual batch size, run folders, and
checkpoint link per run. Preserve environment.json, request.json, requirements-lock.txt,
and training args.
Back up selected weights and full run outputs to team-accessible Drive. Repository
contains code/configuration and approved aggregate artifacts, not full datasets/weights.

## Smoke validation — completed, not a baseline result

The exact version-1 export passed automated integrity checks: 2,994 training, 103
validation, and 51 test images; paired labels; readable images; valid class IDs/boxes;
no byte-identical image across splits; and complete manifest coverage. The export has no
evidenced capture-session/refrigerator metadata, so its split independence and visual
near-duplicate status remain pending human review.

A one-epoch YOLO11n smoke run on the RTX 3060 Ti used batch 8, workers 0, training plus
validation only. It reported validation precision 0.797, recall 0.800, mAP50 0.852, and
mAP50-95 0.534. This is an operational smoke result, not a quality claim or full baseline;
the held-out test split was not evaluated. External weights/artifacts are in
`C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-smoke-20261022`.

## Held-out test evaluation — completed once

The selected `best.pt` checkpoint from the 50-epoch baseline was evaluated once on the
51-image held-out Roboflow test split, after model selection. Overall results were
precision **0.928**, recall **0.919**, mAP50 **0.943**, and mAP50-95 **0.663**. The
checkpoint and complete evaluation artifacts, including per-class JSON, curves, confusion
matrices, and label/prediction batches, are outside Git at
`C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-test-20261022`.

These metrics meet the stated project targets on the supplied split, but they are not
evidence of generalization to independent refrigerators: the export offers no proven
refrigerator/session grouping and visual near-duplicate review remains pending. The test
set has no `goat_cheese` or `ham` instances, so it cannot measure those classes. The
weakest measured mAP50-95 results were beef (0.327), lime (0.394), and chicken_breast
(0.489); they should guide future data review and collection rather than be hidden.

## Human-in-the-loop external verification — pending on Mac

Donavan will retain personal/friend refrigerator photos on the Mac, outside all current
training and tuning. Unlabelled photos can support qualitative box/class review. To
produce comparable external metrics, they must be annotated in the same 30-class ID order
and checked for overlap before one final external evaluation. Unsupported foods—bell
pepper, broccoli, and yogurt—must be documented as outside this baseline rather than
scored as detector errors.

After selection using validation data, evaluate the best checkpoint on held-out test
images. Report precision, recall, mAP50, mAP50–95, per-class counts/results, confusion
matrix, and at least ten approved prediction examples including failures. Assess
clear/multiple/repeated items, obstruction, low light, packaged foods, unsupported
foods, empty scenes, similar packaging, and an unseen refrigerator when available.

Targets: mAP50 ≥ 0.50 and overall precision/recall ≥ 0.60. These are goals, not results.
Report weak results honestly. Do not tune using test results.

## Verification and limitations

Eight synthetic validator tests pass. They use temporary byte fixtures as images and
an isolated JSON-backed YAML parser; real image decoding and YAML integration are not
verified. Python source syntax checks do not demonstrate model correctness. Windows
CUDA setup, dependency compatibility, model calls, generated plots/metrics, inference,
and teammate reproduction are pending. Exact hashes do not detect visual near duplicates;
annotation quality and data independence require human review. No model quality or
training-time claims are supported yet.

## AI assistance and team review

AI drafted Python scripts, tests, and documentation and performed limited local checks.
No images, licensing, labels, or model outputs have been reviewed by AI in this task.
Record material AI errors and corrections as they arise. Team understanding and approval
of decisions have not been verified; each presenter must be able to explain the method,
evidence, alternatives, and remaining uncertainty in their own words.

## Candidate public dataset acquired — 2026-10-20

Fridge Detection team-fork version 1 has been downloaded as YOLOv11 and extracted
locally. See [source, license, and retrieval record](datasets/fridge-detection-v1.md).
Observed file counts: 2,994 train, 103 validation, 51 test images, each with a label file.
The export declares MIT and includes 30 classes. No object-count or annotation-quality
claim is made. This updates the earlier no-import status: a candidate is now present,
but it has not been selected, mapped into the ten-class project, or used for a model run.
Data provenance, license-notice completeness, duplicates, split integrity, and real-scene
usefulness require further audit. No training/evaluation results exist.

## Interactive prediction review — 2026-10-23

An optional Rerun viewer now reads saved predictions and displays selectable detection
boxes on the original photos, with class names, IDs, and confidence scores. A photo
timeline separates images and clears stale detections. This supports qualitative human
review; it does not calculate additional evaluation metrics or explain internal model
reasoning. Ten synthetic tests and a real SDK synthetic recording check passed. Local
SDK version: 0.33.1. GUI selection and human prediction review still need confirmation.

### HEIC viewer correction

The initial viewer showed only the JPEG while HEIC frames were black. It now decodes
photos with EXIF orientation and logs PNG bytes. All six photos decoded successfully;
dimensions match the inference loader. A corrected local recording is
`runs/external-review-01/review-heic-fixed.rrd`. No predictions or model weights changed;
visual confirmation remains pending.

### Team review instructions

See [Rerun viewer guide](rerun-guide.md) for Windows/Mac setup, opening the corrected
recording, selecting detections, and recording qualitative errors. Reviewers need the
separately shared .rrd and matching viewer, not weights or original photos. Windows
viewer installation and teammate reproduction remain pending.
