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

## Evaluation — held-out test not run

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
