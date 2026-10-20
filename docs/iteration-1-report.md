# Iteration 1 report — work in progress

## Current status

Code scaffold written on the Mac. No real data collected/imported by this workflow,
no ML dependencies installed for it, and no model downloaded, trained, or evaluated.
The workflow uses Python scripts directly; no notebook or Jupyter dependency is required.
Technical owner: Donavan. Teammates help collect approximately 120 refrigerator photos
in total; annotation assistance and second-person review must be recorded as completed.

## Goal and method

Fine-tune pretrained YOLO11n for apple, banana, bell pepper, broccoli, carrot, egg,
milk, onion, tomato, and yogurt. Train later on Windows with an RTX 3060 Ti.
Retain exported label order. Initial settings: 640 pixels, 50 epochs, patience 10,
seed 42, automatic batch size, workers=0, standard training augmentation.
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

No dataset statistics or review completion are currently established.

## Reproducibility record — pending Windows execution

Record Python, package versions, NVIDIA driver, PyTorch CUDA build/install command,
GPU, dataset version, training command, actual batch size, run folders, and checkpoint
link. Preserve environment.json, request.json, requirements-lock.txt, and training args.
Back up selected weights and full run outputs to team-accessible Drive. Repository
contains code/configuration and approved aggregate artifacts, not full datasets/weights.

## Evaluation — not run

After selection using validation data, evaluate the best checkpoint on held-out test
images. Report precision, recall, mAP50, mAP50–95, per-class counts/results, confusion
matrix, and at least ten approved prediction examples including failures. Assess
clear/multiple/repeated items, obstruction, low light, packaged foods, unsupported
foods, empty scenes, similar packaging, and an unseen refrigerator when available.

Targets: mAP50 ≥ 0.50 and overall precision/recall ≥ 0.60. These are goals, not results.
Report weak results honestly. Do not tune using test results.

## Verification and limitations

Seven synthetic validator tests pass. They use temporary byte fixtures as images and
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
