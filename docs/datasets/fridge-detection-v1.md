# Fridge Detection — downloaded candidate

Downloaded on **2026-10-20** for inspection. This is not yet a selected training dataset.

## Source and attribution

- Original creator/workspace: **fridge** on Roboflow Universe.
- Original project: https://universe.roboflow.com/fridge-ydzni/fridge-detection-es0ja
- Original version 1: https://universe.roboflow.com/fridge-ydzni/fridge-detection-es0ja/dataset/1
- Team fork: https://app.roboflow.com/donavans-workspace/fridge-detection-es0ja-ck6ks
- Fork version **1**, `fridge-detection-v1-baseline`:
  https://app.roboflow.com/donavans-workspace/fridge-detection-es0ja-ck6ks/1
- Declared license: **MIT**, confirmed by project page and exported YAML metadata.
  Preserve supplied copyright/license notices when reusing or distributing material.
  The downloaded archive contains no standalone LICENSE file; do not invent an author
  copyright notice or treat the license metadata as verified image ownership/provenance.
- Retain attribution to the original source as well as the fork. Forking does not make
  the team the original dataset creator.

## Local files and verification

The original ZIP remains in Downloads as
`Fridge detection.v1-fridge-detection-v1-baseline.yolov11.zip`.
Extracted files are in `data/fridge-detection-v1/`, excluded from Git.
Original `README.roboflow.txt` and `data.yaml` are preserved unchanged.
The ZIP contained three identical `data.yaml` entries; extraction retains one file.
Archive SHA-256 and machine-readable metadata are in `fridge-detection-v1.json` beside this document.

Export format: YOLOv11. No extra preprocessing or augmentation was added.

| Export split | Image files | Label files |
| --- | ---: | ---: |
| train | 2,994 | 2,994 |
| valid | 103 | 103 |
| test | 51 | 51 |
| Total | 3,148 | 3,148 |

Counts are file counts, not object counts or evidence of annotation correctness.
Neither model training nor prediction was performed. No credit-card information was
entered and no paid upgrade was accepted during version creation/export.

## Before training

- Audit label quality, per-class objects, image readability, duplicates, source provenance,
  and split independence. Unknown capture-session IDs must remain unknown, not fabricated.
- Choose the class scope explicitly. The export has 30 classes; the project's ten-class
  configuration is unchanged. `eggs` also differs from the project name `egg`.
- Missing original target classes: bell pepper, broccoli, yogurt.
- The existing validator does not yet accept this source or class set.
- Exported YAML uses `../train/images`, `../valid/images`, and `../test/images` despite
  being at the dataset root. Prepare a separate corrected configuration before using
  our YAML-relative path resolver; preserve the original export for traceability.
- Reserve independent team refrigerator photos for evaluation. Do not rely on the public
  model score as evidence of this project's accuracy.

Retrieve on Windows by opening fork version 1, choosing Download Dataset, selecting
YOLOv11 and Download ZIP, then extracting under the ignored `data/` directory.
A free Roboflow account may be required. No hosted training/API is needed for local training.
