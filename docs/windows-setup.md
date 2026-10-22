# Windows RTX 3060 Ti setup — Iteration 1

This workflow uses Python scripts, not Jupyter notebooks. The local environment, raw dataset, pretrained weights, and experiment outputs are intentionally outside Git.

## Local locations

- Environment: `C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-venv`
- Raw Roboflow export: `data\fridge-detection-v1\` (Git-ignored)
- Derived config/manifest/reports: `C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-artifacts`
- Runs: external folders under `C:\Users\donav\Documents\GitHub\LearnFFNN\`

## Dataset retrieval and preparation

In Roboflow, sign in and export **donavans-workspace / fridge-detection-es0ja-ck6ks, version 1** as **YOLOv11**, without changing preprocessing or augmentation. Put the entire extracted export at `data\fridge-detection-v1\`; retain its README, attribution, and all supplied files. Do not put an API key in this repository.

```powershell
$py = 'C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-venv\Scripts\python.exe'
$artifacts = 'C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-artifacts'
& $py scripts\prepare_dataset.py --raw-root data\fridge-detection-v1 --output-data "$artifacts\fridge-detection-v1.local.yaml" --output-manifest "$artifacts\fridge-detection-v1.public-manifest.csv"
& $py scripts\validate_dataset.py --data "$artifacts\fridge-detection-v1.local.yaml" --manifest "$artifacts\fridge-detection-v1.public-manifest.csv" --output "$artifacts\dataset-summary.json"
```

The preparer requires exactly 30 classes and preserves the export's names and class-ID order verbatim. `public_unknown` means this public export has no evidenced refrigerator or capture-session metadata; it is not evidence that the supplied splits are independent. Future team photos must use `source=custom` and non-empty refrigerator/session IDs; the validator rejects cross-split session leakage.

## Checks and training

```powershell
& $py scripts\workflow.py check-gpu
& $py -m unittest discover -s tests -v
```

After validation succeeds, the permitted smoke test uses training and validation only; it neither evaluates held-out test data nor begins full training:

```powershell
& $py scripts\workflow.py train --data "$artifacts\fridge-detection-v1.local.yaml" --manifest "$artifacts\fridge-detection-v1.public-manifest.csv" --epochs 1 --batch 8 --output C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-smoke
```

Later, after human review, run the full baseline (not during setup):

```powershell
& $py scripts\workflow.py train --data "$artifacts\fridge-detection-v1.local.yaml" --manifest "$artifacts\fridge-detection-v1.public-manifest.csv" --epochs 50 --batch 8 --output C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-baseline
```

YOLO downloads pretrained `yolo11n.pt` only when `train` is run. No GPU clocks, power, or thermal settings are changed.
