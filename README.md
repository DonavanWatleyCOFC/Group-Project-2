
## Downloaded public candidate

Fridge Detection fork version 1 is available locally under ignored
`data/fridge-detection-v1/`. See [dataset record](docs/datasets/fridge-detection-v1.md)
for source, license, export counts, and retrieval instructions. It has **30 classes**
and is **not yet compatible with the original ten-class validator**. Do not run the
example ingredient training commands against this raw export; class/source integration
and path correction remain pending. No training has occurred.

## Combined AICook experiment

The MIT-licensed [AICook dataset](https://universe.roboflow.com/karel-cornelis-q2qqg/aicook-lcv4d)
has the same 30 ingredient names as this project, but a different numeric class order.
Download its original, unaugmented export into ignored `data/aicook-v1/`, preserving its
`data.yaml`, images, and labels. Then build a new ignored combined dataset:

```powershell
python scripts/build_combined_dataset.py `
  --baseline-data data/fridge-detection-v1/data.yaml `
  --incoming-data data/aicook-v1/data.yaml `
  --output-root data/combined-aicook-v1
```

The builder requires an exact 30-name match, remaps every incoming label by name into
the baseline class-ID order, and refuses to overwrite existing output. It includes
baseline train/validation plus AICook train/validation. The existing baseline test split
is copied unchanged only for historical comparison; AICook's test split and external
household fridge photos never enter training. Train the resulting data with moderate color
augmentation for different fridge lighting:

```powershell
python scripts/workflow.py train --data data/combined-aicook-v1/data.yaml `
  --manifest data/combined-aicook-v1/manifest.csv --epochs 50 --batch 8 `
  --output C:\path\to\new-experiment
```

The training defaults use hue 0.015, saturation 0.35, and brightness/value 0.25;
they preserve useful food color while teaching modest lighting variation. Keep labeled
frames from the CollectData `Inside the Fridge` set in a separate external evaluation
directory and score them only after choosing the model.

## Interactive Rerun detection viewer

Activate the project environment and install the optional viewer:

```bash
source .venv/bin/activate
python -m pip install -r requirements-viewer.txt
```

Open an existing prediction run without rerunning the model:

```bash
python scripts/rerun_viewer.py --predictions runs/external-review-01/predictions.json
```

Or run new CPU predictions and open them in Rerun (choose a new output directory):

```bash
python scripts/workflow.py predict --weights baseline-01/best.pt --source data/new-photos --output runs/rerun-review-01 --device cpu --rerun
```

Select the `photo` timeline to move between photos. Click a box edge to select its
entity; the expanded Selection panel shows `class_name`, `class_id`, and `confidence`.
Boxes also display a class and confidence label. Confidence is a model score, not a
verified probability of correctness. Each detection has its own entity so overlapping
objects can also be selected in the entity tree. Empty photos clear prior boxes.

The integrated command saves `review.rrd` beside predictions; reopen with
`rerun runs/rerun-review-01/review.rrd`. Standalone viewer supports `--save PATH.rrd`
and `--no-spawn` for recording-only use. Run commands from the project root; original
image paths in prediction JSON must still exist. Recordings contain the photos and
are ignored by Git. Do not share personal-image recordings without review.

HEIC support: the optional requirements include pillow-heif. The viewer decodes photos
with EXIF orientation into PNG bytes before logging, leaving original files unchanged.
Old recordings containing HEIC bytes must be regenerated from predictions JSON.

**New to Rerun?** Follow the [team viewer guide](docs/rerun-guide.md) for Windows/Mac
installation, the exact recording-open command, and how to select photos and boxes.
