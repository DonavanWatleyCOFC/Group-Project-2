# YOLO plus OCR in Group-Project-2

`scripts/workflow.py predict --ocr` runs the selected YOLO checkpoint, then reads
package text locally with PaddleOCR. It uses the checkpoint's actual class order
and supports the 44-, 50-, 63- and 64-class models without manually changing IDs.
It does not retrain YOLO. Predictions remain suggestions requiring user confirmation.

## Optional environment (Windows Command Prompt)

From Group-Project-2, create an isolated OCR environment once:

```bat
"..\fridge-to-recipe-i1-venv\Scripts\python.exe" -m venv .ocr-venv
.ocr-venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
```

The main command still runs with the existing YOLO environment. It automatically
finds `.ocr-venv`, or accepts `--ocr-python` for an existing OCR interpreter. The
first OCR run downloads pretrained English models; photos are processed locally.
Model caches are stored by PaddleOCR under the user's `.paddlex` directory.
Install `requirements-viewer.txt` in the YOLO environment to use `--rerun`.

Example, using the project's own `.ocr-venv` automatically:

```bat
"..\fridge-to-recipe-i1-venv\Scripts\python.exe" -B scripts\workflow.py predict --weights weights\best.pt --source data\new_photos\fridge_17.jpg --output "runs\ocr-review-%RANDOM%" --device cpu --confidence 0.25 --ocr --rerun
```

Without `--ocr`, the previous YOLO-only behavior is retained. Each output directory
must be new. CPU inference avoids competing with an active GPU training job.

The local `weights/best.pt` is a copy of the finished 64-class dairy checkpoint
(`fridge-to-recipe-dairy-20261031/artifacts/weights/best.pt` in the parent directory).
Its original 63 class IDs are preserved and yogurt is ID 63. Checkpoints remain
excluded from Git. OCR is enabled by `--ocr`; it is not embedded in the checkpoint.

## Output contract

- `predictions.json`: original YOLO detections, unchanged.
- `class-names.json`: class-ID order extracted from the checkpoint.
- `ocr/ocr.json`: raw OCR text, locations and recognition scores.
- `ocr/predictions.json`: YOLO plus OCR candidates, changes and grouped-line evidence.
- `ocr/comparison.json`: per-photo class lists and changes.
- `ocr/*-comparison.png`: side-by-side images.
- `review.rrd`: when `--rerun` is selected, a saved comparison opened in Rerun.

The left pane is YOLO alone and the right adds OCR. Orange rectangles localize text,
not whole products. The photo timeline selects images. OCR recognition scores are
not ingredient probabilities. For OCR-supported changes, `confidence_kind` is
`ocr_recognition`; `yolo_class_name` and `yolo_confidence` preserve the original
visual result. All OCR additions/changes set `review_required=true`.

Known aliases are matched at >=85% OCR recognition confidence. Two or three nearby,
aligned lines can be joined into exact aliases such as KIDNEY + BEANS. Duplicate
crop readings, ingredient-list headers, weak lines, distant lines and ambiguous
ownership are guarded against. These spatial rules are heuristics and cannot
guarantee that neighboring text belongs to the same product. Typos such as SEANS,
obscured labels, unreadable text and unsupported ingredient aliases can still fail.
OCR does not eliminate visual false positives or estimate food quantity/allergens.

## Verification

```bat
"..\fridge-to-recipe-i1-venv\Scripts\python.exe" -B -m unittest discover -s tests -p test_ocr_pipeline.py
```

Compare both paths on the same held-out, fully labeled images before claiming an
accuracy improvement. Existing household examples are development cases, not an
independent final test. Keep the original labels and prediction provenance for review.
