# Decision log

## 2026-10-19 — Local Windows GPU training and code-first development

**Status:** Adopted from Donavan's instructions. No training performed.

**Decision:** Write the pipeline on the Mac now; later train YOLO11n on Donavan's
Windows RTX 3060 Ti. Use scripts shared by the command line and local notebook.
Require CUDA for training; CPU inference remains available for teammates.

**Reason/evidence:** The team already has access to the GPU and Git repository.
Explicit commands prevent accidental training during development. The scripts use
Ultralytics' documented train/val/predict interfaces and Windows-safe entry points.

**Alternative/tradeoff:** Colab offers hosted compute but introduces temporary sessions
and separate runtime storage. Local training requires driver/dependency setup and backups.
Notebook-only code is simpler initially but duplicates logic and is harder to check.

**Verification:** Seven standard-library synthetic tests passed on the Mac for counts,
background labels, exact-duplicate leakage, session leakage, public-data restrictions,
invalid boxes, missing labels, and manifest coverage (counts/backgrounds share one test).
These tests isolate parsing using a JSON-backed YAML substitute; they do not verify
PyYAML, image decoding, Ultralytics, PyTorch, CUDA, or notebook execution.
All Python scripts and notebook code cells parsed successfully; notebook JSON is valid. Both CLI help commands completed without ML imports. `git diff --check` passed, and `AGENTS.md` remains locally ignored and untracked.

**Limitations/next action:** No packages installed, dataset or weights downloaded, GPU
checked, or model trained/evaluated. Dependency ranges are provisional. On Windows,
select CUDA PyTorch using its official selector, complete the smoke test, record driver
and install command, and retain the successful environment lock. Cross-device exact
numerical reproducibility is not guaranteed by the seed alone.

**AI assistance:** AI drafted scripts, notebook, tests, and documentation and ran only
source/format and synthetic logic checks. Student review and Windows runtime verification
remain pending; no claim of human comprehension or review is made.

## 2026-10-19 — Data integrity before experiments

**Status:** Existing collection/split goals retained; implementation choices documented here.

**Decision:** Preserve exported class order, require all ten class names, explicit empty
labels for reviewed backgrounds, a complete per-image session/source manifest, and
train/val/test directories. Reject cross-split identical files, session leakage, malformed
labels, and public Grocery Store images outside training. Manual Roboflow export is used
initially so no API key is needed in code.

**Reason/evidence:** Independent images and realistic held-out refrigerator scenes are
needed for useful evaluation. Missing labels can otherwise silently look like background.
The manifest makes source and group decisions inspectable. Exact hashes are evidence
of identical bytes, not evidence that different bytes represent independent scenes.

**Alternatives/tradeoffs:** Random image splitting is easier but risks scene leakage.
Automatically assigning missing annotations as background could hide incomplete labeling.
Automatic dataset downloads are convenient but need credential handling; manual export
needs version/link bookkeeping. Strict directory-only input avoids ambiguous path handling
but does not support image-list YAML entries or arbitrary sample datasets yet.

**Verification:** Synthetic tests cover the implemented safeguards; no real dataset has
been checked. Class counts and annotation-review status are unknown.

**Limitations/next action:** Students must verify licensing, public-upload privacy,
annotation correctness, readable images, near duplicates, export filename mapping,
class coverage, second-person review, and refrigerator-level separation. The script
checks session grouping but does not require an entirely held-out refrigerator. Dataset
version and review completion must be recorded before baseline training.

**AI assistance:** AI implemented proposed checks; it has not inspected photographs or
verified any dataset license. Data selection and scientific justification remain team responsibilities.

## 2026-10-19 — Baseline and reporting defaults

**Status:** Proposed baseline encoded; no measured outcomes.

**Decision:** Start with pretrained YOLO11n at image size 640, 50 epochs, patience 10,
seed 42, automatic batch sizing, and workers=0. Save environment/request metadata per run.
Do not automatically retry failed runs or overwrite old output directories. Evaluation
is a separate explicit test command; prediction defaults to CPU and confidence 0.25.

**Reason/evidence:** These settings follow the agreed baseline; workers=0 reduces Windows
process-launch complexity. Separate evaluation makes test exposure intentional. CPU
prediction permits teammates to use the selected checkpoint without training hardware.

**Alternatives/tradeoffs:** A larger model or additional tuning may improve results but
requires evidence and more compute. Manual fixed batch sizes may be needed on the GPU.
The display threshold is provisional, not calibrated. Use validation for any threshold tuning.

**Verification:** Only syntax/source and synthetic validation checks so far; model APIs,
GPU memory requirements, metrics export, and timing are not runtime-verified.

**Limitations/next action:** Record each run's actual settings, failures, package versions,
and result paths. Per-class metric output follows classes reported by Ultralytics;
review test counts and explicitly discuss absent classes. No speed or accuracy claim
can be made before experiments. Backups remain a manual step.

**AI assistance:** AI supplied implementation and documented official API references.
Team acceptance of analytical explanations and model results remains pending.

## References

- [Ultralytics Python API usage](https://docs.ultralytics.com/usage/python/)
- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/)
- [Official PyTorch installation selector](https://pytorch.org/get-started/locally/)

## 2026-10-22 — Version-1 Roboflow export is a 30-class baseline

**Status:** Adopted from Donavan's instructions; code updated, real-export review pending.

**Decision:** Use all 30 names in the exact class-ID order supplied by Roboflow version 1. The pipeline no longer encodes the historical ten-class list. The raw export's `data.yaml` is the authority; `prepare_dataset.py` writes a separate ignored local configuration and copies the list verbatim, without renaming, reordering, relabeling, or subsetting. The baseline does not support bell pepper, broccoli, or yogurt. `eggs` is retained when it is present in the export rather than normalized to the historical singular spelling.

**Reason/evidence:** The requested export is `fridge-detection-v1-baseline`, YOLOv11, declared MIT, expected to contain 2,994 training, 103 validation, and 51 test images. At this entry, the raw export was not present locally, so its actual class names/counts, attribution files, and split contents remain unobserved.

**Data policy:** Public-export items use `public_unknown`: no refrigerator/session IDs are invented and no independence claim is made. Strict session grouping remains mandatory for future `custom` images. The validator checks readable image files, image/label pairing, class IDs, boxes, class counts, byte-identical cross-split images, and manifest coverage; near duplicates and semantic split leakage still require human review.

**AI assistance:** AI updated the scripts, tests, and documentation under explicit instructions. Human review of Roboflow terms, attribution, license applicability, annotations, and the final 30-name sequence is pending.

## 2026-10-22 — Windows environment observation

**Observed:** `nvidia-smi` reported NVIDIA GeForce RTX 3060 Ti, driver 581.15, and CUDA compatibility 13.0. Python 3.11.9 was installed per-user and an external virtual environment was created at `C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-venv`. No GPU clock, power, fan, or thermal setting was changed.

**PyTorch plan:** The official PyTorch Windows selector supports Python 3.10–3.14 and directs CUDA-capable Windows users to pip CUDA builds. The selected command is `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (observed candidate versions: torch 2.11.0+cu128 and torchvision 0.26.0+cu128). The 2.75 GB wheel transfer was started as a background process after foreground transfers were interrupted by the execution time limit. Package installation, CUDA tensor verification, and the project dependency install are not yet claimed as complete.

**Checks:** Eight synthetic validator tests passed under Python 3.11. They validate manifest coverage, missing labels, invalid boxes, duplicate leakage, session leakage, public split restriction, and public data with honestly unknown session metadata. Actual image decoding and real-export checks remain blocked by the missing raw export.

## 2026-10-22 — Completed CUDA environment verification

**Observed results:** The external environment contains Python 3.11.9, torch
2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.184, and PyYAML 6.0.3.
`torch.cuda.is_available()` returned `true`, device 0 was `NVIDIA GeForce RTX 3060 Ti`,
and `(arange(1, 7, device='cuda') ** 2).sum()` returned `91.0`. The project's
`scripts/workflow.py check-gpu` command also passed. NVIDIA driver 581.15 reports CUDA
compatibility 13.0; the installed PyTorch wheel is built for CUDA 12.8.

**Actual installation command:** `python -m pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu128`, followed by `python -m pip install -r
requirements.txt`. This uses PyTorch's official Windows pip CUDA channel. No full training,
test evaluation, GPU overclock, power-limit, or thermal-limit change was performed.

## 2026-10-22 — Exact version-1 export audit and smoke run

**Observed dataset:** The supplied ZIP embedded the requested workspace/project/version,
declared MIT, and exported these 30 ordered names: apple, banana, beef, blueberries,
bread, butter, carrot, cheese, chicken, chicken_breast, chocolate, corn, eggs, flour,
goat_cheese, green_beans, ground_beef, ham, heavy_cream, lime, milk, mushrooms, onion,
potato, shrimp, spinach, strawberries, sugar, sweet_potato, tomato. The raw export and
its `README.roboflow.txt` were retained under ignored `data/fridge-detection-v1`. The ZIP
contains three identical `data.yaml` entries; extraction therefore skipped duplicate paths
without overwriting the first identical metadata file.

**Audit result:** Passed readable-image, image/label pairing, class-ID, box, class-count,
exact cross-split hash, and manifest-coverage checks. Observed image counts matched the
expected 2,994 train / 103 validation / 51 test. `public_unknown` is used because source
capture sessions/refrigerators are not evidenced by this export; this result does not
establish independent splits or rule out visual near duplicates.

**Smoke result:** One epoch of pretrained YOLO11n training completed on CUDA with batch 8
and workers 0, using only 2,994 train and 103 validation images. Validation after the
epoch reported overall precision 0.797, recall 0.800, mAP50 0.852, and mAP50-95 0.534.
These are smoke-run validation observations, not a selected baseline or held-out-test
results. Weights and run artifacts are external at `fridge-to-recipe-i1-smoke-20261022`;
held-out test evaluation was not run.

## 2026-10-22 — Candidate Refrigerator Food ZIP rejected as external data

**Observed:** `Refrigerator Food.v1i.yolov11.zip` identifies itself as
`donavans-workspace/refrigerator-food-clmy1/1`, CC BY 4.0, with 50 images and the same
30-class YAML order. Every one of its 50 image SHA-256 hashes exactly matches an image in
the 3,148-image baseline export. Its labels are syntactically valid and use all 30 IDs,
despite its README's narrower annotation description.

**Decision:** Do not extract, train on, validate with, or report this ZIP as external
evidence. It would duplicate existing data and contaminate any claimed independent
evaluation. A distinct dataset or held-out team refrigerator photos remain necessary for
an external generalization check.

## 2026-10-22 — Iteration 1 full baseline and held-out evaluation

**Observed baseline:** Pretrained YOLO11n trained for 50 epochs on the validated 2,994
training images with batch 8, image size 640, seed 42, patience 10, and workers 0 on the
local RTX 3060 Ti. The selected `best.pt` is stored outside Git at
`C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-i1-baseline-20261022\artifacts\weights\best.pt`.

**Observed held-out result:** One post-selection test evaluation over the 51-image supplied
test split reported precision 0.9281, recall 0.9192, mAP50 0.9434, and mAP50-95 0.6625.
Per-class metrics, curves, confusion matrices, and predicted/labelled batch images are
stored outside Git at `fridge-to-recipe-i1-test-20261022`. No further tuning or training
was performed after seeing this test result.

**Limitations:** `goat_cheese` and `ham` have zero test instances, so the held-out result
does not evaluate them. The public split has no evidenced refrigerator/session grouping;
these scores apply to the supplied split and must not be presented as proof of performance
on independent refrigerators. Beef, lime, and chicken_breast have the lowest measured
mAP50-95 values (0.327, 0.394, and 0.489 respectively). Mac-held refrigerator photos
remain the planned human-in-the-loop external check and were not accessed or uploaded.


## 2026-10-19 — Python scripts replace the notebook interface

**Status:** Adopted at Donavan's explicit request; supersedes the notebook interface
in the initial code-scaffold entry above. That entry is retained as a historical record.

**Decision:** Remove the optional notebook and JupyterLab/ipykernel dependencies. Use
`scripts/workflow.py` and `scripts/validate_dataset.py` directly from VS Code or
PowerShell. Update the README, system-design tools section, iteration report, and local
agent instructions to reflect this preference.

**Reason/evidence:** Donavan prefers Python files. The notebook only called the existing
scripts, so removing it retains the command interfaces and avoids an extra environment
and duplicate usage instructions.

**Alternative/tradeoff:** Retaining the notebook would offer interactive cells and saved
outputs, but would require maintaining another interface. Results remain available as
run artifacts and reports rather than notebook outputs.

**Verification:** Seven existing synthetic validator tests passed after the change.
Python source parsing, both CLI help commands, and Git whitespace checks passed.
The notebook is absent, Jupyter dependencies are removed, and AGENTS.md remains
locally excluded and untracked. No model, installation, download, or GPU work was run.

**Limitations/next action:** Windows dependency setup and end-to-end model execution
remain unverified. This interface change does not establish model correctness or
accuracy. Complete the documented Windows smoke test when training hardware is ready.

**AI assistance:** AI removed the wrapper and updated documentation at the user's
request; no claim of student code review or model validation is made.

## 2026-10-20 — Export and preserve Fridge Detection candidate

**Status:** User authorized download, local extraction, and provenance documentation.
Class selection and approval for training remain pending.

**Decision:** Export team fork version 1 as YOLOv11 and preserve its 30 classes, splits,
annotations, and metadata unchanged under ignored `data/fridge-detection-v1/`.
Record source/fork URLs, declared MIT license, ZIP checksum, and file counts in
`docs/datasets/fridge-detection-v1.json` and its companion Markdown record.

**Reason/evidence:** The public dataset already contains detection annotations and is
worth auditing as a candidate to reduce manual labeling. The export reports MIT and
provides 2,994 train, 103 validation, and 51 test image/label pairs by file counts.
Preserving raw data supports reproducible later transformations.

**Alternative/tradeoff:** Collecting/annotating all training data ourselves gives more
control but requires more work. Immediately filtering to seven classes would simplify
the trial but prematurely settle a class decision that the user has not made.

**Verification:** Browser confirmed version 1 and YOLOv11 ZIP export. Archive paths were
checked before extraction and duplicate YAML entries were verified identical. Extracted
split file counts match 3,148 images/labels. Git ignore checks cover the data and local
agent file. No standalone license file exists in the ZIP; YAML license metadata and
supplied README were preserved. Documentation/Git whitespace checks passed.

**Limitations/next action:** No image/annotation quality, object counts, visual duplicate,
or leakage audit yet. Underlying image rights/provenance are not established by the
uploader's license declaration alone. Existing validator rejects these 30 classes/source;
export YAML paths need a separate corrected configuration. No original scope/class
mapping, training script, or dataset split was changed. Windows setup/testing remains pending.

**AI assistance:** AI operated the authorized browser export, extracted files, counted
entries, and wrote provenance records. These operations do not establish dataset quality
or student review. No training, card entry, paid upgrade, commit, or push occurred.

## 2026-10-23 — Selectable Rerun detection review

**Decision:** Add an optional local Rerun viewer over saved prediction JSON, plus a
`predict --rerun` switch. Show the original photo with one selectable box entity per
detection; attach class name, class ID, and numeric confidence as selection metadata.
Use a photo timeline and clear previous detections on every frame, including empty ones.

**Reason/evidence:** Donavan requested clicking boxes to inspect classifications.
Reusing saved predictions avoids retraining or repeated inference. Separate optional
requirements keep the visualization package out of the training dependency set.
Alternative: static annotated images are simpler but cannot expose selectable metadata.

**Verification:** Ten synthetic tests passed, including box geometry/metadata and stale
box clearing. Rerun SDK 0.33.1 installed in the local Mac virtual environment. A real SDK
recording-only check using a generated image and an empty second frame passed under
strict logging. No training or model modification occurred. API references:
https://ref.rerun.io/docs/python/0.33.1/rerun/ and
https://rerun.io/docs/howto/logging-and-ingestion/custom-data .

**Limitations:** Automated logging tests do not establish GUI click behavior or model
accuracy. Confidence is an uncalibrated model score, not a guarantee of correctness.
Original image files must remain available for JSON replay. Recordings embed photos
and are locally ignored. Unsupported image encodings may require conversion to PNG/JPEG.

**AI assistance:** AI wrote implementation/tests/docs and checked SDK serialization.
Human inspection of predictions and GUI selection behavior remains pending. No push.

**Local replay observation:** Logging the six existing photo records to an ignored
`runs/external-review-01/review.rrd` completed. GUI launch from the restricted execution
environment failed to connect; the initial PATH lookup also found a mismatched global
viewer (0.37.1). Updated launch to explicitly use the project's matching SDK viewer.
Interactive rendering/click verification is still pending user launch in a normal terminal;
unsupported HEIC rendering may require JPEG/PNG copies. The saved recording is not proof
that every photo was visually rendered correctly.

## 2026-10-23 — Fix black HEIC frames in Rerun

**Decision/reason:** User reported only the JPEG rendered. The viewer previously sent
original HEIC bytes, which the viewer did not render. Decode every source image with
Pillow (pillow-heif for HEIC), apply EXIF orientation, and log PNG bytes. Source files,
model weights, predictions, and box coordinates remain unchanged.

**Alternative/tradeoff:** Manual JPEG copies would require filename/path management and
could change coordinate orientation. In-memory PNG conversion keeps originals untouched
but increases recording size and decoding time. Added optional pillow-heif dependency.

**Verification:** All six actual source photos decoded to RGB PNG; dimensions matched
the installed Ultralytics inference loader for every image. Generated a fresh recording
at runs/external-review-01/review-heic-fixed.rrd using the existing predictions, without
model inference or training. All ten automated tests passed; GUI rendering is pending
user confirmation. Installed pillow-heif 1.8.1 in the Mac virtual environment.

**AI assistance/limitations:** AI diagnosed the encoding mismatch and corrected the
logging implementation. Model correctness and human annotation review remain separate;
this is a visualization repair. No push or source-photo changes.

## 2026-10-23 — Beginner team Rerun guide

**Decision/reason:** Add docs/rerun-guide.md with separate Windows/PowerShell and Mac
instructions, version-matched 0.33.1 installation, explicit executable paths, recording
location/open commands, selection/timeline instructions, and troubleshooting. Link it
from the README. Ignore a separate .rerun-venv so reviewers do not alter training setups.

**Alternative/tradeoff:** Installing the full ML environment is unnecessary for opening
an embedded .rrd. A viewer-only environment is smaller and avoids PATH version conflicts;
teammates still need Python and the separately shared recording.

**Verification:** Checked the corrected recording exists, the installed viewer reports
0.33.1, and official Rerun docs confirm packaged viewer and file-open CLI. Git ignore
and whitespace checks passed. Windows commands were reviewed, not executed on this Mac.

**AI assistance/limitations:** AI drafted the guide; teammate installation, graphics
compatibility, and human review remain unverified. No model run, push, or photo sharing.


## 2026-10-23 — Clarify teammate environment setup

**Decision/reason:** Expand the viewer guide to explain local virtual environments,
explicit Python/pip/viewer checks, recording-path checks, optional activation, reopening,
and recovery from copied/broken environments. Use executable paths as the default so
activation/PATH differences do not prevent viewing. Ignore backup environment folders.

**Alternative/tradeoff:** Sharing the training environment adds unnecessary dependencies;
copying virtual environments between machines is unreliable. A separate local viewer
setup needs a one-time installation but isolates reviewer packages.

**Verification/limitations:** Documentation and whitespace checks performed; existing Mac
viewer version was previously verified as 0.33.1. Windows setup and teammate walkthrough
remain unexecuted. No dependencies installed or model operations run for this edit.

**AI assistance:** AI wrote setup explanations and recovery steps; successful teammate
installation is not claimed. No push.
