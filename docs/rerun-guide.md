# View our ingredient detections in Rerun

Rerun lets you view the refrigerator photos, move between them, and click detection
boxes to see what the model classified each object as and its confidence score.
Opening the recording does not train or run the model.

## 1. Get the recording

Ask Donavan for **`review-heic-fixed.rrd`** (about 75 MB). Use this corrected file;
the older `review.rrd` has black frames for HEIC photos.

The recording contains the photos, boxes, class labels, and confidence values.
**You do not need `best.pt`, the original photos, the training dataset, or PyTorch
just to view this file.** The recording is not included when you clone/pull the repo;
Donavan must send it separately. It contains personal refrigerator images, so keep
sharing within the approved team.

Save it in this location inside your copy of the project:

```text
Group-Project-2/
  runs/
    external-review-01/
      review-heic-fixed.rrd
```

Create the `runs` and `external-review-01` folders if needed. Keep the filename exactly
as shown, or adjust the open command to match your filename.

## 2. Install the viewer once

Use **Rerun 0.33.1**, the same version used to generate this recording. The Python
package includes the desktop viewer. No Roboflow account, credit card, or NVIDIA GPU
is needed to open our local recording.

### What is the environment?

A virtual environment is a local folder containing Python and packages for this task.
Each teammate creates their own `.rerun-venv`; pulling Git does not create it or install
anything. Do not copy an environment between computers or operating systems.
For viewing our saved recording, install only Rerun. The full training environment,
CUDA, PyTorch, and `requirements.txt` are not required.

Run each command below separately and wait for it to finish. If a command fails, stop
and save the full error rather than continuing with later steps.

### Windows (PowerShell)

Install Python 3.11 from [python.org](https://www.python.org/downloads/) if Python is
not already installed. Open a new PowerShell window after installation.

Open the project folder in VS Code, then choose **Terminal → New Terminal** and use
PowerShell. Check that the terminal is in `Group-Project-2` (the folder containing
`scripts` and `docs`). Otherwise, use `cd` with your own project location:

```powershell
cd "C:\path\to\Group-Project-2"
```

Replace that example path; do not paste it unchanged. Install into a separate viewer
environment so the training environment is unaffected:

```powershell
py -3.11 -m venv .rerun-venv
.\.rerun-venv\Scripts\python.exe -m pip install "rerun-sdk==0.33.1"
```

Check the environment before opening the file:

```powershell
.\.rerun-venv\Scripts\python.exe --version
.\.rerun-venv\Scripts\python.exe -m pip show rerun-sdk
.\.rerun-venv\Scripts\rerun.exe --version
Test-Path "runs\external-review-01\review-heic-fixed.rrd"
```

The package and viewer versions should be **0.33.1**. `Test-Path` should print **True**;
if it prints False, put the recording in the expected folder first.

No environment activation or PowerShell execution-policy change is required. Always
using the explicit executable paths avoids installing into the wrong Python environment.

### Mac (Terminal)

Install Python 3.11 from [python.org](https://www.python.org/downloads/) if needed.
Open Terminal, or use **Terminal → New Terminal** in VS Code with the project open.
Navigate to your project folder:

```bash
cd "/path/to/Group-Project-2"
```

Replace the example path with your own. Install the viewer:

```bash
python3 -m venv .rerun-venv
./.rerun-venv/bin/python -m pip install "rerun-sdk==0.33.1"
```

Check the environment before opening the file:

```bash
./.rerun-venv/bin/python --version
./.rerun-venv/bin/python -m pip show rerun-sdk
./.rerun-venv/bin/rerun --version
ls "runs/external-review-01/review-heic-fixed.rrd"
```

The package and viewer versions should be **0.33.1**, and `ls` should find the recording.
No environment activation is needed when using these explicit paths.

If you already have the project's working `.venv`, you can use that instead; substitute
`.venv` for `.rerun-venv` in the commands below.

## 3. Open the recording

Run the command for your operating system **from the project root**.
These commands explicitly use the installed viewer so another global Rerun version
cannot accidentally open the file.

**Windows:**

```powershell
.\.rerun-venv\Scripts\rerun.exe "runs\external-review-01\review-heic-fixed.rrd"
```

**Mac:**

```bash
./.rerun-venv/bin/rerun "runs/external-review-01/review-heic-fixed.rrd"
```

For Donavan's existing Mac environment:

```bash
./.venv/bin/rerun "runs/external-review-01/review-heic-fixed.rrd"
```

If an environment is activated and has the correct Rerun version, the short command is:

```bash
rerun "runs/external-review-01/review-heic-fixed.rrd"
```

You can also use an absolute path to the recording if you saved it elsewhere. Put paths
in quotes, especially when they contain spaces. On later visits, repeat only the open
command; no reinstall is needed.

### Next time you open the project

Open a terminal in `Group-Project-2` and run the Windows or Mac command from Step 3.
You do not need to recreate the environment or reinstall Rerun each time.

If you prefer activating the environment, Mac users can run
`source .rerun-venv/bin/activate`; Windows PowerShell users can run
`.\.rerun-venv\Scripts\Activate.ps1`. Activation is optional. If PowerShell blocks it,
use the explicit executable commands instead of changing security settings.
After successful activation, `python` and `rerun` refer to that environment; `deactivate`
leaves it. In VS Code, **Python: Select Interpreter** can select this environment for
editor use, but does not replace installing its packages or using the correct terminal.

### If an existing environment seems broken

First confirm you are in the project root. Run the explicit version checks above.
If the environment's Python works but Rerun is missing or has the wrong version,
repeat that operating system's `python -m pip install "rerun-sdk==0.33.1"` command
using the environment's full Python path from Step 2.

If the environment was copied from another machine, renamed/moved, or its Python
executable no longer works, close terminals using it, rename `.rerun-venv` to
`.rerun-venv-backup` in your file manager, and repeat Step 2 to create a fresh one.
Use another backup name if that name already exists. Do not rename your data or runs
folders. Keep the backup local; it must not be committed either.

## 4. Explore the results

1. Find the timeline at the bottom of the viewer. Select **`photo`** as the timeline,
   rather than `log_time` or `log_tick`.
2. Pause playback if it is moving. Click or drag the timeline cursor between photo
   indices **0–5** to see the six recorded images.
3. Each predicted object has a box with its predicted class and confidence, such as
   `apple 87.5%`. This is an example label, not a claim about a particular photo.
4. **Click the edge of a box.** Its details appear in the **Selection** panel:
   `class_name` is the predicted category, `class_id` is its numeric ID, and
   `confidence` is its numeric model score (0.875 means 87.5%). `source_image` identifies
   the photo; an original Mac path in this field does not mean that file is required.
5. If boxes overlap, use the entity tree to select a child such as
   `image/detections/box_0000`. If the Selection panel is hidden, show it through the
   viewer's panel controls.

Box numbers are local to a photo, not persistent object identities across photos.
A photo with no predictions should have no boxes; that is different from a black image.
Confidence is **not** a verified probability of correctness. A high-confidence label
can be wrong. This viewer shows predictions, not the model's internal reasoning.

## 5. Help review the model

For each issue, send Donavan the photo name or timeline index, the predicted label,
what you think the object actually is, and a short description. Record missed items,
incorrect labels, duplicate boxes, or boxes that do not align with objects. It is okay
to say an object is ambiguous rather than guessing.

Bell pepper, broccoli, and yogurt are outside the current 30-class baseline. These
photos support qualitative review; without reviewed ground-truth annotations they do
not produce an accuracy score.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Python or `py` not found | Install Python, reopen the terminal, and retry. On Windows, if `py` is unavailable but `python --version` works, use `python` for the environment creation command. |
| Rerun command not found | Use the explicit `.rerun-venv` executable path in Step 3 and confirm Step 2 completed. |
| File not found | Confirm your current folder and the file's spelling. The recording is sent separately from Git. |
| Black images except fridge_03 | Open `review-heic-fixed.rrd`, not the original `review.rrd`. |
| Version mismatch warning | Use the explicit environment viewer path; check its version with `--version`. This recording was written with 0.33.1. |
| Only one photo appears | Select the `photo` timeline and move its cursor between indices 0–5. |
| Installation or graphics error | Send the full error, operating system, Python version, and command used to Donavan. Windows viewer setup has not yet been independently verified. |

Do not commit `.rerun-venv` or `.rrd` files. Both are ignored by the project.

Official references: [Rerun getting started](https://rerun.io/docs/getting-started),
[opening saved recordings](https://rerun.io/docs/reference/sdk/operating-modes).
