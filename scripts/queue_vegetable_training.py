"""Run the vegetable experiment after the existing condiment training queue exits."""
import subprocess
import sys
import time
import json
from pathlib import Path

project = Path(r'C:\Users\donav\Documents\GitHub\LearnFFNN\Group-Project-2')
data = project / 'data/combined-condiments-vegetables'
if not (data / 'validation-summary.json').exists():
    raise RuntimeError('Vegetable dataset validation must finish before queueing training.')
summary = json.loads((data / 'validation-summary.json').read_text())
if len(summary['names']) != 63:
    raise RuntimeError('Validation summary does not yet cover all 63 classes.')
print('Vegetable dataset validated; waiting for preceding condiment training.', flush=True)
for pid in (23300, 20640, 20048):
    while str(pid) in subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                                    capture_output=True, text=True).stdout:
        time.sleep(30)
print('Starting 63-class vegetable training.', flush=True)
subprocess.run([sys.executable, '-B', str(project / 'scripts/workflow.py'), 'train',
    '--data', str(data / 'data.yaml'), '--manifest', str(data / 'manifest.csv'),
    '--epochs', '50', '--batch', '8', '--output',
    r'C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-vegetables-20261031'],
    cwd=project, check=True)
