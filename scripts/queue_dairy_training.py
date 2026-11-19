"""Train the validated dairy dataset after earlier queued experiments finish."""
import json
import subprocess
import sys
import time
from pathlib import Path

project = Path(r'C:\Users\donav\Documents\GitHub\LearnFFNN\Group-Project-2')
data = project / 'data/combined-condiments-vegetables-dairy'
summary = json.loads((data / 'validation-summary.json').read_text())
if len(summary['names']) != 64 or summary['names'][-1] != 'yogurt':
    raise RuntimeError('A validated 64-class dairy dataset is required.')
print('Dairy dataset validated; waiting for preceding vegetable training.', flush=True)
for pid in (23300, 20640, 20048, 18496, 21980):
    while str(pid) in subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                                    capture_output=True, text=True).stdout:
        time.sleep(30)
print('Starting 64-class dairy training.', flush=True)
subprocess.run([sys.executable, '-B', str(project / 'scripts/workflow.py'), 'train',
    '--data', str(data / 'data.yaml'), '--manifest', str(data / 'manifest.csv'),
    '--epochs', '50', '--batch', '8', '--output',
    r'C:\Users\donav\Documents\GitHub\LearnFFNN\fridge-to-recipe-dairy-20261031'],
    cwd=project, check=True)
