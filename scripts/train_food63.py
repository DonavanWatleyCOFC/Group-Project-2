"""Validate the fixed Food63 experiment; training requires explicit --run."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
import yaml
from validate_dataset import validate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/train-food63.yaml')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load(args.config.read_text());data=(root/cfg['data']).resolve()
    report=validate(data,data.parent/'manifest.csv',verify_images=args.run)
    assert len(report['names'])==63 and 'spices' not in report['names']
    for split in ('train','val'): assert all(report['splits'][split]['instances'].values())
    import torch
    assert torch.cuda.is_available(),'CUDA unavailable; training not started'
    weights=(root/cfg['model']).resolve();assert weights.is_file(),weights
    output=(root/cfg['output']).resolve();assert not output.exists(),'Choose a new output folder'
    kwargs=dict(cfg['training']);assert kwargs['optimizer']=='AdamW'
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=kwargs)  # Check installed Ultralytics accepts every fixed setting.
    summary=dict(status='preflight passed',gpu=torch.cuda.get_device_name(0),classes=report['names'],
        image_counts={s:report['splits'][s]['images'] for s in ('train','val','test')},
        initialization=str(weights),initialization_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        output=str(output),settings=kwargs,training_started=bool(args.run))
    if args.check:
        print(json.dumps(summary,indent=2));return
    output.mkdir(parents=True)
    (output/'experiment.json').write_text(json.dumps(summary,indent=2))
    (output/'dataset-summary.json').write_text(json.dumps(report,indent=2))
    (output/'training-config.yaml').write_text(args.config.read_text())
    with (output/'requirements-lock.txt').open('w') as stream:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=stream,check=True)
    from ultralytics import YOLO
    # Generic pretrained initialization is explicit: no mismatched 64-class head reuse.
    model=YOLO(str(weights))
    model.train(data=str(data),project=str(output),name='artifacts',exist_ok=False,**kwargs)
    print('Finished; checkpoint remains in experiment folder until evaluated. Existing best.pt not replaced.')

if __name__=='__main__': main()
