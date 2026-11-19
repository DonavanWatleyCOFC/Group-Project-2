"""Explicit commands for local YOLO training, held-out evaluation, and inference."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import sysconfig
import os
from datetime import datetime, timezone
from pathlib import Path

from validate_dataset import validate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def environment(torch):
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'torchvision', 'ultralytics', 'PyYAML')},
            'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def open_recording(recording):
    executable = Path(sysconfig.get_path('scripts')) / ('rerun.exe' if os.name == 'nt' else 'rerun')
    subprocess.Popen([str(executable), str(Path(recording).resolve())],
                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check-gpu')
    train = commands.add_parser('train')
    evaluate = commands.add_parser('evaluate')
    predict = commands.add_parser('predict')
    for command in (train, evaluate):
        command.add_argument('--data', required=True)
        command.add_argument('--manifest', required=True)
    for command in (train, evaluate, predict):
        command.add_argument('--output', required=True, help='New run directory; never overwrites.')
    train.add_argument('--epochs', type=int, default=50)
    train.add_argument('--batch', type=int, default=-1)
    train.add_argument('--hsv-h', type=float, default=0.015,
                       help='Maximum hue shift; preserves food color while varying fridge lighting.')
    train.add_argument('--hsv-s', type=float, default=0.35,
                       help='Maximum saturation variation for color robustness.')
    train.add_argument('--hsv-v', type=float, default=0.25,
                       help='Maximum brightness/value variation for refrigerator lighting.')
    for command in (evaluate, predict):
        command.add_argument('--weights', required=True, type=Path)
        command.add_argument('--device', default='cpu')
    predict.add_argument('--source', required=True, type=Path)
    predict.add_argument('--confidence', type=float, default=0.25)
    predict.add_argument('--rerun', action='store_true', help='Open interactive prediction boxes in Rerun.')
    predict.add_argument('--ocr', action='store_true', help='Add local PaddleOCR label evidence; retain YOLO-only predictions.')
    predict.add_argument('--ocr-python', type=Path,
                         help='Python with requirements-ocr.txt installed; defaults to .ocr-venv or current Python.')
    args = parser.parse_args()
    if args.command == 'train' and (args.epochs < 1 or args.batch == 0 or args.batch < -1):
        parser.error('Use positive epochs and batch=-1 or a positive batch size.')
    if args.command == 'train' and not all(0 <= value <= 1 for value in (args.hsv_h, args.hsv_s, args.hsv_v)):
        parser.error('HSV augmentation values must be in [0, 1].')
    if args.command == 'predict' and not 0 < args.confidence <= 1:
        parser.error('Confidence must be in (0, 1].')
    if args.command == 'predict' and args.ocr_python and not args.ocr:
        parser.error('--ocr-python requires --ocr.')
    if args.command == 'predict' and args.ocr:
        local_ocr = Path(__file__).resolve().parents[1] / '.ocr-venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        args.ocr_python = (args.ocr_python or (local_ocr if local_ocr.exists() else Path(sys.executable))).resolve()
        if not args.ocr_python.is_file():
            parser.error(f'OCR Python does not exist: {args.ocr_python}')
        check = subprocess.run([str(args.ocr_python), '-c',
            'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("paddleocr") and importlib.util.find_spec("paddle") else 1)'],
            capture_output=True, text=True)
        if check.returncode:
            parser.error('OCR dependencies unavailable. Install requirements-ocr.txt in the interpreter selected with --ocr-python.')
    if args.command == 'predict' and args.rerun:
        try:
            import rerun
        except ImportError:
            parser.error('Install viewer dependencies: python -m pip install -r requirements-viewer.txt')
    for attr in ('weights', 'source'):
        if hasattr(args, attr) and not getattr(args, attr).exists():
            parser.error(f'{attr} does not exist: {getattr(args, attr)}')
    dataset = validate(args.data, args.manifest) if args.command in ('train', 'evaluate') else None
    import torch
    if args.command == 'check-gpu':
        print(json.dumps(environment(torch), indent=2))
        if not torch.cuda.is_available():
            raise SystemExit('CUDA GPU unavailable. Training has not started.')
        return
    if args.command == 'train' and not torch.cuda.is_available():
        raise SystemExit('Training requires CUDA; no CPU/Mac fallback is enabled.')
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('Output directory already exists. Choose a new experiment name.')
    output.mkdir(parents=True)
    write_json(output / 'environment.json', environment(torch))
    with (output / 'requirements-lock.txt').open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    write_json(output / 'request.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                                       'arguments': {k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()}})
    if dataset:
        write_json(output / 'dataset-summary.json', dataset)
    from ultralytics import YOLO
    common = {'project': str(output), 'name': 'artifacts', 'exist_ok': False}
    if args.command == 'train':
        model = YOLO('yolo11n.pt')  # May download pretrained weights when explicitly run later.
        model.train(data=dataset['data'], epochs=args.epochs, imgsz=640, batch=args.batch,
                    patience=10, seed=42, device=0, workers=0, save=True, plots=True,
                    hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, **common)
        print(f'Run saved under {model.trainer.save_dir}. Back it up to shared Drive.')
    elif args.command == 'evaluate':
        model = YOLO(str(args.weights.resolve()))
        if list(model.names.values()) != dataset['names']:
            raise ValueError('Checkpoint class order differs from the dataset.')
        metrics = model.val(data=dataset['data'], split='test', imgsz=640, workers=0,
                            device=args.device, plots=True, **common)
        write_json(output / 'metrics.json', {k: float(v) for k, v in metrics.results_dict.items()})
        per_class = []
        for i, class_id in enumerate(metrics.box.ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(i)
            name = dataset['names'][int(class_id)]
            per_class.append({'class': name, 'test_instances': dataset['splits']['test']['instances'][name],
                              'precision': float(p), 'recall': float(r),
                              'mAP50': float(ap50), 'mAP50-95': float(ap)})
        write_json(output / 'per-class-metrics.json', per_class)
    else:
        records = []
        model = YOLO(str(args.weights.resolve()))
        for result in model.predict(source=str(args.source.resolve()), device=args.device,
                                    conf=args.confidence, save=True, stream=True, **common):
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    detections.append({'class_id': cls, 'class_name': result.names[cls],
                                       'confidence': float(box.conf.item()),
                                       'xyxy': box.xyxy[0].cpu().tolist()})
            records.append({'image': str(result.path), 'detections': detections})
        write_json(output / 'predictions.json', records)
        if args.ocr:
            write_json(output / 'class-names.json', list(model.names.values()))
            subprocess.run([str(args.ocr_python), '-B', str(Path(__file__).with_name('ocr_pipeline.py')),
                '--baseline', str(output / 'predictions.json'), '--names', str(output / 'class-names.json'),
                '--output', str(output / 'ocr')], check=True)
            print(f'YOLO-only: {output / "predictions.json"}; YOLO + OCR: {output / "ocr/predictions.json"}')
        if args.rerun:
            if args.ocr:
                subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('ocr_compare_viewer.py')),
                    '--baseline', str(output / 'predictions.json'), '--plus', str(output / 'ocr/predictions.json'),
                    '--save', str(output / 'review.rrd'), '--class-count', str(len(model.names))], check=True)
            else:
                from rerun_viewer import show_predictions
                show_predictions(output / 'predictions.json', spawn=False, recording=output / 'review.rrd')
            open_recording(output / 'review.rrd')


if __name__ == '__main__':
    main()
