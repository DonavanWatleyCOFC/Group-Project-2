"""Inspect saved prediction JSON in Rerun without rerunning the model."""
import argparse
import json
from io import BytesIO
import sysconfig
from pathlib import Path


def image_png(image):
    """Decode HEIC/JPEG with EXIF orientation and send viewer-supported PNG bytes."""
    from PIL import Image, ImageOps
    if Path(image).suffix.lower() in {'.heic', '.heif'}:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    with Image.open(image) as original:
        upright = ImageOps.exif_transpose(original).convert('RGB')
        buffer = BytesIO()
        upright.save(buffer, format='PNG')
        return buffer.getvalue()


def log_frame(rr, index, image, detections):
    """One selectable entity per box; clear old boxes when switching photos."""
    rr.set_time('photo', sequence=index)
    rr.log('image/detections', rr.Clear(recursive=True))
    rr.log('image', rr.EncodedImage(contents=image_png(image), media_type='image/png'))
    rr.log('image', rr.AnyValues(source_image=str(image), detection_count=len(detections)))
    for number, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['xyxy']
        label = detection['class_name']
        confidence = float(detection['confidence'])
        rr.log(
            f'image/detections/box_{number:04d}',
            rr.Boxes2D(mins=[[x1, y1]], sizes=[[x2-x1, y2-y1]],
                       labels=[f'{label} {confidence:.1%}'], show_labels=True,
                       class_ids=[int(detection['class_id'])]),
            rr.AnyValues(class_name=label, class_id=int(detection['class_id']),
                         confidence=confidence, source_image=str(image)),
        )


def show_predictions(predictions, *, spawn=True, recording=None):
    import rerun as rr
    import rerun.blueprint as rrb

    predictions = Path(predictions).resolve()
    records = json.loads(predictions.read_text(encoding='utf-8'))
    if not records:
        raise ValueError('No image records in predictions JSON.')
    images = [Path(record['image']).resolve() for record in records]
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f'Original image missing: {image}. Run from the original project root.')
    if recording and Path(recording).exists():
        raise FileExistsError(f'Refusing to overwrite recording: {recording}')
    rr.init('fridge_detection_review', strict=True)
    if spawn:
        executable = Path(sysconfig.get_path('scripts')) / ('rerun.exe' if __import__('os').name == 'nt' else 'rerun')
        rr.spawn(executable_path=str(executable))
    if recording:
        rr.save(str(recording))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial2DView(origin='image', name='Ingredient detections'),
        rrb.SelectionPanel(state='expanded'), rrb.TimePanel(state='expanded'),
    ))
    for index, (record, image) in enumerate(zip(records, images)):
        log_frame(rr, index, image, record['detections'])
    rr.disconnect()
    print(f'Logged {len(records)} photos. Select the photo timeline and click a box edge; '
          'class_name, class_id and confidence appear in the Selection panel.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--save', type=Path, help='Optional new local .rrd recording path.')
    parser.add_argument('--no-spawn', action='store_true', help='Save a recording without opening the viewer.')
    args = parser.parse_args()
    if args.no_spawn and not args.save:
        parser.error('--no-spawn requires --save')
    try:
        show_predictions(args.predictions, spawn=not args.no_spawn, recording=args.save)
    except ImportError as exc:
        raise SystemExit('Install optional viewer dependencies: python -m pip install -r requirements-viewer.txt') from exc


if __name__ == '__main__':
    main()
