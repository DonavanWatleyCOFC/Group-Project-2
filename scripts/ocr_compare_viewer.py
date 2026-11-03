"""Save a side-by-side YOLO / YOLO+OCR Rerun recording."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rerun_viewer import image_png

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--plus',type=Path,required=True)
    parser.add_argument('--save',type=Path,required=True)
    parser.add_argument('--class-count',type=int,default=44)
    args=parser.parse_args()
    if args.save.exists():
        raise FileExistsError(args.save)
    import rerun as rr
    import rerun.blueprint as rrb
    baseline=json.loads(args.baseline.read_text())
    plus=json.loads(args.plus.read_text())
    assert [r['image'] for r in baseline]==[r['image'] for r in plus]
    rr.init('yoloPlus_comparison',strict=True)
    rr.save(str(args.save))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial2DView(origin='baseline',name=f'YOLO {args.class_count} baseline'),
        rrb.Spatial2DView(origin='plus',name='YOLO + PaddleOCR (orange = text candidate)')),
        rrb.TimePanel(state='expanded'),rrb.SelectionPanel(state='expanded')))
    for index,(before,after) in enumerate(zip(baseline,plus)):
        rr.set_time('photo',sequence=index)
        for root,record in [('baseline',before),('plus',after)]:
            rr.log(root+'/boxes',rr.Clear(recursive=True))
            rr.log(root,rr.EncodedImage(contents=image_png(record['image']),media_type='image/png'))
            for number,det in enumerate(record['detections']):
                x1,y1,x2,y2=det['xyxy']
                text_only=det.get('box_kind')=='text_region'
                label=det['class_name'] + (' [OCR text only]' if text_only else '')
                rr.log(f'{root}/boxes/{number}',rr.Boxes2D(mins=[[x1,y1]],sizes=[[x2-x1,y2-y1]],
                    labels=[label],colors=[[255,160,0] if text_only else [0,220,180]],show_labels=True),
                    rr.AnyValues(method=det.get('method','yolo'),
                        evidence=det.get('evidence_text',''),confidence=det['confidence'],
                        confidence_kind=det.get('confidence_kind','yolo_detection'),
                        original_yolo_class=det.get('yolo_class_name',det['class_name']),
                        review_required=det.get('review_required',False)))
    rr.disconnect()
    print('Saved comparison:',args.save)

if __name__=='__main__':
    main()
