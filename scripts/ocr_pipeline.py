"""Read product labels with PaddleOCR and compare with frozen YOLO predictions.

Run in .ocr-venv after baseline inference. OCR-added rectangles locate text,
not whole products. Confidence scores from OCR and YOLO are not interchangeable.
"""
import argparse
import copy
import json
import re
from itertools import combinations
from pathlib import Path

ALIASES = {
    'butter': ['butter'], 'milk': ['whole milk', 'milk'],
    'spinach': ['baby spinach', 'spinach'], 'spices': ['paprika', 'cumin', 'black pepper', 'seasoning', 'cinnamon'],
    'cheese': ['mozzarella', 'cheddar', 'parmesan', 'cheese'],
    'beans': ['kidney beans', 'black beans', 'pinto beans', 'white beans', 'baked beans'],
    'green_beans': ['green beans', 'cut green beans', 'french style green beans'],
    'corn': ['sweet corn', 'corn'], 'flour': ['flour'], 'sugar': ['sugar'],
    'juice': ['orange juice', 'apple juice', 'juice'], 'water': ['spring water', 'purified water'],
    'soda': ['diet coke', 'coca cola', 'sprite', 'dr pepper', 'pepsi', 'fanta'],
    'tomato_sauce': ['tomato sauce', 'pasta sauce'], 'oil': ['olive oil', 'vegetable oil'],
    'vinegar': ['vinegar'], 'honey': ['honey'], 'jam': ['jam'],
    'rice': ['rice'], 'pasta': ['pasta'], 'nuts': ['almonds', 'walnuts', 'cashews'],
}
ALIASES.update({'ketchup':['ketchup'], 'mustard':['mustard'], 'mayonnaise':['mayonnaise'],
                'relish':['relish'], 'salsa':['salsa'], 'yogurt':['yogurt'],
                'broccoli':['broccoli'], 'bell_pepper':['bell pepper'],
                'cucumber':['cucumber'], 'peas':['peas'], 'cauliflower':['cauliflower']})
EXCLUDE = re.compile(r'ingredients|contains|nutrition|allergen|recipe|distributed|manufactured', re.I)


def match_label(text):
    normalized = re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()
    if EXCLUDE.search(normalized) or len(normalized.split()) > 6:
        return None
    matches = {name for name, aliases in ALIASES.items()
               if any(re.search(r'\b' + re.escape(alias) + r'\b', normalized) for alias in aliases)}
    return next(iter(matches)) if len(matches) == 1 else None


def intersection(a, b):
    return max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))


def area(box):
    return max(0, box[2]-box[0])*max(0,box[3]-box[1])


def grouped_labels(evidence):
    """Join 2-3 aligned, nearby lines only when their full phrase is a known alias.

    Each original line must pass the confidence threshold. Repeated OCR crops are
    deduplicated before grouping; grouped score is the weakest component score.
    Spatial proximity is a heuristic, not proof of shared product ownership.
    """
    def words(text):
        return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()

    aliases = {alias: name for name, phrases in ALIASES.items()
               for alias in phrases if ' ' in alias}
    vocabulary = {word for alias in aliases for word in alias.split()}
    lines = []
    for item in sorted(evidence, key=lambda row: row['score'], reverse=True):
        token = words(item['text'])
        if (item['score'] < .85 or not token or EXCLUDE.search(token)
                or not set(token.split()).issubset(vocabulary)):
            continue
        if any(words(other['text']) == token and
               intersection(item['xyxy'], other['xyxy']) / max(min(area(item['xyxy']),area(other['xyxy'])),1) > .6
               for other in lines):
            continue
        lines.append(item)
    groups = []
    for count in (2, 3):
        for selected in combinations(lines, count):
            ordered = sorted(selected, key=lambda item: (item['xyxy'][1]+item['xyxy'][3])/2)
            phrase = ' '.join(words(item['text']) for item in ordered)
            if phrase not in aliases:
                continue
            valid = True
            for upper, lower in zip(ordered, ordered[1:]):
                a,b = upper['xyxy'], lower['xyxy']
                ha,hb = a[3]-a[1], b[3]-b[1]
                overlap = max(0,min(a[2],b[2])-max(a[0],b[0]))
                widths = min(a[2]-a[0], b[2]-b[0])
                advance = (b[1]+b[3]-a[1]-a[3])/2
                gap = b[1]-a[3]
                if (min(ha,hb,widths) <= 0 or max(ha,hb)/min(ha,hb) > 2.5
                        or overlap/widths < .65 or advance < .3*min(ha,hb)
                        or gap > 1.25*max(ha,hb)):
                    valid = False
                    break
            if not valid:
                continue
            box = [min(item['xyxy'][0] for item in ordered), min(item['xyxy'][1] for item in ordered),
                   max(item['xyxy'][2] for item in ordered), max(item['xyxy'][3] for item in ordered)]
            # A nearby ingredient/nutrition header in the same column makes this
            # unsuitable as product-name evidence.
            if any(EXCLUDE.search(header['text']) and
                   header['xyxy'][1] <= box[1] and
                   0 <= box[1]-header['xyxy'][3] <= 4*(box[3]-box[1]) and
                   max(0,min(box[2],header['xyxy'][2])-max(box[0],header['xyxy'][0])) > .5*(box[2]-box[0])
                   for header in evidence):
                continue
            groups.append({'text':phrase, 'score':min(item['score'] for item in ordered),
                           'xyxy':box, 'matched_class':aliases[phrase],
                           'evidence_kind':'nearby_lines', 'source_lines':copy.deepcopy(ordered)})
    return groups


def fuse(detections, evidence, names):
    plus = copy.deepcopy(detections)
    actions = []
    # Highest-confidence OCR lines first; never force conflicting keywords.
    for item in sorted(evidence, key=lambda entry: entry['score'], reverse=True):
        name = item.get('matched_class')
        if not name or item['score'] < .85 or name not in names:
            continue
        box = item['xyxy']
        owners = [(i, det) for i,det in enumerate(plus)
                  if det.get('box_kind', 'object') == 'object'
                  and intersection(box,det['xyxy']) / max(area(box),1) > .8]
        if len(owners) == 1:
            index, det = owners[0]
            if det.get('ocr_class') and det['ocr_class'] != name:
                actions.append({'action':'conflicting_text_review', 'text':item['text']})
                continue
            old = det['class_name']
            det.setdefault('yolo_class_name', old)
            det.setdefault('yolo_confidence', det['confidence'])
            det.update(class_name=name, class_id=names.index(name), ocr_class=name,
                       evidence_text=item['text'], ocr_score=item['score'],
                       confidence=item['score'], confidence_kind='ocr_recognition',
                       method='yolo+ocr', review_required=True)
            if item.get('source_lines'):
                det['source_lines'] = item['source_lines']
            actions.append({'action':'confirmed' if old == name else 'corrected',
                            'from':old, 'to':name, 'text':item['text']})
        elif not owners:
            # Avoid repeated line/crop evidence at the same position.
            if any(det['class_name'] == name and intersection(box,det['xyxy']) / max(min(area(box),area(det['xyxy'])),1) > .3
                   for det in plus):
                continue
            plus.append({'class_name':name,'class_id':names.index(name),
                         'xyxy':box,'confidence':item['score'],'ocr_score':item['score'],
                         'method':'ocr_text_candidate','box_kind':'text_region','confidence_kind':'ocr_recognition',
                         'evidence_text':item['text'],'review_required':True})
            if item.get('source_lines'):
                plus[-1]['source_lines'] = item['source_lines']
            actions.append({'action':'text_candidate','to':name,'text':item['text']})
    return plus, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--names', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cached-ocr', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    records = json.loads(args.baseline.read_text())
    names = json.loads(args.names.read_text())
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected a nonempty unique checkpoint class list')
    engine = None
    cached = json.loads(args.cached_ocr.read_text()) if args.cached_ocr else {}
    if any(str(Path(record['image'])) not in cached for record in records):
        from paddleocr import PaddleOCR
        engine = PaddleOCR(lang='en', ocr_version='PP-OCRv5', device='cpu',
                          text_detection_model_name='PP-OCRv5_mobile_det',
                          text_recognition_model_name='en_PP-OCRv5_mobile_rec',
                          use_doc_orientation_classify=False, use_doc_unwarping=False,
                          use_textline_orientation=False, enable_mkldnn=False, cpu_threads=4)
    all_ocr, results = {}, []
    for record in records:
        path = Path(record['image'])
        evidence = cached.get(str(path))
        if evidence is None:
            from PIL import Image, ImageOps
            import numpy as np
            if path.suffix.lower() in {'.heic', '.heif'}:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            with Image.open(path) as original:
                img = ImageOps.exif_transpose(original).convert('RGB')
                width,height = img.size
                # Whole-image OCR plus four overlapping tiles recovers small labels.
                regions = [(0,0,width,height)]
                for x in (0,int(width*.4)):
                    for y in (0,int(height*.4)):
                        regions.append((x,y,min(width,x+int(width*.6)),min(height,y+int(height*.6))))
                evidence = []
                for x,y,right,bottom in regions:
                    crop = np.asarray(img.crop((x,y,right,bottom)))
                    for result in engine.predict(crop):
                        for text,score,poly in zip(result['rec_texts'],result['rec_scores'],result['rec_polys']):
                            points = np.asarray(poly)
                            box = [float(points[:,0].min()+x),float(points[:,1].min()+y),
                                   float(points[:,0].max()+x),float(points[:,1].max()+y)]
                            evidence.append({'text':text,'score':float(score),'xyxy':box,
                                             'matched_class':match_label(text)})
            print(path.name, 'OCR lines:',len(evidence), flush=True)
        for item in evidence:
            item['matched_class'] = match_label(item['text'])
        all_ocr[str(path)] = evidence
        (args.output/'ocr.json').write_text(json.dumps(all_ocr,indent=2))
        groups = grouped_labels(evidence)
        detections,actions = fuse(record['detections'],evidence+groups,names)
        results.append({'image':str(path),'detections':detections,'actions':actions,
                        'grouped_ocr':groups})
    (args.output/'predictions.json').write_text(json.dumps(results,indent=2))
    summaries = []
    for original,plus in zip(records,results):
        summaries.append({'image':Path(original['image']).name,
            'baseline_classes':sorted({d['class_name'] for d in original['detections']}),
            'plus_classes':sorted({d['class_name'] for d in plus['detections']}),
            'actions':plus['actions']})
    (args.output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    from PIL import Image, ImageOps, ImageDraw
    for original,plus in zip(records,results):
        with Image.open(original['image']) as source:
            source = ImageOps.exif_transpose(source).convert('RGB')
            scale = 1000/source.width
            height = round(source.height*scale)
            canvas = Image.new('RGB',(2000,height+40),'white')
            for column,record in enumerate((original,plus)):
                canvas.paste(source.resize((1000,height)),(column*1000,40))
                draw = ImageDraw.Draw(canvas)
                draw.text((column*1000+10,10),f'YOLO {len(names)}' if column==0 else 'YOLO + OCR; orange = text region',fill='black')
                for det in record['detections']:
                    x1,y1,x2,y2 = det['xyxy']
                    box = [column*1000+x1*scale,40+y1*scale,column*1000+x2*scale,40+y2*scale]
                    color = 'orange' if det.get('box_kind')=='text_region' else 'cyan'
                    draw.rectangle(box,outline=color,width=3)
                    draw.text((box[0],max(40,box[1]-14)),det['class_name'],fill=color,stroke_width=1,stroke_fill='black')
            canvas.save(args.output/(Path(original['image']).stem+'-comparison.png'))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
