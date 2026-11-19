"""Apply approved beans review, remove spices and add reviewed household targets.

Creates a new dataset; never edits source data or checkpoints. Use with --project.
Household capture independence is unknown, so all household images stay in training.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import yaml
from validate_dataset import validate

SCHEMA='food63-no-spices-v1'
CROPS={'beans05','beans06','beans07','beans10','milk01','milk02','milk03'}

def remap(text,old_names,new_names,decision='keep'):
    result=[]
    for line in text.splitlines():
        if not line.strip(): continue
        cls,*box=line.split()
        name=old_names[int(cls)]
        if name=='spices': continue
        if decision=='relabel' and name=='beans': name='green_beans'
        result.append(str(new_names.index(name))+' '+' '.join(box))
    return '\n'.join(result)+ ('\n' if result else '')

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    project=args.project.resolve();out=args.output.resolve()
    source=project/'data/cleaned-food-v1';house=project/'data/household-annotation-draft-v1'
    if out.exists(): raise SystemExit('Refusing to overwrite '+str(out))
    cfg=yaml.safe_load((source/'data.yaml').read_text())
    old=cfg['names'];old=[old[i] for i in range(len(old))] if isinstance(old,dict) else old
    assert len(old)==64 and old[40]=='spices'
    names=[n for n in old if n!='spices']
    review=read_csv(source/'review/semantic-review.csv')
    # Spreadsheet exports may include completely empty trailing rows.
    review=[r for r in review if any((v or '').strip() for v in r.values())]
    assert len(review)==136 and len({r['image'] for r in review})==136
    decisions={r['image']:r for r in review}
    for r in review:
        assert r['status']=='reviewed' and r['decision'] in ('keep','relabel','exclude')
        assert r['source_category']=='beans'
        if r['decision']=='relabel': assert r['notes'].strip().lower() in ('change beans to green beans','change to green beans')
    manifest=read_csv(source/'manifest.csv')
    assert set(decisions).issubset({r['image'] for r in manifest})
    assert all(r['image'] in decisions for r in manifest if '5k_' in r['image'] and '_beans__' in r['image'])
    for sub in ('train','val','test'):
        for kind in ('images','labels'): (out/sub/kind).mkdir(parents=True)
    (out/'review/household-previews').mkdir(parents=True)
    changes=[];new_rows=[];excluded=Counter();touched=Counter();protected=0
    for r in manifest:
        relative=Path(r['image']);image=source/relative
        label=relative.parent.parent/'labels'/relative.with_suffix('.txt').name
        original=(source/label).read_text()
        decision=decisions.get(r['image'],{}).get('decision','keep')
        if decision=='exclude':
            excluded['beans_review_exclude']+=1
            changes.append(dict(image=r['image'],action='exclude',reason='approved beans review'));continue
        original_names=[old[int(line.split()[0])] for line in original.splitlines() if line.strip()]
        # Exclude spices-only images; do not turn them into unreviewed backgrounds.
        if original_names and set(original_names)=={'spices'}:
            excluded['spices_only_image']+=1
            changes.append(dict(image=r['image'],action='exclude',reason='removed spices class'));continue
        text=remap(original,old,names,decision)
        shutil.copy2(image,out/relative);(out/label).write_text(text,encoding='utf-8')
        row=dict(r);row['semantic_review_required']='false' if r['image'] in decisions else r.get('semantic_review_required','false')
        new_rows.append(row)
        if decision=='relabel': touched['beans_images_relabelled']+=1
        if 'spices' in original_names: touched['mixed_scene_spices_annotations_removed']+=original_names.count('spices')
        if text!=original: changes.append(dict(image=r['image'],action='rewrite_labels',reason=decision+'; explicit 64-to-63 ID remap'))
        if r['split']=='test':
            assert image.read_bytes()==(out/relative).read_bytes()
            # Baseline test has only IDs 0-29; must remain byte-for-byte identical.
            assert original==(out/label).read_text();protected+=1
    assert protected==51
    household_rows=read_csv(house/'review.csv');assert len(household_rows)==22
    previews=[]
    for r in household_rows:
        src=house/r['image'];stem=src.stem
        assert hashlib.sha256(Path(r['original_image']).read_bytes()).hexdigest()==r['original_sha256']
        with Image.open(src) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
        text=remap((house/r['label']).read_text(),old,names)
        mode='approved_full_image'
        if stem in CROPS:
            # Crop around the approved target to avoid unresolved background labels.
            fields=text.splitlines()[0].split();cls=int(fields[0]);x,y,w,h=map(float,fields[1:])
            iw,ih=im.size
            rect=(max(0,math.floor((x-w/2)*iw)),max(0,math.floor((y-h/2)*ih)),
                  min(iw,math.ceil((x+w/2)*iw)),min(ih,math.ceil((y+h/2)*ih)))
            left,top,right,bottom=rect
            x1=max(0,((x-w/2)*iw-left)/(right-left));x2=min(1,((x+w/2)*iw-left)/(right-left))
            y1=max(0,((y-h/2)*ih-top)/(bottom-top));y2=min(1,((y+h/2)*ih-top)/(bottom-top))
            im=im.crop(rect)
            text=f'{cls} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n'
            mode='approved_target_crop'
        name='household__'+stem+'.jpg';relative='train/images/'+name
        im.save(out/relative,quality=97)
        (out/'train/labels'/Path(name).with_suffix('.txt')).write_text(text)
        row=dict(image=relative,source='custom',refrigerator_id='user_household_01',
            session_id='household_batch_20261103',split='train',dataset_source='household_training',
            original_image=r['original_image'],original_split='unassigned',
            group_id='household_batch_20261103',semantic_review_required='false',annotation_mode=mode)
        new_rows.append(row)
        preview=ImageOps.contain(im,(700,700));draw=ImageDraw.Draw(preview)
        for line in text.splitlines():
            cid,x,y,w,h=map(float,line.split());pw,ph=preview.size
            rect=((x-w/2)*pw,(y-h/2)*ph,(x+w/2)*pw,(y+h/2)*ph)
            draw.rectangle(rect,outline='#00ff60',width=3)
            draw.text((max(0,rect[0]+3),max(0,rect[1]+3)),names[int(cid)],fill='red',stroke_width=1,stroke_fill='white')
        preview.save(out/'review/household-previews'/name,quality=93);previews.append((name,preview))
    cfg.update(path=str(out),names=names,nc=63,class_schema=SCHEMA,
        cleanup_status='approved beans review applied; spices removed; approved household target annotations included')
    (out/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_csv(out/'manifest.csv',new_rows)
    owners={}
    for r in new_rows:
        group=r['group_id']
        assert group not in owners or owners[group]==r['split'],group
        owners[group]=r['split']
    (out/'class-id-map.json').write_text(json.dumps([dict(old_id=i,name=n,new_id=names.index(n) if n in names else None) for i,n in enumerate(old)],indent=2))
    (out/'changes.json').write_text(json.dumps(changes,indent=2))
    shutil.copy2(source/'review/semantic-review.csv',out/'review/approved-beans-review.csv')
    shutil.copytree(source/'source-notices',out/'source-notices')
    # Build a review sheet of the final exported image/label pairs.
    for start in range(0,len(previews),6):
        sheet=Image.new('RGB',(1200,1560),'white');draw=ImageDraw.Draw(sheet)
        for j,(name,im) in enumerate(previews[start:start+6]):
            im=ImageOps.contain(im,(580,480));x=j%2*600+(600-im.width)//2;y=j//2*520+30
            sheet.paste(im,(x,y));draw.text((j%2*600+8,j//2*520+8),name,fill='black')
        sheet.save(out/f'review/household-sheet-{start//6+1}.jpg',quality=93)
    print('Copied dataset; validating all image/label pairs...',flush=True)
    report=validate(out/'data.yaml',out/'manifest.csv')
    (out/'validation-summary.json').write_text(json.dumps(report,indent=2))
    absent={s:[n for n,v in report['splits'][s]['instances'].items() if not v] for s in ('train','val')}
    assert not any(absent.values()),absent
    summary=dict(class_count=63,class_schema=SCHEMA,excluded=dict(excluded),changes=dict(touched),
        review_decisions=dict(Counter(r['decision'] for r in review)),
        image_counts={s:report['splits'][s]['images'] for s in ('train','val','test')},
        household_images=22,household_full_images=15,household_target_crops=7,
        household_split='all train, one conservative household/capture group',
        historical_test_images_unchanged=protected,absent_classes=absent,
        limitations=['Grocery validation was seen by old models; not an independent old-model comparison.',
                    'No independent household validation or final test was added.',
                    'Sparse condiment validation remains; no external accuracy claim.',
                    'Suspected vegetable overlap exclusions remain excluded.',
                    'Seven household images use target-only crops; full originals and approved drafts preserved.'],
        training_started=False)
    (out/'preparation-report.json').write_text(json.dumps(summary,indent=2))
    (out/'README.md').write_text('''# Food63 ready dataset

Approved beans review applied; spices removed with explicit old/new ID map.
Original datasets, household photos, and weights/best.pt are untouched.
22 approved household targets are training-only as a single conservative capture group.
Seven images use target crops to avoid training with missing background-food annotations.
Final export previews: review/household-sheet-1.jpg through household-sheet-4.jpg.
All 63 classes have train and validation objects; condiment counts remain sparse.
Historical 51-image public test is unchanged, not external household verification.
Review preparation-report.json and validation-summary.json for counts and limitations.
Use scripts/train_food63.py --check before explicit --run; no training started by preparation.
''')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
