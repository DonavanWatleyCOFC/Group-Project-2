"""Build a structurally cleaned, versioned dataset without changing originals.

Semantic labels are never guessed. Grouped grocery validation is for FUTURE
training runs only: these images were seen by earlier checkpoints.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps


def identity(filename):
    """Original photo, before Roboflow export IDs; keep source namespaces."""
    body = filename.split('__', 1)[-1].split('.rf.', 1)[0]
    if filename.startswith(('baseline__', 'aicook__')):
        return 'fridge', body, 'fridge'
    match = re.match(r'5k_(?:train|val)_(.+?)__', filename)
    if match:
        return '5k', body, match.group(1)
    for prefix,source in [('condiments_', 'condiments'),('vegetables_', 'vegetables'),('dairy_', 'dairy')]:
        if filename.startswith(prefix):
            return source, body, source
    raise ValueError(f'Unknown source prefix: {filename}')


def validate_rows(text, count):
    rows=[]
    for number,line in enumerate(text.splitlines(),1):
        if not line.strip():
            continue
        parts=line.split()
        if len(parts)!=5:
            raise ValueError(f'line {number}: expected detection box')
        cls,x,y,w,h=map(float,parts)
        if not all(math.isfinite(v) for v in (cls,x,y,w,h)):
            raise ValueError(f'line {number}: non-finite values')
        if not cls.is_integer() or not 0<=cls<count:
            raise ValueError(f'line {number}: invalid class ID')
        if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
            raise ValueError(f'line {number}: invalid dimensions')
        if min(x-w/2,y-h/2)<-1e-5 or max(x+w/2,y+h/2)>1+1e-5:
            raise ValueError(f'line {number}: box outside image')
        rows.append((int(cls),x,y,w,h))
    return rows


def dhash(image):
    pixels=list(image.convert('L').resize((9,8)).getdata())
    value=0
    for y in range(8):
        for x in range(8):
            value=(value<<1) | (pixels[y*9+x]>pixels[y*9+x+1])
    return value


def link_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    # Independent image copies allow later manual edits without touching originals.
    shutil.copy2(source,target)


def union_groups(records):
    """Group original photos and conservative same-category visual neighbors."""
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[b]=a
    canonical={}
    buckets=defaultdict(list)
    for i,record in enumerate(records):
        key=(record['source'],record['photo'])
        if key in canonical:
            join(i,canonical[key])
        else:
            canonical[key]=i
        buckets[(record['source'],record['bucket'])].append(i)
    near_pairs=0
    for indices in buckets.values():
        for pos,i in enumerate(indices):
            for j in indices[:pos]:
                if (records[i]['dhash']^records[j]['dhash']).bit_count()<=4:
                    join(i,j)
                    near_pairs+=1
    groups=defaultdict(list)
    for i,record in enumerate(records):
        groups[find(i)].append(record)
    return list(groups.values()),near_pairs


def choose_validation(groups, fraction=.2):
    """Deterministic source-stratified split with positive coverage where possible."""
    groups=sorted(groups,key=lambda g:min(r['original'] for r in g))
    random.Random(42).shuffle(groups)
    classes=[set(row[0] for record in group for row in record['rows']) for group in groups]
    available=Counter(cls for values in classes for cls in values)
    selected=set()
    covered=set()
    remaining=available.copy()
    target=max(1,round(sum(map(len,groups))*fraction))
    size=0
    # Ensure every class with at least two independent groups has val coverage.
    for i,values in enumerate(classes):
        needed={cls for cls in values if cls not in covered and available[cls]>=2}
        if needed and all(remaining[cls]>1 for cls in values):
            selected.add(i); covered.update(values); remaining.subtract(values); size+=len(groups[i])
    for i,values in enumerate(classes):
        if i not in selected and size<target and all(remaining[cls]>1 for cls in values):
            selected.add(i); covered.update(values); remaining.subtract(values); size+=len(groups[i])
    return [(group,'val' if i in selected else 'train') for i,group in enumerate(groups)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    project=args.project.resolve()
    source=project/'data/combined-condiments-vegetables-dairy'
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config=yaml.safe_load((source/'data.yaml').read_text())
    names=config['names']
    assert len(names)==64 and names[-1]=='yogurt'
    output.mkdir(parents=True)
    for split in ('train','val','test'):
        for sub in ('images','labels'):
            (output/split/sub).mkdir(parents=True)
    manifest=list(csv.DictReader((source/'manifest.csv').open(encoding='utf-8')))
    baseline_photos={identity(Path(row['image']).name)[1] for row in manifest if Path(row['image']).name.startswith('baseline__')}
    removed=[]; quarantined=[]; records=[]; content={}; decoded={}
    # Keep the historical test and source validation first when exact duplicates occur.
    ordering={'test':0,'val':1,'train':2}
    for index,row in enumerate(sorted(manifest,key=lambda r:(ordering[r['split']],r['image']))):
        image=source/row['image']; label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        src,photo,bucket=identity(image.name)
        if image.name.startswith('aicook__') and photo in baseline_photos:
            removed.append({'image':row['image'],'reason':'same_original_photo_as_baseline','source':'aicook'})
            continue
        try:
            text=label.read_text(encoding='utf-8')
            boxes=validate_rows(text,len(names))
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            with Image.open(image) as opened:
                im=ImageOps.exif_transpose(opened).convert('RGB')
                dimensions=im.size
                pixel_digest=hashlib.sha256(str(dimensions).encode()+im.tobytes()).hexdigest()
                visual_hash=dhash(im) if src in ('5k','condiments') else None
            annotation=tuple(sorted(boxes))
            previous=content.get(digest) or decoded.get(pixel_digest)
            if previous:
                if previous['annotation']!=annotation:
                    # Preserve the prior protected example; quarantine the conflicting copy.
                    raise ValueError('identical image with conflicting annotations: '+previous['image'])
                if row['split']=='test':
                    raise ValueError('duplicate historical test image requires manual review')
                removed.append({'image':row['image'],'reason':'identical_file_or_decoded_pixels','kept':previous['image']})
                continue
            content[digest]=decoded[pixel_digest]={'annotation':annotation,'image':row['image']}
            record={'original':row['image'],'original_split':row['split'],'path':image,'label':label,
                'text':text,'rows':boxes,'source':src,'photo':photo,'bucket':bucket,
                'dimensions':dimensions,'dhash':visual_hash,'sha256':digest}
            records.append(record)
        except Exception as exc:
            quarantined.append({'image':row['image'],'reason':str(exc)})
            link_copy(image,output/'quarantine/images'/image.name)
            if label.exists():
                target=output/'quarantine/labels'/label.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(label,target)
        if index and index%1500==0:
            print('Inspected',index,'of',len(manifest),flush=True)
    mutable=[r for r in records if r['source'] in ('5k','condiments')]
    fixed=[r for r in records if r['source'] not in ('5k','condiments')]
    # Preserve existing split boundaries; exclude train siblings of held-out photos.
    protected={(r['source'],r['photo']) for r in fixed if r['original_split'] in ('test','val')}
    assignments=[]
    for record in fixed:
        if record['original_split']=='train' and (record['source'],record['photo']) in protected:
            removed.append({'image':record['original'],'reason':'potential_heldout_original_or_mosaic_overlap'})
            continue
        gid=hashlib.sha256((record['source']+'/'+record['photo']).encode()).hexdigest()[:16]
        assignments.append((record,record['original_split'],gid))
    groups,pairs=union_groups(mutable)
    buckets=defaultdict(list)
    for group in groups:
        buckets[(group[0]['source'],group[0]['bucket'])].append(group)
    for bucket,bucket_groups in sorted(buckets.items()):
        for group,split in choose_validation(bucket_groups):
            gid=hashlib.sha256('|'.join(sorted(r['original'] for r in group)).encode()).hexdigest()[:16]
            assignments.extend((record,split,gid) for record in group)
    # Verify group ownership and all retained historical test bytes/labels.
    owners={}; counts={s:Counter() for s in ('train','val','test')}; images=Counter()
    review=[]; output_rows=[]; group_rows=[]
    for record,split,gid in assignments:
        if gid in owners and owners[gid]!=split:
            raise ValueError('group crosses splits: '+gid)
        owners[gid]=split
        target=output/split/'images'/record['path'].name
        link_copy(record['path'],target)
        (output/split/'labels'/record['label'].name).write_text(record['text'],encoding='utf-8')
        counts[split].update(names[row[0]] for row in record['rows']); images[split]+=1
        image_relative=target.relative_to(output).as_posix()
        flagged=record['source']=='5k' and record['bucket'] in ('beans','spices')
        output_rows.append([image_relative,'public_unknown','','',split,record['source'],record['original'],record['original_split'],gid,str(flagged).lower()])
        group_rows.append({'image':image_relative,'source':record['source'],'original_photo':record['photo'],'group_id':gid,'split':split})
        if flagged:
            reason='beans versus green_beans; confirm bean type' if record['bucket']=='beans' else 'define spices versus salt and other seasonings'
            review.append({'image':image_relative,'label':f'{split}/labels/{record["label"].name}',
                'split':split,'source_category':record['bucket'],'group_id':gid,'reason':reason,
                'status':'pending','decision':'','notes':''})
    old_test={r['image']:source/r['image'] for r in manifest if r['split']=='test'}
    new_test={r[0]:output/r[0] for r in output_rows if r[4]=='test'}
    assert set(old_test)==set(new_test), 'Historical test membership changed'
    for key,path in old_test.items():
        assert path.read_bytes()==new_test[key].read_bytes()
        assert (path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text()==(new_test[key].parent.parent/'labels'/path.with_suffix('.txt').name).read_text()
    with (output/'manifest.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['image','source','refrigerator_id','session_id','split','dataset_source','original_image','original_split','group_id','semantic_review_required'])
        writer.writerows(output_rows)
    (output/'review').mkdir(exist_ok=True)
    with (output/'review/semantic-review.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','label','split','source_category','group_id','reason','status','decision','notes'])
        writer.writeheader(); writer.writerows(review)
    (output/'review/original-photo-groups.json').write_text(json.dumps(group_rows,indent=2))
    (output/'review/removed-images.json').write_text(json.dumps(removed,indent=2))
    (output/'review/quarantined-images.json').write_text(json.dumps(quarantined,indent=2))
    # Annotated review sheets: filenames are keys into the review CSV.
    for category in ('beans','spices'):
        samples=[r for r in review if r['source_category']==category]
        for start in range(0,len(samples),20):
            sheet=Image.new('RGB',(5*240,4*270),'white'); draw=ImageDraw.Draw(sheet)
            for pos,item in enumerate(samples[start:start+20]):
                path=output/item['image']; col=pos%5; row=pos//5
                with Image.open(path) as original:
                    tile=ImageOps.exif_transpose(original).convert('RGB').resize((224,224))
                td=ImageDraw.Draw(tile)
                for cls,x,y,w,h in validate_rows((output/item['label']).read_text(),len(names)):
                    box=[(x-w/2)*224,(y-h/2)*224,(x+w/2)*224,(y+h/2)*224]
                    td.rectangle(box,outline='red',width=2)
                sheet.paste(tile,(col*240+8,row*270+38))
                identifier=path.stem.split('__')[-1].split('.rf.')[0]
                draw.text((col*240+8,row*270+5),f'{category} {identifier}',fill='black')
                draw.text((col*240+8,row*270+20),item['split'],fill='black')
            sheet.save(output/f'review/{category}-{start//20+1:02d}.jpg',quality=90)
    config.update(path=str(output),nc=len(names),names=names,
        cleanup_status='structural_cleanup_complete_semantic_review_pending')
    (output/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summary={'source_dataset':str(source),'output':str(output),'class_count':len(names),
        'class_ids_unchanged':True,'image_counts':dict(images),'instances':counts,
        'removed_counts':dict(Counter(row['reason'] for row in removed)),
        'quarantined_images':len(quarantined),'perceptual_neighbor_pairs_grouped':pairs,
        'grocery_original_and_visual_groups':len(groups),'semantic_review_images':len(review),
        'validation_absent_classes':[name for name in names if counts['val'][name]==0],
        'test_unchanged':True,'split_seed':42,'grocery_validation_target_fraction':.2,
        'limitations':['New grocery validation was seen by previous checkpoints; valid only for future retraining.',
          'Original-photo/dHash grouping is not verified product, household or capture-session independence.',
          'Semantic labels are unchanged and flagged for human review.','No training was started.']}
    (output/'cleanup-report.json').write_text(json.dumps(summary,indent=2))
    sys.path.insert(0,str(project/'scripts'))
    from validate_dataset import validate
    validation=validate(output/'data.yaml',output/'manifest.csv')
    (output/'validation-summary.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
