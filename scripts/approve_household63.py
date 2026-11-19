"""Export approved household boxes separately while beans decisions are restored."""
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
import yaml
from prepare_food63 import CROPS,read_csv,remap,write_csv

def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'data/household-annotation-draft-v1'
    output=root/'data/household-approved63-v1'
    assert not output.exists(),'Refusing to overwrite approved household export'
    old=yaml.safe_load((source/'class-names.yaml').read_text())['names']
    names=[n for n in old if n!='spices']
    for sub in ('images','labels','previews'): (output/sub).mkdir(parents=True)
    rows=[];previews=[]
    for r in read_csv(source/'review.csv'):
        path=source/r['image'];stem=path.stem
        assert hashlib.sha256(Path(r['original_image']).read_bytes()).hexdigest()==r['original_sha256']
        with Image.open(path) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
        text=remap((source/r['label']).read_text(),old,names)
        mode='approved_full_image'
        if stem in CROPS:
            cls,x,y,w,h=map(float,text.splitlines()[0].split());iw,ih=im.size
            left=max(0,math.floor((x-w/2)*iw));top=max(0,math.floor((y-h/2)*ih))
            right=min(iw,math.ceil((x+w/2)*iw));bottom=min(ih,math.ceil((y+h/2)*ih))
            a=max(0,((x-w/2)*iw-left)/(right-left));b=max(0,((y-h/2)*ih-top)/(bottom-top))
            c=min(1,((x+w/2)*iw-left)/(right-left));d=min(1,((y+h/2)*ih-top)/(bottom-top))
            im=im.crop((left,top,right,bottom))
            text=f'{int(cls)} {(a+c)/2:.8f} {(b+d)/2:.8f} {c-a:.8f} {d-b:.8f}\n'
            mode='approved_target_crop'
        im.save(output/'images'/path.name,quality=97)
        (output/'labels'/f'{stem}.txt').write_text(text)
        preview=ImageOps.contain(im,(700,700));draw=ImageDraw.Draw(preview)
        for line in text.splitlines():
            cls,x,y,w,h=map(float,line.split());pw,ph=preview.size
            assert int(cls) in (20,30,39) and 0<=x-w/2 and 0<=y-h/2
            assert x+w/2<=1+1e-7 and y+h/2<=1+1e-7
            rect=((x-w/2)*pw,(y-h/2)*ph,(x+w/2)*pw,(y+h/2)*ph)
            draw.rectangle(rect,outline='#00ff60',width=3)
            draw.text((rect[0]+3,rect[1]+3),names[int(cls)],fill='red',stroke_fill='white',stroke_width=1)
        preview.save(output/'previews'/path.name,quality=94);previews.append((path.name,preview))
        rows.append(dict(image='images/'+path.name,label='labels/'+stem+'.txt',
            source='custom',refrigerator_id='user_household_01',session_id='household_batch_20261103',
            group_id='household_batch_20261103',split='train',status='user_approved',annotation_mode=mode,
            original_image=r['original_image'],original_sha256=r['original_sha256']))
    assert len(rows)==22
    write_csv(output/'manifest.csv',rows)
    for start in range(0,len(previews),6):
        sheet=Image.new('RGB',(1200,1560),'white');draw=ImageDraw.Draw(sheet)
        for j,(name,im) in enumerate(previews[start:start+6]):
            im=ImageOps.contain(im,(580,480));x=j%2*600+(600-im.width)//2;y=j//2*520+30
            sheet.paste(im,(x,y));draw.text((j%2*600+8,j//2*520+8),name,fill='black')
        sheet.save(output/f'review-sheet-{start//6+1}.jpg',quality=94)
    (output/'class-names.yaml').write_text(yaml.safe_dump({'class_schema':'food63-no-spices-v1','names':names},sort_keys=False))
    report=dict(images=22,boxes=23,full_images=15,target_crops=7,split='train only, one household group',
        class_count=63,approved_by_user=True,training_started=False,
        integration_status='Prepared separately; combined dataset waits for saved beans decisions')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    (output/'README.md').write_text('Approved household targets using the 63-class no-spices schema.\n'
        'All images remain in one training group. Seven target crops avoid unresolved background-food labels.\n'
        'Original full photos and annotation drafts are preserved. No validation or unseen test claim.\n'
        'Prepared independently while approved beans review CSV is restored. No training started.\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
