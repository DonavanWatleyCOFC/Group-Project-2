import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_food63 import remap
from validate_dataset import ROOT,validate
from ocr_pipeline import fuse,match_label

class Food63Tests(unittest.TestCase):
    def setUp(self):
        policy=json.loads((ROOT/'config/classes.json').read_text())
        self.old=policy['baseline_class_names']+policy['allowed_appended_class_names']
        self.new=[n for n in self.old if n!='spices']

    def test_removal_and_shift(self):
        result=remap('20 .5 .5 .2 .2\n40 .5 .5 .2 .2\n41 .5 .5 .2 .2\n63 .5 .5 .2 .2\n',self.old,self.new)
        self.assertEqual([int(line.split()[0]) for line in result.splitlines()],[20,40,62])
        self.assertEqual(self.new[40],'tomato_sauce')
        self.assertEqual(self.new[62],'yogurt')

    def test_relabel_only_beans_preserves_geometry(self):
        self.assertEqual(remap('30 .4 .6 .2 .3\n20 .5 .5 .2 .2\n',self.old,self.new,'relabel'),
                         '15 .4 .6 .2 .3\n20 .5 .5 .2 .2\n')

    def test_ocr_resolves_ids_from_active_names(self):
        self.assertEqual(match_label('cut green beans'),'green_beans')
        self.assertEqual(match_label('white beans'),'beans')
        evidence=[dict(matched_class='spices',score=.99,text='paprika',xyxy=[0,0,50,20]),
                  dict(matched_class='tomato_sauce',score=.99,text='tomato sauce',xyxy=[100,0,150,20])]
        output,_=fuse([],evidence,self.new)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]['class_id'],40)
        self.assertEqual(output[0]['class_name'],'tomato_sauce')

    def test_explicit_schema_and_legacy_both_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg={'names':self.new,'nc':63,'class_schema':'food63-no-spices-v1'}
            for i,split in enumerate(('train','val','test')):
                (root/split/'images').mkdir(parents=True);(root/split/'labels').mkdir()
                (root/split/'images/a.jpg').write_bytes(bytes([i]))
                (root/split/'labels/a.txt').write_text('62 .5 .5 .2 .2\n')
                cfg[split]=split+'/images'
            data=root/'data.yaml';data.write_text(json.dumps(cfg))
            self.assertEqual(len(validate(data,verify_images=False)['names']),63)
            cfg.pop('class_schema');data.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'ordered prefix'):validate(data,verify_images=False)
            cfg['names']=self.old;cfg['nc']=64;data.write_text(json.dumps(cfg))
            self.assertEqual(len(validate(data,verify_images=False)['names']),64)
            cfg['class_schema']='invented';data.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Unknown class schema'):validate(data,verify_images=False)

if __name__=='__main__':unittest.main()
