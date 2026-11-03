import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ocr_pipeline import match_label, fuse, grouped_labels

class FusionTests(unittest.TestCase):
    def line(self, text, box, score=.98):
        return {'text':text,'xyxy':box,'score':score,'matched_class':match_label(text)}

    def test_split_kidney_beans_and_duplicate_crops(self):
        evidence=[self.line('KIDNEY',[2273,942,2364,972]),
                  self.line('BEANS',[2278,958,2356,983],.99),
                  self.line('BEANS',[2278,959,2357,983],.97)]
        groups=grouped_labels(evidence)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['text'],'kidney beans')
        self.assertEqual(groups[0]['matched_class'],'beans')
        self.assertEqual(groups[0]['score'],.98)
        self.assertEqual(len(groups[0]['source_lines']),2)
        plus,_=fuse([],groups,['beans'])
        self.assertEqual(plus[0]['class_name'],'beans')

    def test_do_not_join_separate_products_or_distant_lines(self):
        for box in ([250,24,340,44],[10,180,100,200],[110,10,200,30]):
            self.assertEqual(grouped_labels([self.line('KIDNEY',[10,10,100,30]),
                                            self.line('BEANS',box)]),[])

    def test_reject_weak_or_ingredient_list_groups(self):
        first=self.line('KIDNEY',[10,20,100,40])
        last=self.line('BEANS',[10,42,100,62])
        self.assertEqual(grouped_labels([first,dict(last,score=.7)]),[])
        self.assertEqual(grouped_labels([self.line('INGREDIENTS',[10,0,120,15]),first,last]),[])

    def test_ingredient_list_not_product(self):
        self.assertIsNone(match_label('ingredients: milk butter salt'))
        self.assertIsNone(match_label('butter milk'))
        self.assertEqual(match_label('PAPRIKA'), 'spices')

    def test_spatial_correction_and_missed_product(self):
        names=['butter','spices','spinach']
        detections=[{'class_name':'butter','class_id':0,'xyxy':[10,10,100,100],'confidence':.6}]
        evidence=[{'matched_class':'spices','score':.99,'text':'PAPRIKA','xyxy':[20,20,70,40]},
                  {'matched_class':'spinach','score':.95,'text':'SPINACH','xyxy':[200,200,300,220]}]
        plus,actions=fuse(detections,evidence,names)
        self.assertEqual(detections[0]['class_name'],'butter')
        self.assertEqual(plus[0]['class_name'],'spices')
        self.assertEqual(plus[0]['yolo_class_name'],'butter')
        self.assertEqual(plus[0]['yolo_confidence'],.6)
        self.assertEqual(plus[0]['confidence_kind'],'ocr_recognition')
        self.assertEqual(plus[0]['confidence'],.99)
        self.assertEqual(plus[1]['box_kind'],'text_region')
        self.assertEqual(actions[0]['action'],'corrected')

    def test_low_confidence_and_ambiguous_location(self):
        detections=[{'class_name':'butter','class_id':0,'xyxy':[0,0,100,100],'confidence':.6}]*2
        evidence=[{'matched_class':'spices','score':.99,'text':'paprika','xyxy':[20,20,30,30]}]
        plus,actions=fuse(detections,evidence,['butter','spices'])
        self.assertEqual(actions,[])
        self.assertEqual(len(plus),2)

if __name__=='__main__':
    unittest.main()
