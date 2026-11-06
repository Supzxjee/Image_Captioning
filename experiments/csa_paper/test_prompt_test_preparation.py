"""Metadata-only preparation fixtures, no model or scoring execution."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from prepare_prompt_test import map_records


class Checks(unittest.TestCase):
    def fixture(self):
        row=dict(coco_id=10,filename='a.jpg',group='test',source_split='test',width=100,height=100,region_kinds=['predicted_roi'],captions=['secret reference'],sequences=[[1,2]])
        arrays=dict(ids=np.array([10]),offsets=np.array([0,1]),boxes=np.array([[0,0,100,100]]))
        return row,arrays

    def test_prompt_payload_never_copies_references(self):
        row,a=self.fixture()
        with patch('prepare_prompt_test.detector_regions',return_value=[]),patch('prepare_prompt_test.join_entry',return_value=dict(objects=[],relations=[])):
            result=map_records([row],a,{'a.jpg':{}},{'a.jpg':[]})[0]
        self.assertEqual(result['group'],'test');self.assertEqual(result['roi_count'],1)
        self.assertNotIn('captions',result);self.assertNotIn('sequences',result)

    def test_missing_source_wrong_split_or_order_fails(self):
        row,a=self.fixture()
        with self.assertRaises(ValueError):map_records([row],a,{}, {})
        row['group']='val'
        with self.assertRaises(ValueError):map_records([row],a,{'a.jpg':{}},{'a.jpg':[]})
        row['group']='test';a['ids'][0]=11
        with self.assertRaises(ValueError):map_records([row],a,{'a.jpg':{}},{'a.jpg':[]})

    def test_sources_parse(self):
        for n in ('prepare_prompt_test.py','make_prompt_test_preparation.py'):
            ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
