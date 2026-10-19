"""Pure-data preflight, no model/notebook execution."""
import json
from pathlib import Path
import unittest
import numpy as np
from oracle_data import select_cases,oracle_slots,summarize
from probe_data import identity


class DataChecks(unittest.TestCase):
    def test_locked_cases_and_controlled_intervention(self):
        m=json.loads((Path(__file__).parent/'cases_manifest.json').read_text(encoding='utf-8'));rows=m['cases']
        self.assertEqual(len(rows),20);self.assertEqual(len({r['coco_id'] for r in rows}),20)
        self.assertEqual(identity(rows),m['cases_sha256'])
        for row in rows:
            self.assertEqual(row['task'],'wrong_predicate');self.assertIn(row['caption_split'],('train','restval'))
            positive,negative=oracle_slots(row,True),oracle_slots(row,False)
            for field in ('bounds','roles','valid'):np.testing.assert_array_equal(positive[field],negative[field])
            self.assertEqual([i for i,(p,n) in enumerate(zip(positive['texts'],negative['texts'])) if p!=n],[2])
            self.assertEqual(positive['valid'].sum(),3)

    def test_selection_excludes_correct_and_duplicate_images(self):
        row=json.loads((Path(__file__).parent/'cases_manifest.json').read_text(encoding='utf-8'))['cases'][0]
        extra=dict(row,case_id='another_case_same_image')
        selected=select_cases({'calibration':[row,extra]},size=1)
        self.assertEqual(len(selected),1)
        changed=dict(row,scores={'boxed':[1.,0.] if row['positive_index']==0 else [0.,1.]})
        with self.assertRaises(ValueError):select_cases({'calibration':[changed]},size=1)

    def test_summary_measures_sensitivity_not_semantic_accuracy(self):
        records=[]
        for i in range(20):
            conditions={k:dict(caption='unchanged',positive_minus_negative_logp=float(v),positive_minus_negative_mean_logp=float(v)/4) for k,v in [('off',0),('upt',0),('annotation_positive',2),('annotation_negative',-1)]}
            records.append(dict(case_id=str(i),conditions=conditions,evidence_change=dict(R_attention_L1=.2,R_token_relative_L2=.1)))
        result=summarize(records)['controlled_annotation_positive_vs_negative']
        self.assertEqual(result['mean_preference_shift'],3.)
        self.assertEqual(result['captions_changed'],0)


if __name__=='__main__':unittest.main(verbosity=2)
