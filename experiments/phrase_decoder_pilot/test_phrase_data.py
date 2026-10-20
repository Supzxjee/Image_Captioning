import copy
import json
from pathlib import Path
import unittest
import numpy as np
from phrase_data import build_manifest,identity,geometry

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_locked_split_schedule_and_fit_only_stats(self):
        m=json.loads((HERE/'locked_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(identity({k:v for k,v in m.items() if k!='manifest_content_sha256'}),m['manifest_content_sha256'])
        by={g['group_id']:g for g in m['groups']};fit=m['partitions']['fit'];held=m['partitions']['adaptation_holdout']
        self.assertEqual((len(fit),len(held)),(14,7));self.assertFalse(set(fit)&set(held))
        ids=lambda keys:{e['image_id'] for k in keys for e in by[k]['events']}
        self.assertFalse(ids(fit)&ids(held));self.assertEqual(len(m['schedule']),300)
        self.assertTrue(all(len(batch)==8 and set(batch)<=set(fit) for batch in m['schedule']))
        self.assertEqual(identity(m['schedule']),m['schedule_sha256'])
        raw=np.stack([geometry(e['subject_box'],e['object_box']) for k in fit for e in by[k]['events']])
        np.testing.assert_allclose(raw.mean(0),m['geometry_fit_mean'],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(np.maximum(raw.std(0),1e-3),m['geometry_fit_std'],rtol=1e-6,atol=1e-6)
        self.assertEqual(sum(by[k]['same_image'] for k in fit),2)
        self.assertEqual(sum(by[k]['same_image'] for k in held),2)
        self.assertTrue(all(by[k]['family']!='bicycle_posture' for k in held))
        for g in by.values():
            self.assertEqual(set(g['positive_indices']),{0,1})
            for e in g['events']:
                self.assertIn(e['caption_split'],('train','restval'))
                self.assertFalse(set(e)&{'source_hint','references','upt_selected_objects'})

    def test_manifest_matches_parent_human_targets_if_available(self):
        parent=HERE.parent/'phrase_pair_feasibility/FEASIBLE_GROUPS_21.json'
        if not parent.exists():self.skipTest('Only embedded manifest available')
        bundle=json.loads(parent.read_text(encoding='utf-8'));built=build_manifest(bundle)
        actual=json.loads((HERE/'locked_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(built,{k:v for k,v in actual.items() if k!='manifest_content_sha256'})

    def test_geometry_finite_and_subject_object_order_matters(self):
        a=[0,0,.3,.5];b=[.5,.4,.9,.9]
        self.assertTrue(np.isfinite(geometry(a,b)).all());self.assertFalse(np.allclose(geometry(a,b),geometry(b,a)))


if __name__=='__main__':unittest.main(verbosity=2)
