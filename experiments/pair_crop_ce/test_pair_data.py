"""Pure manifest and compatibility-gate checks; never run inference."""
import json
from pathlib import Path
import unittest
import numpy as np
from pair_data import compare_features,validate_stage
from phrase_data import identity


class Checks(unittest.TestCase):
    def test_manifest_partition_and_schedule(self):
        parent=Path(__file__).parent;path=parent/'locked_manifest.json'
        if not path.exists():path=parent/'PAIR_MANIFEST.json'
        m=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(identity({k:v for k,v in m.items() if k!='manifest_content_sha256'}),m['manifest_content_sha256'])
        self.assertEqual([len(v) for v in m['partitions'].values()],[14,7,8]);self.assertEqual(m['cross_weight'],0.)
        self.assertEqual(len(m['schedule']),300);self.assertTrue(all(len(v)==8 for v in m['schedule']))
        self.assertEqual(set(sum(m['schedule'],[])),set(m['partitions']['fit']))
        by={g['group_id']:g for g in m['groups']};sets=[{e['image_id'] for k in ids for e in by[k]['events']} for ids in m['partitions'].values()]
        self.assertTrue(all(not sets[i]&sets[j] for i in range(3) for j in range(i)))

    def test_relative_error_gate_cannot_be_replaced_by_loose_allclose(self):
        old=np.ones((197,768));self.assertTrue(compare_features(old+.0005,old)['pass_gate'])
        self.assertFalse(compare_features(old+.002,old)['pass_gate'])

    def test_single_bad_patch_and_nonfinite_rejected(self):
        old=np.ones((197,768));new=old.copy();new[0,0]+=.03
        self.assertFalse(compare_features(new,old)['pass_gate'])
        new[0,0]=np.nan
        with self.assertRaises(AssertionError):compare_features(new,old)

    def test_stage_mapping_rejects_swapped_events_and_wrong_coordinates(self):
        import tempfile
        from crop_geometry import crop_rectangle,patch_centers
        events=[dict(event_id='a',image_id=1,width=100,height=100,subject_box=[0,0,.5,1],object_box=[.4,.2,.8,.8]),
                dict(event_id='b',image_id=1,width=100,height=100,subject_box=[.5,0,1,1],object_box=[.6,.5,.9,.9])]
        m={'groups':[{'events':events}]}
        values=dict(image_ids=np.array([1]),event_ids=np.array(['a','b']),full=np.zeros((1,197,768),np.float16),
            global_memory=np.zeros((1,32,512),np.float32),crop=np.zeros((2,196,768),np.float16),
            full_xy=np.asarray(patch_centers([0,0,1,1],1,1),np.float32),
            crop_xy=np.asarray([patch_centers(crop_rectangle(e['subject_box'],e['object_box'],100,100),100,100) for e in events],np.float32))
        with tempfile.TemporaryDirectory(prefix='pair_check_',dir=Path(__file__).parent) as d:
            path=Path(d)/'features.bin'
            def write():
                with path.open('wb') as f:np.savez(f,**values)
            write();validate_stage(path,m)
            values['event_ids']=np.array(['b','a']);write()
            with self.assertRaises(AssertionError):validate_stage(path,m)
            values['event_ids']=np.array(['a','b']);values['crop_xy'][0,0,0]+=.01;write()
            with self.assertRaises(AssertionError):validate_stage(path,m)


if __name__=='__main__':unittest.main(verbosity=2)
