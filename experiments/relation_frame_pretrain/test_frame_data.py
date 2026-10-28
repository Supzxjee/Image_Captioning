"""Metadata/NumPy checks only. No Torch imports, including toy models."""
import unittest
import numpy as np
from frame_data import areas, geometry, lexical_targets, schedule, FILES
from compare_frames import compare


class Checks(unittest.TestCase):
    def test_spans_and_multiword_entities(self):
        for relation,is_word in [('next to','is '),('contains','')]:
            q=f'the traffic light {is_word}{relation} the sports ball'
            vocab={w:i+4 for i,w in enumerate(sorted(set(q.split())))}
            vocab.update({'<bos>':1,'<eos>':2})
            p=dict(statement=q,relation=relation,positive=dict(subject='traffic light',object='sports ball'))
            ids,w=lexical_targets(p,vocab)
            self.assertEqual(len(ids)-1,len(w));self.assertAlmostEqual(float(w.sum()),1,places=6)
            words=q.split()
            self.assertEqual([x for x,v in zip(words,w) if v>0],['traffic','light']+relation.split()+['sports','ball'])
            for x,v in zip(words,w):
                if x in ('the','is'):self.assertEqual(v,0)
            del vocab['ball']
            with self.assertRaises(ValueError):lexical_targets(p,vocab)

    def test_fractional_roi_preserves_tiny_boxes_and_area(self):
        for b in ([0,0,1,1],[.0001,.01,.0002,.011],[.13,.27,.88,.91]):
            a=areas(b);self.assertGreater(a.sum(),0)
            self.assertAlmostEqual(float(a.sum()),196*(b[2]-b[0])*(b[3]-b[1]),places=5)
        np.testing.assert_array_equal(areas([0,0,1,1]),np.ones(196,dtype='float32'))
        with self.assertRaises(ValueError):areas([.5,.5,.4,.7])

    def test_ordered_geometry_and_fixed_global_batches(self):
        s=np.array([.1,.1,.3,.3]);o=np.array([.5,.5,.7,.7])
        self.assertGreater(geometry(s,o)[0],0);self.assertLess(geometry(o,s)[0],0)
        self.assertEqual(set(np.concatenate([schedule(i) for i in range(16)])),set(range(256)))
        for i in range(200):
            self.assertEqual(len(set(schedule(i))),16)
            self.assertFalse(set(schedule(i)[:8])&set(schedule(i)[8:]))
            np.testing.assert_array_equal(schedule(i),schedule(i))

    def test_gate_requires_improvement_and_complete_pairing(self):
        protocol=dict(seed=42,updates=200,device_gpu_hours_cap=.5,source_commit='fixed',metadata_sha256='fixed',cache_sha256=FILES,checkpoint_sha256='fixed')
        def result(arm,gap):
            return dict(status='ENDPOINT_200_COMPLETE',updates=200,arm=arm,initial_model_digest='same',protocol=protocol,
                rows=[dict(pair_id=str(i),relation=0,stratum='contact_proximity' if i<32 else 'projective_containment',likelihood_gap=gap) for i in range(64)])
        p=result('RelPos',-1);c=result('RelContrast',1)
        self.assertEqual(compare(p,c)['status'],'PRETRAIN_GATE_PASS')
        self.assertEqual(compare(result('RelPos',1),c)['status'],'PRETRAIN_GATE_NO_GO')
        c['rows'].pop()
        with self.assertRaises(ValueError):compare(p,c)


if __name__=='__main__':unittest.main(verbosity=2)
