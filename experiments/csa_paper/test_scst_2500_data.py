"""Safe NumPy/static follow-up checks. Never imports Torch or runs notebooks."""
from collections import Counter
import ast
from pathlib import Path
import unittest
import numpy as np
from scst_pilot_data import schedule,event_schedule,study_updates


class Checks(unittest.TestCase):
    def test_2500_full_pool_two_passes_and_prefix(self):
        pool,ids=schedule(range(113287),updates=2500)
        oldpool,old=schedule(range(113287))
        self.assertEqual(pool,oldpool);self.assertEqual(ids.shape,(2500,8))
        np.testing.assert_array_equal(ids[:500],old)
        np.testing.assert_array_equal(old,np.random.default_rng(42).permutation(pool)[:4000].reshape(500,8))
        self.assertEqual(Counter(Counter(ids.ravel()).values()),{2:10000})
        self.assertEqual(set(ids[:1250].ravel()),set(pool))
        self.assertEqual(set(ids[1250:].ravel()),set(pool))
        _,again=schedule(reversed(range(113287)),updates=2500)
        np.testing.assert_array_equal(ids,again)

    def test_reference_rotation_and_prefix(self):
        pool,ids=schedule(range(113287),updates=2500)
        data=dict(ids=np.array(pool+[999999]),group=np.array([0]*10000+[1]),
            caption_image=np.repeat(np.arange(10001),5))
        events=event_schedule(data,ids);previous=event_schedule(data,ids[:500])
        np.testing.assert_array_equal(events[:500],previous)
        index={i:j for j,i in enumerate(pool)}
        legacy=np.array([[index[int(i)]*5+(step+j)%5 for j,i in enumerate(row)] for step,row in enumerate(ids[:500])])
        np.testing.assert_array_equal(previous,legacy)
        np.testing.assert_array_equal(data['ids'][data['caption_image'][events]],ids)
        first={int(i):int(e)%5 for i,e in zip(ids[:1250].ravel(),events[:1250].ravel())}
        for i,e in zip(ids[1250:].ravel(),events[1250:].ravel()):self.assertEqual(int(e)%5,(first[int(i)]+1)%5)
        with self.assertRaises(ValueError):event_schedule(data,np.array([[999999]]))

    def test_horizon_guard(self):
        self.assertEqual(study_updates({}),500);self.assertEqual(study_updates({'updates':2500}),2500)
        for value in (True,0,501,3000,'2500',2500.0):
            with self.assertRaises(ValueError):study_updates(dict(updates=value))

    def test_generator_and_workers_have_no_500_endpoint(self):
        here=Path(__file__).parent
        worker=(here/'run_scst_pilot.py').read_text(encoding='utf-8')
        for stale in ('range(initial,500)',"w['updates']==500",'image_events=4000','updates=500,global_batch'):
            self.assertNotIn(stale,worker)
        for name in ('run_scst_pilot.py','make_scst_pilot_notebooks.py','scst_pilot_data.py'):
            ast.parse((here/name).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
