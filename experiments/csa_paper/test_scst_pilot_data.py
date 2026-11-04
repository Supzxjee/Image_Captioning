"""Author-safe NumPy/static checks. Does not import Torch or any model."""
import ast
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
from scst_pilot_data import schedule,event_schedule,exact_checkpoint


class Checks(unittest.TestCase):
    def test_schedule_identity_unique_and_budget(self):
        pool,ids=schedule(range(113287))
        _,again=schedule(reversed(range(113287)))
        self.assertEqual(ids.shape,(500,8));self.assertEqual(len(set(ids.ravel())),4000)
        self.assertEqual(len(pool),10000);np.testing.assert_array_equal(ids,again)
        self.assertTrue(set(ids.ravel())<=set(pool))
        with self.assertRaises(ValueError):schedule([1,1],pool=2,updates=1,batch=2)

    def test_heldout_filter_and_reference_join(self):
        data=dict(ids=np.array([10,20,30]),group=np.array([0,0,1]),caption_image=np.array([0,0,1,1,1,2]))
        ids=np.array([[20,10],[10,20]])
        events=event_schedule(data,ids)
        np.testing.assert_array_equal(data['ids'][data['caption_image'][events]],ids)
        with self.assertRaises(ValueError):event_schedule(data,np.array([[30]]))

    def test_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'best.bin').write_bytes(b'wrong')
            with self.assertRaises(ValueError):exact_checkpoint(d)

    def test_existing_object_reward_count_and_neutral(self):
        root=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root))
        from captioning.object_reward import TrainingObjectReward
        scorer=TrainingObjectReward([dict(image_id=10,supported=['person','bicycle'],salient=['bicycle'],detector_neutral=['car'])],
            parser_version='aliases_v2',penalty_version='unique_count_v2')
        values,details=scorer.score([10,10,10],['a person on a bicycle','a person with a horse','a person near a car'])
        np.testing.assert_allclose(values,[.5,-1,0])
        self.assertEqual(details[1]['unsupported_classes'],['horse'])
        self.assertEqual(details[2]['neutral_classes'],['car'])

    def test_all_delivered_python_parse(self):
        here=Path(__file__).parent
        for name in ('scst_pilot_data.py','scst_pilot_model.py','run_scst_pilot.py','test_scst_pilot_model.py','make_scst_pilot_notebooks.py'):
            ast.parse((here/name).read_text(encoding='utf-8'),filename=name)


if __name__=='__main__':unittest.main()
