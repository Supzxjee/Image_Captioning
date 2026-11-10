"""Author-safe full grouped sampler/collation/cursor/integration checks."""
import ast
from pathlib import Path
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import csa_full_data as d
import full_data as base

HERE=Path(__file__).resolve().parent


class Reader:
    def __init__(self,root):
        images=np.repeat(np.arange(7),[5,4,6,3,5,2,2])
        seq=np.array([[1,4+i%4,2,0] for i in range(27)],np.int32)
        self.data=dict(caption_image=images,sequences=seq);self.read=[]
    def get(self,i):self.read.append(int(i));return np.full((196,768),i,np.float16)


class Checks(TestCase):
    def test_all_full_events_once_deterministic_and_tail(self):
        # Synthetic unequal group sizes with the same fullTRAIN event total.
        mapping=np.repeat(np.arange(base.TRAIN_IMAGES),np.r_[np.full(312,6),np.full(base.TRAIN_IMAGES-312,5)])
        self.assertEqual(len(mapping),base.TRAIN_EVENTS)
        order=d.grouped_order(mapping,len(mapping),base.TRAIN_IMAGES,1)
        np.testing.assert_array_equal(np.sort(order),np.arange(base.TRAIN_EVENTS))
        np.testing.assert_array_equal(order,d.grouped_order(mapping,len(mapping),base.TRAIN_IMAGES,1))
        self.assertFalse(np.array_equal(order,d.grouped_order(mapping,len(mapping),base.TRAIN_IMAGES,2)))
        self.assertEqual(len(order[-(len(order)%32):]),27)
        with self.assertRaises(ValueError):d.grouped_order(np.r_[mapping[:-1],base.TRAIN_IMAGES],len(mapping),base.TRAIN_IMAGES,1)

    def test_unique_collation_caption_order_tail_and_cursor(self):
        for rank,expected in ((0,16),(1,11)):
            with patch.object(base,'FeatureReader',Reader):
                ds=d.SharedCaptionBatches('unused',np.arange(27),rank=rank)
                features,seq,events=ds[0]
                self.assertEqual(len(seq),expected);self.assertEqual(len(seq),len(events))
                self.assertEqual(len(features),len(set(ds.reader.data['caption_image'][events])))
                self.assertEqual(len(ds.reader.read),len(features))
                np.testing.assert_array_equal(seq[:,1],ds.reader.data['sequences'][events,1])
        with patch.object(base,'FeatureReader',Reader):
            ds=d.SharedCaptionBatches('unused',np.arange(27),cursor=1,batch_size=16)
            self.assertEqual(ds[0][2].tolist(),list(range(16,27)))

    def test_horizon_and_progressive_epoch_budget(self):
        self.assertEqual(base.STEPS,17711)
        self.assertAlmostEqual(base.learning_rate(1),8e-6)
        self.assertAlmostEqual(base.learning_rate(10*base.STEPS),1e-5)
        state=dict(phase='train',batch_cursor=base.STEPS-100)
        estimate=d.current_epoch_projection(state,.2,200)
        self.assertAlmostEqual(estimate,1.35*(100*.2+200+90)+900)
        state.update(phase='validate',batch_cursor=base.STEPS)
        self.assertAlmostEqual(d.current_epoch_projection(state,.2,200),1.35*(200+90)+900)
        state.update(epoch=11,phase='train',batch_cursor=0)
        self.assertEqual(d.current_epoch_projection(state,.2,200),900)

    def test_runner_static_accelerate_sharing_resume_no_legacy_model(self):
        source=(HERE/'run_csa_full.py').read_text(encoding='utf-8');tree=ast.parse(source)
        for value in ('accelerator.prepare_model(model)','accelerator.prepare_optimizer','accelerator.backward(loss)',
                      'image_plan(data[\'caption_image\'],events,True)','current_epoch_projection(state,measured,val_est)',
                      "optimizer.load_state_dict(ck['optimizer'])","rng_by_rank",'mean_words','short_le2'):
            self.assertIn(value,source)
        self.assertNotIn('DDP(',source);self.assertNotIn('remaining=(10*STEPS',source)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Accelerator')
        self.assertEqual(ast.literal_eval(next(k.value for k in call.keywords if k.arg=='mixed_precision')),'no')
        ast.parse((HERE/'csa_full_runtime.py').read_text(encoding='utf-8'))
        ast.parse((HERE/'score_csa_full.py').read_text(encoding='utf-8'))


if __name__=='__main__':main(verbosity=2)
