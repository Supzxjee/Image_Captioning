"""Metadata/NumPy/static checks only. CUDA/gradient/beam guards run on Kaggle."""
import ast
from pathlib import Path
import unittest
import numpy as np
from full_xe_data import TRAIN_EVENTS,STEPS,EPOCHS,learning_rate,epoch_order,validate_state,validate_predictions,is_best
from sharing_data import rank_events,ddp_loss_multiplier

HERE=Path(__file__).parent

class Checks(unittest.TestCase):
    def state(self,epoch=1,cursor=0,phase='train'):
        return dict(epoch=epoch,batch_cursor=cursor,phase=phase,updates=(epoch-1)*STEPS+cursor,
            caption_events=(epoch-1)*TRAIN_EVENTS+min(cursor*32,TRAIN_EVENTS),history=[dict(epoch=i) for i in range(1,epoch)])

    def test_full_schedule_and_warmup_use_full_epochs(self):
        self.assertEqual((STEPS,EPOCHS*STEPS),(17711,177110));self.assertEqual(EPOCHS*TRAIN_EVENTS,5667470)
        self.assertEqual(learning_rate(1),8e-6)
        self.assertAlmostEqual(learning_rate(53133),1e-4);self.assertAlmostEqual(learning_rate(177110),1e-4)
        self.assertLess(learning_rate(1250),learning_rate(17711))
        for step in (0,177111):
            with self.assertRaises(ValueError):learning_rate(step)

    def test_epoch_covers_every_reference_once_tail_and_resume(self):
        counts=np.full(113287,5,dtype=np.int64);counts[:312]+=1
        images=np.repeat(np.arange(113287,dtype=np.int64),counts)
        data=dict(caption_image=images,group=np.zeros(113287,np.uint8))
        order=epoch_order(data,1);np.testing.assert_array_equal(np.sort(order),np.arange(TRAIN_EVENTS))
        np.testing.assert_array_equal(order,epoch_order(data,1));self.assertFalse(np.array_equal(order,epoch_order(data,2)))
        self.assertEqual(len(order[-27:]),27)
        tail=order[(STEPS-1)*32:];self.assertEqual((len(rank_events(tail,0)),len(rank_events(tail,1))),(16,11))
        cursor=1000;remaining=order[cursor*32:];np.testing.assert_array_equal(np.r_[order[:cursor*32],remaining],order)

    def test_resume_cursor_matches_updates_exposure_and_history(self):
        for state in (self.state(),self.state(2,1000),self.state(4,STEPS,'validate'),self.state(11)):
            validate_state(state)
        for field,value in [('updates',999),('caption_events',10),('history',[dict(epoch=2)]),('phase','bad')]:
            state=self.state(2,1000);state[field]=value
            with self.assertRaises(ValueError):validate_state(state)
        with self.assertRaises(ValueError):validate_state(self.state(1,1000,'validate'))
        with self.assertRaises(ValueError):validate_state(self.state(11,1))

    def test_validation_coverage_and_earlier_ties(self):
        refs=[dict(coco_id=1),dict(coco_id=2)]
        rows=[dict(coco_id=i,caption='',score=-1.) for i in (1,2)]
        validate_predictions(rows,refs)
        for bad in (rows[:1],rows+[rows[0]],[rows[0],rows[0]],[rows[0],dict(coco_id=3,caption='x',score=0.)]):
            with self.assertRaises(ValueError):validate_predictions(bad,refs)
        self.assertTrue(is_best(0,None));self.assertTrue(is_best(1.1,1));self.assertFalse(is_best(1,1));self.assertFalse(is_best(.9,1))
        with self.assertRaises(ValueError):is_best(float('nan'),1)

    def test_ddp_global_token_mean_not_mean_of_rank_means(self):
        tokens=np.array([32,11]);grad=np.array([100.,3.])
        actual=np.mean(grad*ddp_loss_multiplier(2,int(tokens.sum())))
        self.assertAlmostEqual(actual,grad.sum()/tokens.sum());self.assertNotAlmostEqual(actual,np.mean(grad/tokens))

    def test_runtime_guards_and_model_compile_without_execution(self):
        for name in ('model.py','full_xe_data.py','full_xe_decode.py','run_full_xe.py'):
            ast.parse((HERE/name).read_text(encoding='utf-8'))
        source=(HERE/'run_full_xe.py').read_text(encoding='utf-8')
        self.assertIn("state['phase']='validate';checkpoint()",source)
        self.assertIn('rng_by_rank=rngs',source);self.assertIn("torch.cuda.set_rng_state(rng['cuda'],device)",source)
        self.assertIn('beam_parity(base,x,valid)',source);self.assertIn('score,_=Cider().compute_score(refs_ptb,hyps)',source)
        self.assertNotIn('shared_collate',source);self.assertNotIn('autocast',source)
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                text=ast.unparse(node);self.assertNotIn('import torch',text)
        decode=(HERE/'full_xe_decode.py').read_text(encoding='utf-8')
        self.assertIn('logp[:,:,:2]=-torch.inf',decode);self.assertIn('finished.gather',decode)
        self.assertIn('beam_search(model,vo,vr,valid)',decode)

if __name__=='__main__':unittest.main(verbosity=2)
