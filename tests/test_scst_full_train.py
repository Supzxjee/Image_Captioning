import unittest
import numpy as np
from captioning.scst import TrainingCider
from run_scst_full_train import full_groups, select_checkpoint


class LazyRewardTests(unittest.TestCase):
    def test_limited_cache_exactly_preserves_eager_scores_after_eviction(self):
        refs={i:[s]*5 for i,s in enumerate(['person riding bicycle','dog on grass','person near sink'])}
        eager=TrainingCider(refs);lazy=TrainingCider(refs,reference_cache_size=1)
        for ids,text in [([0,1],['person on bicycle','dog on grass']),([2,0],['sink','person riding bicycle'])]:
            np.testing.assert_array_equal(eager.score(ids,text),lazy.score(ids,text))
            self.assertLessEqual(len(lazy.vectors),1)
        self.assertEqual(eager.df,lazy.df)

    def test_missing_id_and_invalid_cache_fail(self):
        refs={i:['dog']*5 for i in range(2)}
        with self.assertRaises(ValueError):TrainingCider(refs,0)
        with self.assertRaises(KeyError):TrainingCider(refs,1).score([99],['dog'])


class FullProtocolTests(unittest.TestCase):
    def test_all_train_restval_and_only_val_for_selection_evaluation(self):
        rows=[{'split':split,'filename':f'COCO_val2014_{i:012d}.jpg','filepath':'val2014'}
              for i,split in enumerate(['train','restval','train']+['val']*5000+['test']*5)]
        groups=full_groups({'images':rows})
        self.assertEqual({k:len(v) for k,v in groups.items()},{'train':3,'val':1000,'validation_remainder':4000})
        self.assertEqual({r['image_id'] for r in groups['train']},{0,1,2})
        self.assertEqual(groups,full_groups({'images':rows}))
        self.assertFalse({r['image_id'] for r in groups['val']}&{r['image_id'] for r in groups['validation_remainder']})

    def test_incomplete_validation_rejected(self):
        with self.assertRaises(ValueError):full_groups({'images':[]})

    def metric(self,cider=1.,chairs=.04,chairi=.03,recall=.4):
        return {'caption':{'CIDEr':cider},'chair':{'CHAIRs':chairs,'CHAIRi':chairi,'Recall':recall}}

    def test_higher_cider_unsafe_candidate_cannot_win(self):
        metrics={'original_fixed':self.metric(),'mid_fixed':self.metric(1.02),'end_fixed':self.metric(1.2,chairs=.05)}
        self.assertEqual(select_checkpoint(metrics,['mid','end']),('mid',['mid']))

    def test_tie_keeps_earlier_and_baseline_fallback(self):
        m={'original_fixed':self.metric(),'mid_fixed':self.metric(),'end_fixed':self.metric(.99)}
        self.assertEqual(select_checkpoint(m,['mid','end'])[0],'original')
        m['mid_fixed']=self.metric(1.02);m['end_fixed']=self.metric(1.02)
        self.assertEqual(select_checkpoint(m,['mid','end'])[0],'mid')

    def test_recall_guard_and_both_hallucination_metrics_enforced(self):
        m={'original_fixed':self.metric(),'mid_fixed':self.metric(1.1,chairi=.031),'end_fixed':self.metric(1.2,recall=.38)}
        self.assertEqual(select_checkpoint(m,['mid','end']),('original',[]))


if __name__=='__main__':unittest.main()
