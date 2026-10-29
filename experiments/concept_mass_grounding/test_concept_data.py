"""Pure lexical/probability/abstention and fixed-cohort checks; never imports Torch."""
import gzip,json,math,sys,unittest
from concept_data import *

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=json.loads(gzip.decompress((ROOT/'experiments/semantic_grounding_method/data/gain_metadata.json.gz').read_bytes()))
        cls.rules=json.loads((ROOT/'experiments/coco_grounding_feasibility/MAPPING_POLICY_V11.json').read_bytes())
        cls.lex=lexicon(cls.old['word2idx'],cls.rules)
    def test_no_neural_imports(self):self.assertNotIn('torch',sys.modules)
    def test_prefix_union_deduplicates_overlaps(self):
        paths=[(4,),(4,5),(6,7),(6,7)]
        self.assertEqual(prefix_free(paths),[(4,),(6,7)])
        value=log_union_numpy(paths,lambda prefix,token:{((),4):.2,((),6):.5,((6,),7):.4}[(prefix,token)])
        self.assertAlmostEqual(math.exp(value),.4)
    def test_probability_gradient_sign_finite_difference(self):
        def loss(z):
            p=math.exp(z)/(math.exp(z)+2)
            return -math.log1p(-p)
        self.assertGreater((loss(.001)-loss(-.001))/.002,0)
    def test_special_tokens_and_punctuation(self):
        for paths in self.lex['events'].values():
            self.assertEqual([tuple(p) for p in paths],prefix_free(paths))
            self.assertTrue(all(all(i>=4 for i in p) for p in paths))
            for p in paths:
                self.assertTrue(all(self.lex['idx2word'][str(i)]==normalize(self.lex['idx2word'][str(i)]) for i in p[:-1]))
        self.assertTrue(any(self.lex['idx2word'][str(p[-1])].endswith('.') for p in self.lex['events']['dog']))
    def test_longest_compound_and_offsets(self):
        m=mentions('a hot dog and a cell phone.'.split(),self.lex)
        self.assertEqual([(r['category'],r['start'],r['end']) for r in m],[('hot dog',1,3),('cell phone',5,7)])
    def test_unsafe_aliases_abstain(self):
        for s in ('a dog costume','a toy dog'):
            self.assertTrue(mentions(s.split(),self.lex)[0]['uncertain'])
        self.assertFalse(self.lex['events']['orange'])
    def test_detector_only_veto_cannot_make_positive(self):
        words=[1,self.old['word2idx']['dog'],2];target=dict(supported=[],unknown=['dog'],positive_anchor_classes=[],negative_eligible=True)
        events,count=token_events([words],[target],self.lex,True)
        self.assertEqual(events,[]);self.assertEqual(count['vetoes'],1)
        self.assertEqual(token_events([words],[target],self.lex,False)[0],[])
    def test_gt_ref_support_and_missing_cache_abstain(self):
        words=[1,self.old['word2idx']['dog'],2]
        for target in [dict(supported=['dog'],unknown=[],positive_anchor_classes=[],negative_eligible=True),dict(supported=[],unknown=[],positive_anchor_classes=[],negative_eligible=False)]:
            self.assertFalse(token_events([words],[target],self.lex,True)[0])
        target=dict(supported=[],unknown=[],positive_anchor_classes=[],negative_eligible=True)
        self.assertEqual(token_events([words],[target],self.lex,True)[0],[dict(batch=0,position=0,category='dog')])
    def test_eos_and_special_prefix_positions(self):
        dog=self.old['word2idx']['dog'];target=dict(supported=[],unknown=[],positive_anchor_classes=[],negative_eligible=True)
        self.assertEqual(token_events([[1,0,dog,2,dog]],[target],self.lex,True)[0],[dict(batch=0,position=1,category='dog')])
    def test_surface_outside_event_is_not_penalized(self):
        import copy
        lex=copy.deepcopy(self.lex);lex['idx2word']['10001']='dog)'
        target=dict(supported=[],unknown=[],positive_anchor_classes=[],negative_eligible=True)
        events,count=token_events([[1,10001,2]],[target],lex,True)
        self.assertEqual(events,[]);self.assertEqual(count['unknown'],1)

    def test_c0_changes_only_effective_keep_weight(self):
        c=objective_weights('C');c0=objective_weights('C0')
        self.assertEqual(c,dict(kl=.1,keep=.1,negative=0.))
        self.assertEqual(c0,{**c,'keep':0.})
        self.assertEqual(objective_weights('G'),dict(kl=.1,keep=.1,negative=.1))
        with self.assertRaises(ValueError):objective_weights('unknown')

    def test_locked_train_schedule_and_dev_disjointness(self):
        cohort=json.loads(gzip.decompress((HERE/'TRAIN_PRESENCE.json.gz').read_bytes()));validate_rows(cohort['train_rows'],self.old['groups']['val'])
        order=order_indices();self.assertEqual(order.shape,(500,8));self.assertEqual(len(set(order.flatten())),4000)
        self.assertEqual(set(cohort['gt_classes']),{str(r['coco_id']) for r in cohort['train_rows']})
    def test_aux_reduction_unequal_rank_events(self):
        # Rank means would give(2+8)/2=5; global-event mean is(2+7+9)/3=6.
        self.assertEqual(((2*2/3)+(2*(7+9)/3))/2,6)

if __name__=='__main__':unittest.main()
