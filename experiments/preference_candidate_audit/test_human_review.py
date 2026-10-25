import copy
import unittest
from audit_human_review import summarize


class Checks(unittest.TestCase):
    def setUp(self):
        self.key={str(i):'X' for i in range(30)}
        self.review=dict(schema=1,manifest_sha256='fixed',reviewer='Tester',reviewer_type='human_verified_ai_assisted',human_verified=True,
            annotations=dict(self.key),verified={i:True for i in self.key})

    def test_locked_threshold_and_ties(self):
        for i in range(3):self.review['annotations'][str(i)]='tie'
        self.assertTrue(summarize(self.review,self.key,'fixed',self.key)['agreement_gate_pass'])
        self.review['annotations']['3']='unclear'
        result=summarize(self.review,self.key,'fixed',self.key)
        self.assertFalse(result['agreement_gate_pass']);self.assertEqual(result['agreement_fraction'],26/30)
        self.assertEqual(result['ties'],3);self.assertEqual(result['unclear'],1)

    def test_reject_missing_wrong_or_unverified(self):
        for mutate in (lambda r:r.update(manifest_sha256='wrong'),lambda r:r['annotations'].pop('0'),
                       lambda r:r['verified'].update({'0':False}),lambda r:r['annotations'].update({'0':''}),
                       lambda r:r.update(reviewer_type='ai_preannotation')):
            r=copy.deepcopy(self.review);mutate(r)
            with self.assertRaises(ValueError):summarize(r,self.key,'fixed',self.key)

    def test_changes_preserve_human_labels(self):
        self.review['annotations']['0']='Y'
        result=summarize(self.review,self.key,'fixed',self.key)
        self.assertEqual(result['opposes_selected_preference'],1)
        self.assertEqual(result['changes_from_AI'],[dict(coco_id=0,AI='X',human='Y')])
        self.assertEqual(self.review['annotations']['0'],'Y')


if __name__=='__main__':unittest.main(verbosity=2)
