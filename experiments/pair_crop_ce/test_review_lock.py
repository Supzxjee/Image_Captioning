import copy
import unittest
from lock_human_review import classify


class Checks(unittest.TestCase):
    def setUp(self):
        self.g={'events':[{'image_id':123}]}
        self.a=dict(confirmed_by_human=True,truth=[['supported','contradicted'],['contradicted','supported']],
                    boxes=[['clear','clear'],['clear','clear']],detail_224=['no','unclear'],review_flags=['ambiguous_carrying'])

    def test_confirmed_choices_override_old_ai_flags_and_visibility(self):
        self.assertEqual(classify(self.g,self.a),[])

    def test_unconfirmed_and_unclear_boxes_never_become_gold(self):
        a=copy.deepcopy(self.a);a['confirmed_by_human']=False
        self.assertEqual(classify(self.g,a),['not_human_confirmed'])
        a=copy.deepcopy(self.a);a['boxes'][0][1]='unclear'
        self.assertIn('boxes_not_all_clear',classify(self.g,a))

    def test_structural_montage_and_noncontrasts_are_rejected(self):
        self.assertIn('photo_montage_not_natural_same_time_scene',classify({'events':[{'image_id':487943}]},self.a))
        a=copy.deepcopy(self.a);a['truth'][0]=['supported','supported']
        self.assertIn('not_a_clear_two_way_contrast',classify(self.g,a))


if __name__=='__main__':unittest.main(verbosity=2)
