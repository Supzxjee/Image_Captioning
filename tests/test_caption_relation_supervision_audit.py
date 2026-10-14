import unittest
from audit_caption_relation_supervision import lexical_claim


class CaptionRelationAuditTests(unittest.TestCase):
    def test_reading_is_not_literal_holding_and_does_not_mean_false(self):
        x=lexical_claim('a girl reading a book','holding','book')
        self.assertTrue(x['object']);self.assertFalse(x['joint_cooccurrence'])
        self.assertEqual(x['semantic_truth'],'not assessed')

    def test_negation_and_multiple_people_are_not_entailment(self):
        x=lexical_claim('a person is not holding a cup','holding','cup')
        self.assertTrue(x['joint_cooccurrence']);self.assertTrue(x['potential_negation'])
        self.assertEqual(x['semantic_truth'],'not assessed')


if __name__=='__main__':unittest.main()
