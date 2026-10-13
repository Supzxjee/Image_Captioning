import unittest
from captioning.relation_candidates import candidate_phrases, select_candidate, union_crop_box


class RelationCandidateTests(unittest.TestCase):
    def test_skateboard_has_kneeling_alternative_and_neutral(self):
        candidates = candidate_phrases('person', 'skateboard')
        labels = {x['predicate'] for x in candidates}
        self.assertTrue({'riding', 'kneeling_beside', 'kneeling_on', 'near'} <= labels)

    def test_view_disagreement_abstains(self):
        c = [{'predicate': 'riding'}, {'predicate': 'kneeling_beside'}]
        result = select_candidate(c, [.3, .2], [.2, .3])
        self.assertIsNone(result['proposed_predicate'])

    def test_tie_neutral_and_nonfinite_cannot_be_forced_into_action(self):
        c = [{'predicate': 'holding'}, {'predicate': 'near'}]
        self.assertIsNone(select_candidate(c, [.3, .3], [.3, .3])['proposed_predicate'])
        self.assertIsNone(select_candidate(c, [.2, .3], [.2, .3])['proposed_predicate'])
        with self.assertRaises(ValueError):
            select_candidate(c, [float('nan'), .3], [.3, .2])

    def test_agreement_and_margin_returns_proposal_for_review(self):
        c = [{'predicate': 'holding'}, {'predicate': 'near'}]
        self.assertEqual(select_candidate(c, [.4, .2], [.3, .2])['proposed_predicate'], 'holding')

    def test_crop_contains_pair_and_clips_padding_to_image(self):
        crop = union_crop_box([0., .1, .2, .5], [.8, .6, 1., .9], 640, 480)
        self.assertEqual(crop[0], 0)
        self.assertEqual(crop[2], 640)
        self.assertLessEqual(crop[1], 48)
        self.assertGreaterEqual(crop[3], 432)
        with self.assertRaises(ValueError):
            union_crop_box([.5, .1, .2, .5], [.8, .6, 1., .9], 640, 480)


if __name__ == '__main__':
    unittest.main()
