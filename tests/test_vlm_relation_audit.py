import json
import unittest
from captioning.vlm_relation_audit import audit_prompt, parse_teacher_answer, shard_pairs


class VLMRelationAuditTests(unittest.TestCase):
    def answer(self):
        return {'subject_visible': True, 'object_visible': True,
                'subject_description': 'person', 'object_description': 'glove',
                'subject_label_status': 'supported',
                'object_label_status': 'supported', 'relation_status': 'supported',
                'relations': ['wearing', 'holding'], 'uncertainty': 'none',
                'evidence': 'The glove is on the hand and held in front of the body.'}

    def test_multiple_valid_relations_survive(self):
        result = parse_teacher_answer('```json\n' + json.dumps(self.answer()) + '\n```')
        self.assertEqual(result['proposed_relations'], ['wearing', 'holding'])

    def test_wrong_object_or_uncertainty_abstains(self):
        answer = self.answer()
        answer['object_label_status'] = 'mismatch'
        self.assertEqual(parse_teacher_answer(json.dumps(answer))['proposed_relations'], [])
        answer['object_label_status'] = 'supported'
        answer['uncertainty'] = 'action_unclear'
        self.assertEqual(parse_teacher_answer(json.dumps(answer))['decision'], 'unclear')

    def test_string_booleans_unknown_codes_and_contradictions_rejected(self):
        for key, value in [('subject_visible', 'false'), ('relations', ['imagined_action']),
                           ('relation_status', 'none')]:
            answer = self.answer()
            answer[key] = value
            with self.assertRaises(ValueError):
                parse_teacher_answer(json.dumps(answer))

    def test_no_old_predicate_or_clip_scores_in_prompt(self):
        prompt = audit_prompt({'subject': 'person', 'object': 'skateboard',
                               'old_predicate': 'unique_old_label', 'clip_proposal': 'unique_clip_label'})
        self.assertNotIn('unique_old_label', prompt)
        self.assertNotIn('unique_clip_label', prompt)

    def test_two_shards_cover_pairs_once(self):
        pairs = list(range(100))
        combined = shard_pairs(pairs, 0, 2) + shard_pairs(pairs, 1, 2)
        self.assertEqual(sorted(i for i, _ in combined), list(range(100)))


if __name__ == '__main__':
    unittest.main()
