import json
import unittest
from captioning.vlm_relation_audit import (audit_prompt, parse_teacher_answer, shard_pairs,
                                          verification_prompt, parse_verification)


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

    def test_schema_repair_is_opt_in_and_exact(self):
        answer = self.answer()
        answer.update(relation_status='holding', relations=['holding'])
        with self.assertRaises(ValueError):
            parse_teacher_answer(json.dumps(answer))
        fixed = parse_teacher_answer(json.dumps(answer), repair_status=True)
        self.assertEqual(fixed['schema_repair']['original'], 'holding')
        self.assertEqual(fixed['proposed_relations'], ['holding'])
        answer['relations'] = ['wearing']
        with self.assertRaises(ValueError):
            parse_teacher_answer(json.dumps(answer), repair_status=True)

    def test_verifier_rejects_wrong_binding_even_when_verdict_supported(self):
        answer = {'subject_identity': 'supported', 'object_identity': 'supported',
                  'subject_description': 'man', 'object_description': 'tie',
                  'actor_is_marked_A': True, 'target_is_marked_B': True,
                  'other_actor_or_object_explains_evidence': False,
                  'direct_evidence': 'Tie attached around the neck.', 'verdict': 'supported'}
        self.assertTrue(parse_verification(json.dumps(answer))['eligible_for_review'])
        for key, value in [('target_is_marked_B', False), ('actor_is_marked_A', False),
                           ('other_actor_or_object_explains_evidence', True),
                           ('object_identity', 'unclear'), ('verdict', 'contradicted')]:
            changed = {**answer, key: value}
            self.assertFalse(parse_verification(json.dumps(changed))['eligible_for_review'])
        answer['target_is_marked_B'] = 'true'
        with self.assertRaises(ValueError):
            parse_verification(json.dumps(answer))

    def test_verifier_does_not_reveal_teacher_rationale(self):
        prompt = verification_prompt({'subject': 'person', 'object': 'tie',
                                      'evidence': 'secret_old_explanation',
                                      'old_predicate': 'secret_old_predicate'}, 'wearing')
        self.assertNotIn('secret_old', prompt)
        self.assertIn('marked A wearing marked B', prompt)
        self.assertNotIn('|', prompt)
        example, _ = json.JSONDecoder().raw_decode(prompt[prompt.find('{'):])
        self.assertFalse(parse_verification(json.dumps(example))['eligible_for_review'])


if __name__ == '__main__':
    unittest.main()
