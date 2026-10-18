import copy
import unittest

from captioning.vcoco_relation_audit import audit_image, evaluate, fit_thresholds, official_upt_claims
from prepare_vcoco_grounded_caption_data import grounded_images
from tests.test_vcoco_grounded_caption_data import fixture


def data():
    actions, instances, captions = fixture()
    for index, row in enumerate(instances['annotations']):
        row['bbox'] = [20 + 80 * index, 20, 30, 50]
    records, _ = grounded_images(actions, instances, captions, {100})
    detections = [{'label': 'person' if r['category_id'] == 1 else 'horse', 'confidence': .9,
                   'box': [r['bbox'][0] / 640, 20 / 480, (r['bbox'][0] + 30) / 640, 70 / 480]}
                  for r in instances['annotations']]
    claims = [{'subject': s, 'object': o, 'predicate': p, 'score': score}
              for s, o, p, score in [(0, 3, 'holding', .95), (0, 4, 'holding', .94),
                                     (1, 3, 'holding', .1), (2, 3, 'holding', .8),
                                     (0, 3, 'riding', .9), (1, 4, 'riding', .85)]]
    return records[0], {'detections': detections, 'claims': claims, 'native_pairs': 5}, instances


def scored():
    r, p, i = data()
    return audit_image(r, p, i['annotations'], i['images'][0], {1: 'person', 2: 'horse'})


class RelationAuditTests(unittest.TestCase):
    def test_official_scores_preserved_with_correct_pair_and_verb(self):
        output = {'pairing': [[2, 0, 2], [3, 1, 4]], 'labels': [2, 0, 1], 'scores': [.0123, .9, .004]}
        self.assertEqual(official_upt_claims(output, ['other', 'ride', 'hold']), [
            {'subject': 2, 'object': 3, 'predicate': 'holding', 'score': .0123},
            {'subject': 2, 'object': 4, 'predicate': 'riding', 'score': .004}])
        self.assertEqual(official_upt_claims({'pairing': [[], []], 'labels': [], 'scores': []},
                                            ['hold', 'ride']), [])
        output['scores'].pop()
        with self.assertRaises(ValueError):
            official_upt_claims(output, ['other', 'ride', 'hold'])

    def test_same_actor_wrong_target_unknown_not_negative(self):
        result = scored()
        self.assertEqual([c['label'] for c in result['candidates']], [1, -1, 0, -1, 1, 1])

    def test_missing_detection_stays_in_positive_denominator(self):
        record, prediction, instances = data()
        prediction['detections'] = prediction['detections'][:3]
        prediction.update(claims=[], native_pairs=0)
        audited = audit_image(record, prediction, instances['annotations'], instances['images'][0], {1: 'person', 2: 'horse'})
        result = evaluate([audited], fit_thresholds([]))
        self.assertEqual(result['images_without_native_pairs'], 1)
        self.assertEqual(result['by_predicate']['riding']['annotated_positive_links'], 2)
        self.assertEqual(result['by_predicate']['riding']['full_positive_recall'], 0)

    def test_all_gt_compete_and_unmatched_actor_unknown(self):
        record, prediction, instances = data()
        old = instances['annotations'][0]
        instances['annotations'].append({**copy.deepcopy(old), 'id': 99})
        old['bbox'][0] += 1
        result = audit_image(record, prediction, instances['annotations'], instances['images'][0], {1: 'person', 2: 'horse'})
        self.assertEqual(result['candidates'][0]['label'], -1)

    def test_threshold_ignores_unknown_in_precision_but_counts_selected_unknown(self):
        row = scored()
        policy = fit_thresholds([row], precision=.9, min_known=1)
        self.assertEqual(policy['predicates']['holding']['threshold'], .95)
        row['candidates'][1]['score'] = .99
        result = evaluate([row], policy)['by_predicate']['holding']
        self.assertEqual(result['known_precision'], 1)
        self.assertEqual(result['accepted_unknown'], 1)
        self.assertEqual(result['accepted_true'], 1)

    def test_no_both_outcomes_or_insufficient_support_abstains(self):
        policy = fit_thresholds([scored()])
        self.assertTrue(all(p['threshold'] == 1.01 for p in policy['predicates'].values()))
        result = evaluate([scored()], policy)
        self.assertIsNone(result['by_predicate']['holding']['known_precision'])

    def test_tied_scores_do_not_split_true_and_false(self):
        row = scored()
        for c in row['candidates']:
            c['score'] = .5
        self.assertEqual(fit_thresholds([row], min_known=1)['predicates']['holding']['threshold'], 1.01)

    def test_duplicate_claim_or_self_pair_rejected(self):
        for mode in ('duplicate', 'self'):
            record, prediction, instances = data()
            if mode == 'duplicate':
                prediction['claims'].append(copy.deepcopy(prediction['claims'][0]))
            else:
                prediction['claims'][0]['object'] = 0
            with self.assertRaises(ValueError):
                audit_image(record, prediction, instances['annotations'], instances['images'][0], {1: 'person', 2: 'horse'})


if __name__ == '__main__':
    unittest.main()
