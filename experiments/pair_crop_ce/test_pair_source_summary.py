import copy
import hashlib
import json
import unittest

from summarize_pair_source_review import summarize


def fixture():
    tasks, mapping, annotations = [], {}, {}
    for g in range(8):
        for i in range(2):
            task = dict(task_id=f'{g}:{i}', event={'image_id': g}, candidates=[])
            for arm, good in [('FULLFRAME', {0, 4}), ('PAIRCROP', {0, 1, 2, 4, 5, 6})]:
                cid = f'{g}:{i}:{arm}'
                task['candidates'].append(dict(candidate_id=cid, caption=cid, truncated=False))
                mapping[cid] = dict(arm=arm, group_id=str(g), event_index=i,
                                    family='skate' if g < 4 else 'surf')
                annotations[cid] = dict(status='correct_specific' if g in good else 'wrong_relation_or_fact', notes='')
            tasks.append(task)
    payload = dict(schema_version=1, tasks=tasks)
    sha = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    payload['review_sha256'] = sha
    key = dict(review_sha256=sha, mapping=mapping)
    review = dict(schema_version=1, complete=True, reviewer_type='human_self_reported',
                  reviewer='Synthetic fixture', review_sha256=sha, annotations=annotations)
    return review, key, payload


class Checks(unittest.TestCase):
    def test_group_criterion_does_not_count_individual_sentences_as_groups(self):
        review, key, payload = fixture()
        result = summarize(review, key, payload)
        self.assertEqual(result['joint_correct_groups'], dict(FULLFRAME=2, PAIRCROP=6))
        self.assertEqual(result['status'], 'GO')
        # One false event makes the whole group fail, though its other event stays correct.
        review['annotations']['1:0:PAIRCROP']['status'] = 'wrong_instance'
        result = summarize(review, key, payload)
        self.assertEqual(result['joint_correct_groups']['PAIRCROP'], 5)
        self.assertEqual(result['correct_events']['PAIRCROP'], 11)
        self.assertEqual(result['status'], 'NO_GO_FOR_FULL_SCALE_PAIRCROP')

    def test_partial_wrong_fingerprint_and_missing_candidate_are_rejected(self):
        for mutate in (lambda r: r.update(complete=False),
                       lambda r: r.update(review_sha256='wrong'),
                       lambda r: r['annotations'].pop('0:0:FULLFRAME')):
            review, key, payload = fixture()
            mutate(review)
            with self.assertRaises(AssertionError):
                summarize(review, key, payload)

    def test_labels_are_preserved_and_unclear_is_not_a_success(self):
        review, key, payload = fixture()
        review['annotations']['1:0:PAIRCROP']['status'] = 'unclear'
        original = copy.deepcopy(review)
        result = summarize(review, key, payload)
        self.assertEqual(review, original)
        self.assertEqual(result['unclear_count'], 1)
        self.assertEqual(result['joint_correct_groups']['PAIRCROP'], 5)


if __name__ == '__main__':
    unittest.main()
