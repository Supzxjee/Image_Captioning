import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from captioning.upt_cache import pack_prediction, image_plan, identity, finish_shard
from captioning.upt_prompts import build_prompt, from_audit, from_cache, relation_text
from build_upt_prompts import records
from run_vcoco_upt_full_audit import CHECKPOINT
from tests.test_upt_full_cache import fixture


class UptPromptTests(unittest.TestCase):
    def fixture(self):
        dataset, vocab, prediction = fixture()
        row = pack_prediction(image_plan(dataset)[0]['images'][0], prediction, vocab, 'hash', [640, 480])
        dets, claims = from_cache(row, vocab)
        policy = {'predicates': {'holding': {'threshold': .05}, 'riding': {'threshold': 1.01}}}
        return dets, claims, policy

    def test_locked_abstain_not_bypassed_by_rendering(self):
        dets, claims, policy = self.fixture()
        entry = build_prompt('a.jpg', dets, claims, policy)
        self.assertEqual([c['predicate'] for c in entry['relations']], ['holding'])
        self.assertIn('holding a horse', entry['prompt'])
        self.assertNotIn('riding', entry['prompt'])
        self.assertFalse(entry['candidate_preview'][0]['passes_locked_threshold'])

    def test_broad_ride_does_not_invent_specific_action(self):
        self.assertEqual(relation_text('riding', 'surfboard'), 'a person riding a surfboard')
        self.assertEqual(relation_text('riding', 'skis'), 'a person on skis')
        self.assertEqual(relation_text('holding', 'umbrella'), 'a person holding an umbrella')

    def test_empty_and_objects_only_fallback(self):
        dets, _, policy = self.fixture()
        self.assertEqual(build_prompt('a.jpg', [], [], policy)['mode'], 'no_objects')
        entry = build_prompt('a.jpg', dets, [], policy)
        self.assertEqual(entry['mode'], 'object_only')
        self.assertIn('a horse', entry['prompt'])
        self.assertFalse(entry['relations'])

    def test_annotation_fields_cannot_change_prompt(self):
        dets, claims, policy = self.fixture()
        row = {'detection_details': dets, 'candidates': claims}
        expected = build_prompt('a.jpg', *from_audit(row), policy)
        row['reference_captions'] = ['a unicorn flies']
        for c in row['candidates']:
            c.update(label=-1, matched_subject_id=123, matched_object_id=999)
        self.assertEqual(build_prompt('a.jpg', *from_audit(row), policy), expected)

    def test_duplicate_cells_deduplicated_instances_not_merged(self):
        dets, claims, policy = self.fixture()
        dets.append({**copy.deepcopy(dets[1]), 'box': [.6, .2, .9, .7]})
        claims += [copy.deepcopy(claims[0]), {**claims[0], 'object': 2}]
        entry = build_prompt('a.jpg', dets, claims, policy)
        self.assertEqual(len(entry['relations']), 2)
        self.assertEqual(len({c['object_instance'] for c in entry['relations']}), 2)
        self.assertEqual(entry['above_threshold_claims'], 2)

    def test_cap_and_below_object_threshold_endpoint_kept(self):
        dets, claims, policy = self.fixture()
        dets[1]['confidence'] = .1
        entry = build_prompt('a.jpg', dets, claims, policy, max_relations=1)
        self.assertEqual(len(entry['relations']), 1)
        self.assertEqual(len(entry['objects']), 2)

    def test_invalid_score_box_or_policy_rejected(self):
        for mode in ('score', 'box', 'policy'):
            dets, claims, policy = self.fixture()
            if mode == 'score': claims[0]['score'] = float('nan')
            elif mode == 'box': dets[0]['box'][0] = 2
            else: policy['predicates']['holding']['threshold'] = -1
            with self.assertRaises(ValueError): build_prompt('a.jpg', dets, claims, policy)

    def test_native_sparse_dimension_rejected(self):
        dataset, vocab, prediction = fixture()
        row = pack_prediction({}, prediction, vocab, 'hash', [640, 480])
        row['native_scores'][0].pop()
        with self.assertRaises(ValueError): from_cache(row, vocab)

    def test_full_cache_adapter_and_missing_shard(self):
        dataset, vocab, pred = fixture()
        tasks = image_plan(dataset)
        meta = {'schema': 'upt_full_coco_supported_logits_v1', 'checkpoint_sha256': CHECKPOINT,
                'vocabulary': vocab, 'image_plan_sha256': identity(tasks)}
        sig = identity(meta)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'cache_identity.json').write_text(json.dumps({'signature': sig, 'identity': meta}))
            (root / 'session_config.json').write_text(json.dumps({'signature': sig, 'tasks': tasks, 'vocabulary': vocab}))
            packed = [pack_prediction(row, pred, vocab, 'hash', [640, 480]) for row in tasks[0]['images']]
            finish_shard(root, tasks[0], sig, packed)
            args = SimpleNamespace(audit_zip=None, cache=root, allow_partial=True)
            self.assertEqual(len(list(records(args))), 3)
            args.allow_partial = False
            with self.assertRaises(FileNotFoundError): list(records(args))


if __name__ == '__main__':
    unittest.main()
