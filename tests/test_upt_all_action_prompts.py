import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from captioning.upt_action_text import ACTION_TEXT, action_predicate, uniform_policy
from captioning.upt_prompts import from_cache, relation_text, build_prompt
from captioning.upt_cache import image_plan, identity, pack_prediction, finish_shard
from build_upt_prompts import run
from run_vcoco_upt_full_audit import CHECKPOINT
from tests.test_upt_full_cache import fixture


class AllActionTests(unittest.TestCase):
    def test_complete_builder_from_cache_to_prompt_and_review(self):
        dataset, vocab, pred = fixture()
        vocab['verbs'][2] = 'eat'
        tasks = image_plan(dataset)
        meta = {'schema': 'upt_full_coco_supported_logits_v1', 'checkpoint_sha256': CHECKPOINT,
                'vocabulary': vocab, 'image_plan_sha256': identity(tasks),
                'split_images': {'train': 3, 'val': 1, 'test': 1}}
        sig = identity(meta)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root / 'cache'; cache.mkdir()
            (cache / 'cache_identity.json').write_text(json.dumps({'signature': sig, 'identity': meta}))
            (cache / 'session_config.json').write_text(json.dumps({'signature': sig, 'tasks': tasks, 'vocabulary': vocab}))
            for task in tasks:
                finish_shard(cache, task, sig, [pack_prediction(r, pred, vocab, 'hash', [640, 480]) for r in task['images']])
            args = SimpleNamespace(cache=cache, audit_zip=None, output=root / 'result', policy=None,
                                   all_actions=True, relation_threshold=.5, allow_partial=False,
                                   max_relations=3, object_threshold=.5, max_object_names=8, review_images=None)
            with contextlib.redirect_stdout(io.StringIO()): run(args)
            summary = json.loads((args.output / 'prompt_summary.json').read_text())
            prompts = json.loads((args.output / 'prompt_cache.json').read_text())
            self.assertEqual(summary['supported_actions'], 116)
            self.assertFalse(summary['threshold_calibrated'])
            self.assertEqual(summary['images'], 5)
            self.assertTrue(summary['complete'])
            self.assertTrue(all('riding a horse' in p['prompt'] for p in prompts.values()))
            self.assertIn('116', (args.output / 'prompt_review_50.html').read_text(encoding='utf-8'))

    def test_all_600_supported_combinations_and_116_actions(self):
        data = json.loads((Path(__file__).resolve().parents[1] / 'audit/upt_prompt_action_vocabulary.json').read_text())
        self.assertEqual(set(ACTION_TEXT), set(data['verbs']) - {'no_interaction'})
        correspondence = [[] for _ in data['objects']]
        for _, obj, verb in data['correspondence']:
            correspondence[obj].append(verb)
        vocab = {**data, 'verbs_by_object': correspondence}
        row = {'resized_size_hw': [100, 100], 'labels': [0] + list(range(80)),
               'object_scores': [.9] * 81, 'boxes_resized_xyxy': [[1, 1, 90, 90]] * 81,
               'pairs': [[0, obj + 1] for obj in range(80)],
               'native_scores': [[.8] * len(verbs) for verbs in correspondence]}
        dets, claims = from_cache(row, vocab, all_actions=True)
        self.assertEqual(len(data['correspondence']), 600)
        self.assertEqual(len(claims), 520)  # 80 no_interaction combinations omitted.
        self.assertEqual(len({c['predicate'] for c in claims}), 116)
        for c in claims:
            text = relation_text(c['predicate'], dets[c['object']]['label'])
            self.assertTrue(text.startswith('a person'))
            self.assertNotIn('_', text)
        result = build_prompt('a.jpg', dets, claims, uniform_policy(), max_relations=3)
        self.assertEqual(len(result['relations']), 3)
        self.assertEqual(result['above_threshold_claims'], 520)
        self.assertNotIn('no_interaction', result['prompt'])
        self.assertFalse(build_prompt('a.jpg', dets, claims, uniform_policy(.9))['relations'])

    def test_prepositions_and_gerunds(self):
        examples = [('eat', 'apple', 'eating an apple'), ('carry', 'backpack', 'carrying a backpack'),
                    ('sit_on', 'chair', 'sitting on a chair'), ('lie_on', 'bed', 'lying on a bed'),
                    ('type_on', 'laptop', 'typing on a laptop'), ('wear', 'tie', 'wearing a tie'),
                    ('cut_with', 'scissors', 'cutting with scissors')]
        for verb, obj, expected in examples:
            self.assertEqual(relation_text(verb, obj), 'a person ' + expected)
        self.assertIn('riding', relation_text(action_predicate('ride'), 'bicycle'))
        with self.assertRaises(ValueError): relation_text('no_interaction', 'chair')

    def test_invalid_exploratory_threshold_rejected(self):
        for threshold in (0, -1, 1.01, float('nan')):
            with self.assertRaises(ValueError): uniform_policy(threshold)


if __name__ == '__main__':
    unittest.main()
