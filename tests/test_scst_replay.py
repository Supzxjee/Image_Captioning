import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_scst_replay import compare_runs


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a, self.b = [Path(self.tmp.name)/name for name in ('a', 'b')]
        runtime = {'deterministic_algorithms': True, 'sdp_math': True, 'warn_only': False,
                   'cublas_workspace': ':4096:8', 'gpu': {'name': 'fixture GPU'}}
        fields = ('batch_indices visual captions prompts caption_masks prompt_masks sample_tokens '
                  'greedy_tokens model_after rng_before rng_after').split()
        history = [{'step': i, 'seconds': float(i), 'loss': 1.0, 'ce_loss': 1.0, 'scst_loss': 0.,
                    'grad_norm': .3, 'sample_cider_reward': .5, 'greedy_cider_reward': .5,
                    **{k+'_sha256': 'fixture_'+k+str(i) for k in fields}} for i in range(1, 33)]
        for root in (self.a, self.b):
            root.mkdir()
            for name, data in [('runtime', runtime), ('history', history),
                    ('initial_model_identity', {'state_sha256': 'initial'}),
                    ('replay_summary', {'steps': 32, 'train_images_for_cider_idf': 20000,
                                       'final_model_sha256': 'final', 'optimizer_sha256': 'optimizer'})]:
                self.write(root, name, data)

    def write(self, root, name, data):
        (root/(name+'.json')).write_text(json.dumps(data), encoding='utf-8')

    def change(self, name, callback):
        data = json.loads((self.b/(name+'.json')).read_text())
        callback(data)
        self.write(self.b, name, data)

    def test_identical_runs_ignore_only_wall_time(self):
        self.change('history', lambda rows: rows[0].update(seconds=999.))
        self.assertEqual(compare_runs(self.a, self.b)['status'], 'PASS')

    def test_step_two_one_ulp_change_is_detected(self):
        self.change('history', lambda rows: rows[1].update(ce_loss=1.0000001192092896))
        result = compare_runs(self.a, self.b)
        self.assertFalse(result['exact_replay'])
        self.assertEqual(result['differences'][0]['step'], 2)
        self.assertIn('ce_loss', result['differences'][0]['changed_fields'])

    def test_missing_trace_is_not_a_pass_even_when_both_missing(self):
        for root in (self.a, self.b):
            rows = json.loads((root/'history.json').read_text())
            del rows[0]['sample_tokens_sha256']
            self.write(root, 'history', rows)
        self.assertFalse(compare_runs(self.a, self.b)['exact_replay'])

    def test_final_optimizer_difference_detected(self):
        self.change('replay_summary', lambda data: data.update(optimizer_sha256='changed'))
        self.assertFalse(compare_runs(self.a, self.b)['exact_replay'])

    def test_31_steps_cannot_pass(self):
        self.change('history', lambda rows: rows.pop())
        self.assertFalse(compare_runs(self.a, self.b)['exact_replay'])

    def test_equal_non_strict_runs_cannot_pass(self):
        for root in (self.a, self.b):
            self.write(root, 'runtime', {})
        self.assertFalse(compare_runs(self.a, self.b)['exact_replay'])

    def test_nonfinite_loss_cannot_pass(self):
        self.change('history', lambda rows: rows[0].update(loss=float('inf')))
        self.assertFalse(compare_runs(self.a, self.b)['exact_replay'])


class StateDigestTests(unittest.TestCase):
    def test_optimizer_digest_tracks_tensor_values_but_not_dict_order(self):
        import torch
        from captioning.scst_runtime import state_digest
        state = {'state': {0: {'exp_avg': torch.tensor([1., 2.]), 'step': torch.tensor(32.)}},
                 'param_groups': [{'lr': 5e-6, 'params': [0]}]}
        reordered = dict(reversed(list(state.items())))
        self.assertEqual(state_digest(state), state_digest(reordered))
        changed = copy.deepcopy(state)
        changed['state'][0]['exp_avg'][0] += 1
        self.assertNotEqual(state_digest(state), state_digest(changed))


if __name__ == '__main__': unittest.main()
