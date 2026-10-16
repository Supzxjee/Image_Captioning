from pathlib import Path
import unittest
from unittest.mock import patch, call
import numpy as np
from captioning.object_reward import TrainingObjectReward
from run_scst_comparison import train_pair


class CountRewardTests(unittest.TestCase):
    def reward(self, **kwargs):
        rows = [{'image_id': 1, 'supported': ['person', 'sink'], 'salient': ['person'],
                 'detector_neutral': ['dog']}]
        return TrainingObjectReward(rows, penalty_version='unique_count_v2', **kwargs)

    def test_supported_background_cannot_dilute_penalty(self):
        scores, terms = self.reward().score([1, 1], ['person motorcycle', 'person sink motorcycle'])
        np.testing.assert_array_equal(scores, [-.5, -.5])
        self.assertEqual([r['penalty'] for r in terms], [1, 1])
        self.assertNotEqual(terms[0]['unsupported_fraction'], terms[1]['unsupported_fraction'])

    def test_each_distinct_unsupported_class_costs_one(self):
        scores, terms = self.reward().score([1]*3, ['person', 'person motorcycle', 'person motorcycle horse'])
        np.testing.assert_array_equal(scores, [.5, -.5, -1.5])
        self.assertEqual([r['unsupported_count'] for r in terms], [0, 1, 2])

    def test_duplicates_and_neutral_words_cannot_change_reward(self):
        scores, _ = self.reward().score([1]*3, ['person motorcycle', 'person person motorcycle motorcycle',
                                               'person motorcycle dog astronaut'])
        np.testing.assert_array_equal(scores, [-.5]*3)

    def test_salience_coverage_reward_retained(self):
        reward = TrainingObjectReward([{'image_id': 1, 'supported': ['person', 'bicycle'],
                                        'salient': ['person', 'bicycle'], 'detector_neutral': []}],
                                      penalty_version='unique_count_v2')
        scores, _ = reward.score([1]*3, ['person', 'person bicycle', 'person bicycle motorcycle'])
        np.testing.assert_array_equal(scores, [.25, .5, -.5])

    def test_empty_support_remains_neutral(self):
        reward = TrainingObjectReward([{'image_id': 1, 'supported': [], 'salient': [], 'detector_neutral': []}],
                                      penalty_version='unique_count_v2')
        scores, terms = reward.score([1], ['person horse'])
        self.assertEqual(scores[0], 0.)
        self.assertFalse(terms[0]['available'])
        self.assertEqual(terms[0]['penalty'], 0.)

    def test_alias_revision_is_explicit_and_changes_detected_error(self):
        legacy = self.reward()
        revised = self.reward(parser_version='aliases_v2')
        self.assertEqual(legacy.score([1], ['person turkey'])[0][0], .5)
        self.assertEqual(revised.score([1], ['person turkey'])[0][0], -.5)

    def test_invalid_penalty_rejected(self):
        with self.assertRaises(ValueError): TrainingObjectReward([], penalty_version='typo')


class SameGpuSchedulingTests(unittest.TestCase):
    def test_same_gpu_arms_are_sequential_single_worker_calls(self):
        cfg = {'object_reward_comparison': True, 'train_same_gpu': True, 'strict_reproducibility': True}
        with patch('run_scst_comparison._train_parallel') as worker:
            train_pair(cfg, Path('protocol.json'), Path('.'))
        self.assertEqual(worker.call_args_list, [call(cfg, Path('protocol.json'), Path('.'), ('scst_ce',)),
                                                call(cfg, Path('protocol.json'), Path('.'), ('scst_ce_object',))])

    def test_failure_in_control_prevents_treatment(self):
        cfg = {'object_reward_comparison': True, 'train_same_gpu': True, 'strict_reproducibility': True}
        with patch('run_scst_comparison._train_parallel', side_effect=RuntimeError('worker failed')) as worker:
            with self.assertRaises(RuntimeError): train_pair(cfg, Path('p'), Path('.'))
        self.assertEqual(worker.call_count, 1)

    def test_legacy_parallel_schedule_preserved(self):
        cfg = {'object_reward_comparison': True}
        with patch('run_scst_comparison._train_parallel') as worker:
            train_pair(cfg, Path('p'), Path('.'))
        self.assertEqual(worker.call_args.args[-1], ('scst_ce', 'scst_ce_object'))

    def test_strict_mode_required_for_same_gpu(self):
        with self.assertRaises(ValueError): train_pair({'train_same_gpu': True}, Path('p'), Path('.'))


if __name__ == '__main__': unittest.main()
