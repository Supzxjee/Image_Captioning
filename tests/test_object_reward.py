import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from captioning.object_reward import TrainingObjectReward, combine_rewards, make_targets
from captioning.scst import policy_loss
from run_scst_comparison import training_arms, objective_weights


class ObjectRewardTests(unittest.TestCase):
    def reward(self):
        return TrainingObjectReward([{'image_id': 1, 'supported': ['person', 'bicycle'],
                                     'salient': ['person', 'bicycle'], 'detector_neutral': ['dog']}])

    def test_correct_objects_score_above_generic_and_hallucinated(self):
        scores, terms = self.reward().score([1]*3, ['a person riding a bicycle', 'a person outside', 'a person riding a motorcycle'])
        np.testing.assert_allclose(scores, [.5, .25, -.25])
        self.assertEqual(terms[2]['unsupported_classes'], ['motorcycle'])

    def test_duplicates_cannot_farm_reward_and_no_objects_not_full_credit(self):
        scores, _ = self.reward().score([1]*3, ['person bicycle', 'person person bicycle bicycle', 'outside today'])
        self.assertEqual(scores[0], scores[1]); self.assertEqual(scores[2], 0.)

    def test_detector_neutral_no_positive_reward_or_penalty_dilution(self):
        scores, terms = self.reward().score([1]*3, ['a dog', 'person motorcycle', 'person motorcycle dog'])
        self.assertEqual(scores[0], 0.); self.assertEqual(scores[1], scores[2])
        self.assertEqual(terms[2]['neutral_classes'], ['dog'])

    def test_unknown_vocabulary_neutral_and_missing_id_rejected(self):
        self.assertEqual(self.reward().score([1], ['an astronaut'])[0][0], 0.)
        with self.assertRaises(KeyError): self.reward().score([2], ['person'])
        with self.assertRaises(ValueError): self.reward().score([1], [])

    def test_empty_annotation_image_neutral_instead_of_negative_for_every_object(self):
        reward = TrainingObjectReward([{'image_id': 1, 'supported': [], 'salient': [], 'detector_neutral': []}])
        scores, terms = reward.score([1], ['person bicycle horse'])
        self.assertEqual(scores[0], 0.); self.assertFalse(terms[0]['available'])

    def test_aliases_longest_match_no_dog_from_hot_dog(self):
        reward = TrainingObjectReward([{'image_id': 1, 'supported': ['person', 'hot dog'],
                                       'salient': ['person', 'hot dog'], 'detector_neutral': []}])
        scores, terms = reward.score([1], ['a man eating hot dogs'])
        self.assertEqual(float(scores[0]), .5)
        self.assertNotIn('dog', terms[0]['recognized_classes'])

    def test_empty_salience_is_finite_and_invalid_states_rejected(self):
        reward = TrainingObjectReward([{'image_id': 1, 'supported': ['person'], 'salient': [], 'detector_neutral': []}])
        scores, _ = reward.score([1]*2, ['person', 'horse'])
        np.testing.assert_allclose(scores, [0., -1.])
        with self.assertRaises(ValueError): TrainingObjectReward([{'image_id': 1, 'supported': ['person'], 'salient': ['horse'], 'detector_neutral': []}])

    def test_reward_changes_policy_gradient_not_decoder_input(self):
        cid = np.array([1.], dtype=np.float32)
        obj, _ = self.reward().score([1], ['person motorcycle'])
        advantage = torch.from_numpy(combine_rewards(cid, obj, 1.)-cid)
        logits = torch.zeros(1, 1, 4, requires_grad=True)
        policy_loss(logits, torch.tensor([[3]]), torch.ones(1, 1).bool(), advantage).backward()
        self.assertGreater(float(logits.grad[0, 0, 3]), 0.)
        np.testing.assert_equal(combine_rewards(cid, obj, 0.), cid)
        with self.assertRaises(ValueError): combine_rewards(cid, obj, -1.)

    def test_object_comparison_preserves_ce_and_rl_in_both_arms(self):
        cfg = {'object_reward_comparison': True, 'scst_weight': .05, 'ce_weight': 1.}
        self.assertEqual(training_arms(cfg), ('scst_ce', 'scst_ce_object'))
        for arm in training_arms(cfg): self.assertEqual(objective_weights(cfg, arm), (.05, 1.))

    def test_target_builder_only_reads_selected_training_annotation(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key == 20: raise AssertionError('Read held-out annotation')
                return super().__getitem__(key)
        source = SimpleNamespace(imid_to_objects=Guard({10: {'person'}, 20: {'dog'}}))
        groups = {'train': [{'image_id': 0, 'coco_id': 10, 'filename': 'train.jpg'}],
                  'val': [{'image_id': 1, 'coco_id': 20, 'filename': 'val.jpg'}]}
        dataset = {'images': [{'split': 'train', 'filename': 'train.jpg', 'sentences': [{'raw': 'person riding bicycle'}]*5},
                              {'split': 'val', 'filename': 'val.jpg', 'sentences': [{'raw': 'dog'}]*5}]}
        detections = {'train.jpg': [{'label': 'dog', 'conf': .95}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'annotation.pkl'; p.write_bytes(b'test')
            with patch('captioning.object_reward.CACHE_SHA256', hashlib.sha256(b'test').hexdigest()), \
                 patch('captioning.object_reward._AnnotationReader') as reader:
                reader.return_value.load.return_value = source
                bundle = make_targets(groups, dataset, p, detections)
                self.assertEqual(len(bundle['data']), 1)
                self.assertEqual(bundle['data'][0]['coco_id'], 10)
                self.assertEqual(bundle['data'][0]['supported'], ['bicycle', 'person'])
                self.assertEqual(bundle['data'][0]['detector_neutral'], ['dog'])
                dataset['images'][0]['split'] = 'test'
                with self.assertRaises(ValueError): make_targets(groups, dataset, p, detections)


if __name__ == '__main__': unittest.main()
