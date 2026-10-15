import unittest
from types import SimpleNamespace
import numpy as np
import torch
from pycocoevalcap.cider.cider_scorer import CiderScorer
from captioning.scst import TrainingCider, normalize, rollout, policy_loss, decode
from run_scst_comparison import cohorts


class ScstTests(unittest.TestCase):
    def setUp(self):
        self.refs = {1: ['A man riding a horse.']*5, 2: ['A dog runs through grass.']*5,
                     3: ['A woman holding an umbrella.']*5}

    def test_reward_matches_official_scorer_on_identically_normalized_corpus(self):
        reward = TrainingCider(self.refs)
        texts = ['A man rides a horse!', 'a dog', 'a woman holding an umbrella']
        official = CiderScorer()
        for i, text in zip(self.refs, texts): official += (normalize(text), reward.references[i])
        _, expected = official.compute_score()
        np.testing.assert_allclose(reward.score(list(self.refs), texts), expected, rtol=1e-6)

    def test_reward_batch_independence_and_repeated_ids(self):
        reward = TrainingCider(self.refs)
        a = reward.score([1], ['a man riding a horse'])[0]
        b = reward.score([2, 1, 1], ['a dog', 'a man riding a horse', 'a man riding a horse'])
        self.assertEqual(a, b[1]); self.assertEqual(a, b[2])

    def test_empty_unknown_and_oversized_captions_finite(self):
        result = TrainingCider(self.refs).score([1]*3, ['', 'xyz unknown', 'horse '*100])
        self.assertTrue(np.isfinite(result).all()); self.assertEqual(float(result[0]), 0.)

    def test_reject_missing_references_and_unknown_image(self):
        with self.assertRaises(ValueError): TrainingCider({1: ['a']})
        with self.assertRaises(KeyError): TrainingCider(self.refs).score([999], ['a'])

    def test_advantage_sign_and_stop_gradient(self):
        for advantage, sign in [(1., -1), (-1., 1), (0., 0)]:
            logits = torch.zeros(1, 1, 3, requires_grad=True)
            adv = torch.tensor([advantage], requires_grad=True)
            policy_loss(logits, torch.tensor([[1]]), torch.tensor([[True]]), adv).backward()
            self.assertEqual(int(logits.grad[0, 0, 1].sign()), sign)
            self.assertIsNone(adv.grad)

    def test_eos_included_post_eos_excluded_sum_not_length_normalized(self):
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        loss = policy_loss(logits, torch.tensor([[1, 2, 0]]), torch.tensor([[True, True, False]]), torch.ones(1))
        self.assertAlmostEqual(float(loss.detach()), 2*np.log(4), places=6)
        loss.backward(); self.assertEqual(float(logits.grad[:, 2].abs().sum()), 0.)

    def test_rollout_stops_each_image_at_own_eos(self):
        class Decoder:
            def __call__(self, seq, memory, **kwargs):
                result = torch.full((2, seq.size(1), 5), -100.)
                result[0, -1, 2] = 100.
                result[1, -1, 3 if seq.size(1) < 3 else 2] = 100.
                return result
        tokenizer = SimpleNamespace(bos_idx=1, eos_idx=2, pad_idx=0, UNK='<unk>', idx2word={3:'horse'})
        seq, mask = rollout(SimpleNamespace(decoder=Decoder()), torch.zeros(2, 1, 4), torch.zeros(2, 1).bool(), tokenizer, False)
        self.assertEqual(mask.tolist(), [[True, False, False], [True, True, True]])
        self.assertEqual(decode(seq, tokenizer), ['', 'horse horse'])
        self.assertFalse(seq.requires_grad)

    def test_no_eos_uses_all_steps(self):
        class Decoder:
            def __call__(self, seq, memory, **kwargs):
                logits = torch.zeros(1, seq.size(1), 4); logits[..., 3] = 20; return logits
        t = SimpleNamespace(bos_idx=1, eos_idx=2, pad_idx=0)
        seq, mask = rollout(SimpleNamespace(decoder=Decoder()), torch.zeros(1, 1, 4), torch.zeros(1, 1).bool(), t, False, max_len=5)
        self.assertEqual(seq.shape, (1, 5)); self.assertTrue(mask.all())

    def test_rescoring_gradients_reach_memory_and_decoder(self):
        decoder = torch.nn.Linear(3, 4)
        memory = torch.randn(2, 3, requires_grad=True)
        logits = decoder(memory)[:, None, :].expand(-1, 2, -1)
        policy_loss(logits, torch.tensor([[1, 2], [3, 2]]), torch.ones(2, 2).bool(), torch.tensor([1., -.5])).backward()
        self.assertGreater(float(memory.grad.abs().sum()), 0.)
        self.assertGreater(float(decoder.weight.grad.abs().sum()), 0.)

    def test_cohort_deterministic_disjoint_excludes_test(self):
        data = {'images': [{'split': s, 'filename': f'COCO_train2014_{i:012}.jpg', 'filepath': 'train2014'}
                           for i, s in enumerate(['train']*8+['restval']*3+['val']*5+['test']*4)]}
        a = cohorts(data, 8, 3)
        self.assertEqual(a, cohorts(data, 8, 3))
        self.assertTrue(set(r['image_id'] for r in a['train']).isdisjoint(r['image_id'] for r in a['val']))
        self.assertTrue(all(r['image_id'] < 16 for rows in a.values() for r in rows))

    def test_real_qformer_rollout_rescore_and_ce_backward(self):
        # The cached-feature path never calls CLIPModel. Isolate that unused
        # import so this CPU test needs no pretrained download/torchvision ops.
        from unittest.mock import patch
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(CLIPModel=object)}):
            from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
        torch.set_num_threads(2); torch.manual_seed(42)
        model = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
                  num_visual_queries=32, qformer_layers=2), CaptionDecoder(vocab_size=12)).eval()
        image, prompt = torch.randn(1, 197, 768), torch.randn(1, 20, 512)
        padding = torch.ones(1, 20, dtype=torch.long)
        t = SimpleNamespace(bos_idx=1, eos_idx=2, pad_idx=0)
        with torch.no_grad():
            memory, memory_padding = model.build_memory(image, prompt, padding)
            sequence, active = rollout(model, memory, memory_padding, t, True, max_len=5)
            rollout_logits = model.decoder(sequence[:, :-1], memory, memory_key_padding_mask=memory_padding)
        logits = model(image, prompt, padding, sequence[:, :-1])
        torch.testing.assert_close(logits.detach(), rollout_logits)
        policy_loss(logits, sequence[:, 1:], active, torch.ones(1)).backward()
        for component in (model.encoder.qformer, model.encoder.prompt_visual_gate, model.decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None), 0.)
        model.zero_grad(set_to_none=True)
        target = torch.tensor([[1, 4, 2, 0]]).expand(5, -1)
        logits = model(image, prompt, padding, target[:, :-1], tgt_key_padding_mask=target[:, :-1].eq(0))
        self.assertEqual(logits.shape, (5, 3, 12))
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), target[:, 1:].reshape(-1), ignore_index=0).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__': unittest.main()
