import copy
import unittest
import torch
from run_scst_comparison import training_arms, objective_weights, caption_ce
from captioning.scst import policy_loss


class TinyCaptioner(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = torch.nn.Embedding(8, 4); self.output = torch.nn.Linear(4, 8)

    def forward(self, images, prompt, prompt_mask, tgt, tgt_key_padding_mask=None):
        memory = images.repeat_interleave(tgt.size(0)//images.size(0), 0)
        return self.output(self.embedding(tgt)+memory[:, None, :])


class AnchorTests(unittest.TestCase):
    def test_old_comparison_keeps_original_weights(self):
        self.assertEqual(training_arms({}), ('ce', 'scst'))
        self.assertEqual(objective_weights({}, 'ce'), (0., 1.))
        self.assertEqual(objective_weights({}, 'scst'), (1., 0.))

    def test_matched_rl_scale_only_treatment_has_ce(self):
        cfg = {'ce_anchor_comparison': True, 'scst_weight': .05, 'ce_weight': 1.}
        self.assertEqual(training_arms(cfg), ('scst', 'scst_ce'))
        self.assertEqual(objective_weights(cfg, 'scst'), (.05, 0.))
        self.assertEqual(objective_weights(cfg, 'scst_ce'), (.05, 1.))
        with self.assertRaises(ValueError): objective_weights(cfg, 'ce')

    def test_sequential_accumulation_equals_joint_loss_and_one_optimizer_step(self):
        torch.manual_seed(42)
        separate = TinyCaptioner(); joint = copy.deepcopy(separate)
        images = torch.randn(2, 4)
        prompt, pm = torch.zeros(2, 1), torch.ones(2, 1)
        sampled = torch.tensor([[1, 4, 2], [1, 5, 2]])
        mask = torch.ones(2, 2).bool(); advantage = torch.tensor([1., -.5])
        captions = torch.tensor([[[1, 3, 2, 0]]*5, [[1, 6, 2, 0]]*5])
        padding = captions.eq(0)
        def rl(model):
            return policy_loss(model(images, prompt, pm, sampled[:, :-1]), sampled[:, 1:], mask, advantage)
        def ce(model): return caption_ce(model, images, prompt, pm, captions, padding, 0)
        (.05*rl(separate)).backward(); ce(separate).backward()
        (.05*rl(joint)+ce(joint)).backward()
        for a, b in zip(separate.parameters(), joint.parameters()): torch.testing.assert_close(a.grad, b.grad)
        for model in (separate, joint):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            torch.optim.AdamW(model.parameters(), lr=5e-6).step()
        for a, b in zip(separate.parameters(), joint.parameters()): torch.testing.assert_close(a, b)


if __name__ == '__main__': unittest.main()
