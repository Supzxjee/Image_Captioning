import unittest

try:
    import torch
    from captioning.models import LightweightQFormer
except (ImportError, OSError):
    torch = None
    LightweightQFormer = None


@unittest.skipIf(torch is None, 'PyTorch is unavailable in this environment.')
class PromptConditionedQFormerTests(unittest.TestCase):
    def test_prompt_changes_queries_and_padding_is_ignored(self):
        torch.manual_seed(3)
        model = LightweightQFormer(
            embed_dim=8, num_heads=2, num_queries=3, num_layers=1,
            dropout=0.0, prompt_conditioned=True,
        ).eval()
        visual = torch.randn(2, 5, 8)
        prompt = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
        prompt[1, :2] = prompt[0, :2] + 2.0
        prompt[1, 2:] = 1000.0
        output = model(visual, prompt, mask)
        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertFalse(torch.allclose(output[0], output[1]))

        altered_padding = prompt.clone()
        altered_padding[:, 2:] = -999.0
        self.assertTrue(torch.allclose(
            model(visual, prompt, mask), model(visual, altered_padding, mask), atol=1e-6
        ))

    def test_conditioned_model_requires_prompt(self):
        model = LightweightQFormer(8, 2, 3, 1, prompt_conditioned=True)
        with self.assertRaisesRegex(ValueError, 'requires prompt'):
            model(torch.randn(1, 5, 8))


if __name__ == '__main__':
    unittest.main()
