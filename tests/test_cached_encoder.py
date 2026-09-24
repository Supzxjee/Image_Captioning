"""CPU regression tests without downloading CLIP weights."""
import importlib
import types
import unittest
from unittest.mock import patch
import torch
import torch.nn as nn


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer('tokens', torch.randn(1, 197, 768).half().float())
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=self.tokens.expand(pixel_values.size(0), -1, -1))


class CachedEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        fake_clip = types.SimpleNamespace(from_pretrained=lambda name: types.SimpleNamespace(vision_model=StubBackbone()))
        with patch.dict('sys.modules', {'transformers': types.SimpleNamespace(CLIPModel=fake_clip)}):
            models = importlib.import_module('captioning.models')
        self.models = models
        self.encoder = models.UniversalVisionEncoder(embed_dim=32, num_heads=4)
        self.encoder.eval()
        self.prompt = torch.randn(2, 20, 512)
        self.mask = torch.ones(2, 20, dtype=torch.long)
        self.mask[:, -2:] = 0
        self.cached = self.encoder.feature_extractor.tokens.expand(2, -1, -1).half()

    def test_same_output_without_backbone_call(self):
        with torch.no_grad():
            direct = self.encoder(torch.zeros(2, 3, 224, 224), self.prompt, self.mask)
            calls = self.encoder.feature_extractor.calls
            cached = self.encoder(self.cached, self.prompt, self.mask)
        self.assertEqual(self.encoder.feature_extractor.calls, calls)
        for a, b in zip(direct, cached):
            # Expanded vs contiguous inputs may use different FP32 GEMM kernels.
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(cached[1][:, -2:], torch.zeros(2, 2, 32)))

    def test_cached_path_keeps_projection_attention_and_gate_trainable(self):
        self.encoder.train()
        visual, prompt = self.encoder(self.cached, self.prompt, self.mask)
        (visual.sum() + prompt[..., 0].sum()).backward()
        for layer in [self.encoder.vis_projection, self.encoder.prompt_projection[0],
                      self.encoder.prompt_visual_gate, self.encoder.prompt_to_visual_attn]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))
        self.assertIsNone(self.encoder.feature_extractor.weight.grad)
        self.assertEqual(self.encoder.feature_extractor.calls, 0)

    def test_decoder_accepts_cached_tokens_for_five_captions_per_image(self):
        decoder = self.models.CaptionDecoder(vocab_size=16, d_model=32, nhead=4, num_layers=1)
        model = self.models.ImageCaptioningModel(self.encoder, decoder).eval()
        targets = torch.randint(0, 16, (10, 5))
        with torch.no_grad():
            cached = model(self.cached, self.prompt, self.mask, targets)
            self.assertEqual(self.encoder.feature_extractor.calls, 0)
            direct = model(torch.zeros(2, 3, 224, 224), self.prompt, self.mask, targets)
        self.assertEqual(cached.shape, (10, 5, 16))
        torch.testing.assert_close(cached, direct, rtol=1e-5, atol=1e-6)

    def test_wrong_cached_shape_fails(self):
        with self.assertRaisesRegex(ValueError, '197, 768'):
            self.encoder(torch.zeros(2, 1, 768), self.prompt, self.mask)

    def test_cached_only_encoder_does_not_load_or_require_backbone(self):
        with patch.object(self.models.CLIPModel, 'from_pretrained',
                          side_effect=AssertionError('must not download CLIP')):
            encoder = self.models.UniversalVisionEncoder(
                embed_dim=32, num_heads=4, load_backbone=False).eval()
        self.assertIsNone(encoder.feature_extractor)
        with torch.no_grad():
            visual, prompt = encoder(self.cached, self.prompt, self.mask)
        self.assertEqual(visual.shape, (2, 197, 32))
        self.assertEqual(prompt.shape, (2, 20, 32))
        with self.assertRaisesRegex(RuntimeError, 'backbone was not loaded'):
            encoder.extract_visual_features(torch.zeros(1, 3, 224, 224))

    def test_qformer_compresses_visual_memory_and_keeps_gradients(self):
        encoder = self.models.UniversalVisionEncoder(
            embed_dim=32,
            num_heads=4,
            load_backbone=False,
            visual_adapter='qformer',
            num_visual_queries=8,
            qformer_layers=2,
        )
        visual, prompt = encoder(self.cached, self.prompt, self.mask)
        self.assertEqual(visual.shape, (2, 8, 32))
        self.assertEqual(prompt.shape, (2, 20, 32))
        (visual[..., 0].sum() + prompt[..., 0].sum()).backward()
        self.assertIsNotNone(encoder.qformer.query_tokens.grad)
        self.assertGreater(encoder.qformer.query_tokens.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
