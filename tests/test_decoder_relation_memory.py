import importlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from captioning.checkpoints import load_checkpoint
from captioning.config import Config


class DecoderRelationMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        with patch.dict('sys.modules', {'transformers': types.SimpleNamespace(CLIPModel=object)}):
            models = importlib.import_module('captioning.models')
        encoder = models.UniversalVisionEncoder(embed_dim=16, num_heads=4,
            load_backbone=False, visual_adapter='qformer', num_visual_queries=4,
            qformer_layers=1, grounded_relation_alignment=True, decoder_relation_memory=True)
        self.model = models.ImageCaptioningModel(encoder,
            models.CaptionDecoder(vocab_size=12, d_model=16, nhead=4, num_layers=1)).eval()
        self.args = (torch.randn(2, 197, 768), torch.randn(2, 5, 512), torch.ones(2, 5).long())
        self.kwargs = dict(relation_tokens=torch.randn(2, 2, 512),
            negative_relation_tokens=torch.randn(2, 2, 512),
            subject_boxes=torch.tensor([[[0, 0, .4, 1], [0, 0, .4, 1]]]*2),
            object_boxes=torch.tensor([[[.5, 0, 1, 1], [.5, 0, 1, 1]]]*2),
            relation_confidences=torch.tensor([[.8, .7], [.8, .7]]),
            relation_mask=torch.tensor([[1, 0], [0, 0]]))

    def test_shapes_mask_and_empty_row_decoder_equivalence(self):
        on_memory, on_mask = self.model.build_memory(*self.args, **self.kwargs)
        self.assertEqual(on_memory.shape, (2, 11, 16))
        self.assertEqual(on_mask[:, -2:].tolist(), [[False, True], [True, True]])
        self.assertTrue(torch.equal(on_memory[1, -2:], torch.zeros(2, 16)))
        tgt = torch.ones(2, 3).long()
        direct = self.model.decoder(tgt, on_memory, memory_key_padding_mask=on_mask)
        self.model.encoder.decoder_relation_memory = False
        legacy_memory, legacy_mask = self.model.build_memory(*self.args, **self.kwargs)
        legacy = self.model.decoder(tgt, legacy_memory, memory_key_padding_mask=legacy_mask)
        torch.testing.assert_close(direct[1], legacy[1], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(direct[0], legacy[0]))

    def test_gradient_and_visual_output_excludes_relation_slots(self):
        target = torch.ones(2, 3).long()
        logits, visual = self.model(*self.args, tgt=target, return_visual=True, **self.kwargs)
        self.assertEqual(visual.shape, (2, 4, 16))
        logits[0, 0, 0].backward()
        self.assertGreater(self.model.encoder.relation_alignment.pair_projection[0].weight.grad.abs().sum(), 0)

    def test_zero_confidence_masks_relation(self):
        self.kwargs['relation_confidences'].zero_()
        memory, mask = self.model.build_memory(*self.args, **self.kwargs)
        self.assertTrue(mask[:, -2:].all())
        self.assertEqual(memory[:, -2:].abs().sum().item(), 0)

    def test_checkpoint_evaluation_mismatch_is_not_silent(self):
        vocabulary = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        saved = dict(encoder_state_dict=self.model.encoder.state_dict(),
            decoder_state_dict=self.model.decoder.state_dict(), caption_word2idx=vocabulary,
            visual_adapter='qformer', num_visual_queries=4, qformer_layers=1,
            grounded_relation_alignment=True)  # Old checkpoint: no direct-memory flag.
        tokenizer = types.SimpleNamespace(word2idx=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'; torch.save(saved, path)
            with self.assertRaisesRegex(RuntimeError, 'decoder relation memory'):
                load_checkpoint(self.model, path, tokenizer)
            load_checkpoint(self.model, path, tokenizer, warm_start=True)

    def test_config_disallows_bypassing_verified_v3(self):
        with self.assertRaises(ValueError):
            Config(decoder_relation_memory=True)
        with self.assertRaises(ValueError):
            Config(decoder_relation_memory=True, grounded_relation_alignment=True,
                   grounded_relation_version='verified_v3', visual_adapter='qformer',
                   grounded_relation_cache_path='x.pt')


if __name__ == '__main__': unittest.main()
