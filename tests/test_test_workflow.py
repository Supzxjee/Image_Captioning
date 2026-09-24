import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from captioning.config import Config, post_train_test_config
from captioning.cli import main, parse_args
from captioning.metrics import score_files


class TestWorkflowTests(unittest.TestCase):
    def test_post_train_uses_new_checkpoint_and_full_test(self):
        original = Config(checkpoint='warm_start.pth', split='val', limit=50,
                          test_after_train=True, visual_cache=['cache.h5'])
        test = post_train_test_config(original)
        self.assertEqual((test.mode, test.checkpoint, test.split, test.limit),
                         ('evaluate', '', 'test', 0))
        self.assertEqual(test.visual_cache, ['cache.h5'])
        self.assertEqual(original.checkpoint, 'warm_start.pth')
        self.assertFalse(test.test_after_train)

    def test_cache_train_then_online_test_preserves_reference_profile(self):
        cfg = Config(visual_cache=['train.h5'], test_after_train=True,
                     test_visual_source='images', visual_cache_id_key='karpathy_id',
                     visual_preprocessing='bicubic', visual_precision='amp-fp16')
        test = post_train_test_config(cfg)
        self.assertEqual(test.visual_cache, [])
        self.assertEqual(cfg.visual_cache, ['train.h5'])
        self.assertEqual((test.visual_preprocessing, test.visual_precision), ('bicubic', 'amp-fp16'))

    def test_cli_flag_and_validation(self):
        self.assertTrue(parse_args(['--test-after-train']).test_after_train)
        with self.assertRaises(ValueError):
            Config(mode='predict', test_after_train=True)
        with self.assertRaises(ValueError):
            Config(mode='metrics')
        qformer = parse_args(['--visual-adapter', 'qformer', '--num-visual-queries', '16',
                              '--qformer-layers', '3'])
        self.assertEqual((qformer.visual_adapter, qformer.num_visual_queries,
                          qformer.qformer_layers), ('qformer', 16, 3))
        with self.assertRaises(ValueError):
            Config(visual_adapter='qformer', alignment_weight=0.02,
                   region_targets_path='regions.pt')
        itc = parse_args(['--visual-adapter', 'qformer', '--itc-weight', '0.1',
                          '--caption-embedding-cache-path', 'captions.h5'])
        self.assertEqual((itc.itc_weight, itc.itc_temperature), (0.1, 0.07))
        with self.assertRaises(ValueError):
            Config(itc_weight=0.1, caption_embedding_cache_path='captions.h5')
        with self.assertRaises(ValueError):
            Config(visual_adapter='qformer', itc_weight=0.1)

    def test_metrics_cli_does_not_load_pipeline(self):
        with patch('captioning.metrics.score_files') as score:
            main(['--mode', 'metrics', '--predictions', 'p.json', '--ground-truth', 'g.json'])
        score.assert_called_once_with('p.json', 'g.json', None)

    def test_saved_files_can_be_scored_without_regeneration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            folder = Path(tmp)
            p, g, out = folder/'pred.json', folder/'gt.json', folder/'metrics.json'
            p.write_text(json.dumps([{'image_id': 7, 'caption': 'a bicycle'}]))
            g.write_text(json.dumps({'annotations': [{'image_id': 7, 'caption': 'a bike'}]}))
            before = (p.read_bytes(), g.read_bytes())
            with patch('captioning.metrics.compute_coco_metrics', return_value={'CIDEr': 1.179}) as score:
                self.assertEqual(score_files(p, g, out), {'CIDEr': 1.179})
                score.assert_called_once_with(p, g)
            self.assertEqual(json.loads(out.read_text()), {'CIDEr': 1.179})
            self.assertEqual((p.read_bytes(), g.read_bytes()), before)
            # A failed scorer must leave reusable input files intact.
            with patch('captioning.metrics.compute_coco_metrics', side_effect=ModuleNotFoundError('pycocoevalcap')):
                with self.assertRaises(ModuleNotFoundError):
                    score_files(p, g, out)
            self.assertEqual((p.read_bytes(), g.read_bytes()), before)
            p.write_text(json.dumps([{'image_id': 8, 'caption': 'a bicycle'}]))
            with patch('captioning.metrics.compute_coco_metrics') as score:
                with self.assertRaisesRegex(ValueError, 'match exactly'):
                    score_files(p, g, out)
                score.assert_not_called()


if __name__ == '__main__':
    unittest.main()
