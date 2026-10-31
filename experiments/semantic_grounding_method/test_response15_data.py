"""Metadata/mock checks only. Never imports or runs Torch/CLIP."""
import hashlib
import json
from pathlib import Path
import sys
import shutil
import uuid
import unittest
from response15_data import assets, checked_assets, contrast, safe_path, summary
import run_response15_probe


class Checks(unittest.TestCase):
    def test_imports_do_not_load_neural_modules(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)

    def test_matched_prefix_and_multiword_tail_without_eos(self):
        vocab = {'<bos>': 1, 'the': 4, 'cat': 5, 'is': 6, 'black': 7, 'and': 8, 'white': 9, 'brown': 10}
        a, b = contrast('the cat is black and white', 'the cat is brown', vocab)
        self.assertEqual(a, dict(ids=[1,4,5,6,7,8,9], start=3, count=3))
        self.assertEqual(b, dict(ids=[1,4,5,6,10], start=3, count=1))
        for x, y in [('the cat', 'the cat is black'), ('the cat is black', 'the cat is red')]:
            with self.assertRaises(ValueError):
                contrast(x, y, vocab)

    def test_direction_is_not_correctness_and_caption_change_is_not_truth(self):
        r = dict(changed_d_original=-4., changed_d_edited=-8., stable_d_original=2., stable_d_edited=1.,
                 caption_original='a cat', caption_edited='a dog')
        out = summary([r])
        self.assertEqual(out['changed_direction_correct'], 1)
        self.assertEqual(out['changed_both_ranked_correct'], 0)
        self.assertEqual(out['stable_both_ranked_correct'], 1)
        self.assertEqual(out['greedy_caption_text_changed'], 1)
        self.assertIn('NOT_AUTOMATICALLY_SCORED', out['caption_semantic_accuracy'])

    def test_asset_hashes_counts_and_traversal(self):
        parent = Path(__file__).resolve().parent / 'review'
        root = parent / ('response15-metadata-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.assertTrue(root.resolve().is_relative_to(parent.resolve()))
        self.addCleanup(shutil.rmtree, root)
        for bad in ('../outside', '/outside', 'C:/outside', 'images\\outside'):
            with self.assertRaises(ValueError):
                safe_path(root, bad)
        images = root / 'images'
        images.mkdir()
        files, pairs = {}, []
        for i in range(15):
            r = dict(vg_id=i, coco_id=i, human_confirmed=True)
            for side in ('original', 'edited'):
                name = f'images/{i}_{side}.jpg'
                content = (str(i)+side).encode()  # metadata fixture, intentionally not a JPEG
                (root / name).write_bytes(content)
                files[name] = hashlib.sha256(content).hexdigest()
                r[side+'_file'], r[side+'_sha256'] = name, files[name]
            pairs.append(r)
        marker = root / 'semantic_response_confirmed15_manifest.json'
        marker.write_text(json.dumps(dict(schema='semantic_response_confirmed15_assets_v1', pairs=pairs,
            files=files, vocab_sha256='fixture')), encoding='utf-8')
        receipt = dict(manifest_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(), vocab_sha256='fixture')
        checked_assets(root, receipt)
        located, _ = assets(root, root/'unused', receipt)
        self.assertEqual(located, root)
        (root / pairs[0]['original_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed image'):
            checked_assets(root, receipt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
