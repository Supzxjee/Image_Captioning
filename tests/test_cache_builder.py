import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from captioning.cache_builder import select_part, write_feature_batches
from captioning.visual_cache import VisualCache


class CacheBuilderTests(unittest.TestCase):
    def test_parts_cover_every_image_without_overlap(self):
        source = list(range(123287))
        groups = [select_part(source, i, 4) for i in range(1, 5)]
        self.assertEqual(sum(groups, []), source)
        self.assertLess(max(map(len, groups)) * 197 * 768 * 2, 18_000_000_000)
        with self.assertRaises(ValueError):
            select_part(source, 0, 4)

    def test_writer_preserves_row_ids_and_fp16_features(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / 'cache.h5'
            records = [dict(coco_id=i, eval_id=j, filename=f'COCO_val2014_{i:012d}.jpg',
                            filepath='val2014', split=s)
                       for j, (i, s) in enumerate([(42, 'train'), (7, 'val'), (99, 'test')])]
            array = np.random.default_rng(42).normal(size=(3, 197, 768)).astype(np.float32)
            write_feature_batches(path, records, [array[:2], array[2:]], {'compute_dtype': 'float32'})
            with h5py.File(path, 'r') as f:
                self.assertTrue(f.attrs['complete'])
                self.assertEqual(f['features'].dtype, np.float16)
                self.assertEqual(list(f['imgids'][:]), [42, 7, 99])
                self.assertEqual(list(f['split'].asstr()[:]), ['train', 'val', 'test'])
            cache = VisualCache([str(path)])
            try:
                np.testing.assert_array_equal(cache.read(7), array[1].astype(np.float16).astype(np.float32))
            finally:
                cache.close()
            with self.assertRaises(FileExistsError):
                write_feature_batches(path, records, [array], {})

    def test_incomplete_output_is_rejected_by_reader(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / 'incomplete.h5'
            records = [dict(coco_id=42, eval_id=0, filename='COCO_val2014_000000000042.jpg',
                            filepath='val2014', split='test')]
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                write_feature_batches(path, records, [], {})
            with self.assertRaisesRegex(ValueError, 'not completed'):
                VisualCache([str(path)])


if __name__ == '__main__':
    unittest.main()
