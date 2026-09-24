import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from captioning.caption_cache import CaptionEmbeddingCache


class CaptionEmbeddingCacheTests(unittest.TestCase):
    def make_cache(self, folder, complete=True):
        path = Path(folder) / 'captions.h5'
        values = np.arange(3 * 5 * 512, dtype=np.float32).reshape(3, 5, 512)
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('features', data=values.astype(np.float16), chunks=(1, 5, 512))
            handle.create_dataset('imgids', data=np.array([7, 11, 19], dtype=np.int64))
            handle.attrs['complete'] = complete
        return path, values

    def test_reads_by_coco_id(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path, values = self.make_cache(folder)
            cache = CaptionEmbeddingCache(path)
            cache.require_ids([7, 19], 'train')
            np.testing.assert_allclose(cache.read(11), values[1].astype(np.float16).astype(np.float32))
            cache.close()

    def test_missing_and_incomplete_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path, _ = self.make_cache(folder, complete=False)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                CaptionEmbeddingCache(path)
            path.unlink()
            path, _ = self.make_cache(folder, complete=True)
            cache = CaptionEmbeddingCache(path)
            with self.assertRaisesRegex(ValueError, 'missing'):
                cache.require_ids([999], 'train')


if __name__ == '__main__':
    unittest.main()
