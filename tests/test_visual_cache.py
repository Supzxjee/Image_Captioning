import multiprocessing
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from captioning.visual_cache import VisualCache, coco_image_id


def read_in_worker(cache, queue):
    try:
        queue.put(float(cache.read(42)[0, 0]))
    finally:
        cache.close()


class VisualCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'a.h5'
        self.make_file(self.path, [42, 7])
        self.cache = VisualCache([str(self.path)])

    def make_file(self, path, ids, shape=None):
        with h5py.File(path, 'w') as f:
            f['imgids'] = np.array(ids, dtype=np.int64)
            array = np.zeros(shape or (len(ids), 197, 768), dtype=np.float16)
            for i in range(len(ids)):
                array[i] = i + 1
            f['features'] = array

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_coco_id_is_not_evaluation_row_id(self):
        self.assertEqual(coco_image_id('COCO_val2014_000000000042.jpg'), 42)
        self.assertEqual(self.cache.read(42)[0, 0], 1)
        self.assertEqual(self.cache.read(7)[0, 0], 2)
        self.assertEqual(self.cache.read(7).dtype, np.float32)
        with self.assertRaises(ValueError):
            coco_image_id('invalid.jpg')

    def test_directory_merges_shards_and_checks_coverage(self):
        self.make_file(self.root / 'b.h5', [99])
        cache = VisualCache([str(self.root)])
        try:
            cache.require_ids([7, 42, 99], 'test')
            with self.assertRaisesRegex(ValueError, 'missing'):
                cache.require_ids([7, 100], 'test')
            with self.assertRaises(KeyError):
                cache.read(100)
        finally:
            cache.close()

    def test_invalid_shape_and_duplicate_ids_rejected(self):
        bad = self.root / 'bad.h5'
        self.make_file(bad, [8], (1, 768))
        with self.assertRaisesRegex(ValueError, '197, 768'):
            VisualCache([str(bad)])
        duplicate = self.root / 'duplicate.h5'
        self.make_file(duplicate, [42])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            VisualCache([str(self.path), str(duplicate)])

    def test_nonfinite_features_rejected(self):
        with h5py.File(self.path, 'r+') as f:
            f['features'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.cache.read(42)

    def test_spawn_after_parent_has_opened_file(self):
        self.cache.read(42)
        # HDF5 handles must not be serialized into worker processes.
        restored = pickle.loads(pickle.dumps(self.cache))
        self.assertFalse(restored._handles)
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        worker = ctx.Process(target=read_in_worker, args=(restored, queue))
        worker.start()
        worker.join(30)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            self.fail('Worker did not finish reading cache')
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(queue.get(timeout=5), 1)
        queue.close()
        restored.close()


if __name__ == '__main__':
    unittest.main()
