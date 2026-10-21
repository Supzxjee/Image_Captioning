"""Synthetic HDF5 cache round-trip, run on Kaggle before extraction/staging."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from extract_hidden import arrays
from hidden_data import validate_row


class Checks(unittest.TestCase):
    def test_empty_and_selected_row_round_trip_preserve_identity_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'features.h5'
            rows = [dict(coco_id=10, image_id=2), dict(coco_id=20, image_id=9)]
            with h5py.File(file, 'w') as f:
                arrays(f, rows)
                f['valid'][1, 0] = True
                f['features'][1, 0] = np.arange(1024, dtype=np.float16)
                f['pairs'][1, 0] = [0, 1]; f['native_rows'][1, 0] = 5
                f['labels'][1, 0] = [0, 3]; f['confidence'][1, 0] = [.8, .9]
                f['boxes'][1, 0] = [[0, 0, .5, 1], [.4, .3, .9, .8]]
                f.attrs.update(complete=True, completed_rows=2)
            with h5py.File(file) as f:
                self.assertEqual(f['image_ids'][:].tolist(), [10, 20])
                self.assertEqual(f['dataset_indices'][:].tolist(), [2, 9])
                self.assertTrue(f.attrs['complete'])
                for i in range(2):
                    validate_row(*(f[k][i] for k in ('features', 'valid', 'pairs', 'native_rows', 'labels', 'confidence', 'boxes')))
                np.testing.assert_array_equal(f['features'][1, 0], np.arange(1024, dtype=np.float16))


if __name__ == '__main__':
    unittest.main(verbosity=2)
