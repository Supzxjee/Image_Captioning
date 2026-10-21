import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from package_cache_dataset import package, REQUIRED, sha


class Checks(unittest.TestCase):
    def test_copy_is_byte_identical_repeat_safe_and_changed_source_rejected(self):
        scratch = Path(__file__).parent / 'review'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder); source = root / 'input'; source.mkdir(); dest = root / 'output'
            for name in REQUIRED:
                (source / name).write_bytes(('synthetic fixture:' + name).encode())
            receipt = dict(status='CACHE_COMPLETE', schema='upt_hidden_pair_caption_cache_v1',
                           files={n: sha(source / n) for n in REQUIRED})
            (source / 'cache_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            pinned = sha(source / 'cache_receipt.json')
            package(source, dest, pinned)
            package(source, dest, pinned)
            self.assertEqual(len(list(dest.iterdir())), 7)
            for name in REQUIRED | {'cache_receipt.json'}:
                self.assertEqual((source / name).read_bytes(), (dest / name).read_bytes())
            (source / 'features_rank0.h5').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                package(source, dest, pinned)
            self.assertNotEqual((source / 'features_rank0.h5').read_bytes(), (dest / 'features_rank0.h5').read_bytes())

    def test_same_or_nested_paths_rejected(self):
        for dest in ('cache', 'cache/inside'):
            with self.assertRaises(ValueError):
                package(Path('cache'), Path(dest))


if __name__ == '__main__':
    unittest.main(verbosity=2)
