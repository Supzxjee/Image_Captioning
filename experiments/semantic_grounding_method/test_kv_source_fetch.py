"""Boundary checks for pinned source caching; never performs network or ML work."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import kv_data
import prepare_kv_cpu


class SourceChecks(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent/'review'
        self.base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='kv-source-test-', dir=self.base)).resolve()
        assert self.root.is_relative_to(self.base.resolve())
        self.raw = b'{"image":"000000000001.jpg"}\n'
        source = self.root/'experiments/relation_frames_inventory/sources'
        source.mkdir(parents=True)
        spec = dict(url='https://example.invalid/pinned.jsonl', bytes=len(self.raw),
                    sha256=hashlib.sha256(self.raw).hexdigest())
        (source/'source_receipt.json').write_text(json.dumps(dict(files=dict(train=spec))))
        self.work = self.root/'work'
        self.work.mkdir()
        self.root_patch = patch.object(kv_data, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        assert self.root.is_relative_to(self.base.resolve())
        shutil.rmtree(self.root)

    def test_verified_cache_does_not_fetch(self):
        (self.work/'train.jsonl').write_bytes(self.raw)
        with patch.object(prepare_kv_cpu.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            rows, _ = prepare_kv_cpu.source_rows(self.work)
        self.assertEqual(rows['train'][0]['image'], '000000000001.jpg')

    def test_changed_cache_fails_before_parsing(self):
        (self.work/'train.jsonl').write_bytes(self.raw+b' ')
        with patch.object(prepare_kv_cpu.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                prepare_kv_cpu.source_rows(self.work)

    def test_download_is_bounded_and_verified(self):
        with patch.object(prepare_kv_cpu.urllib.request, 'urlopen', return_value=io.BytesIO(self.raw)) as fetch:
            rows, _ = prepare_kv_cpu.source_rows(self.work)
        self.assertEqual(len(rows['train']), 1)
        self.assertEqual(fetch.call_args.kwargs['timeout'], 60)
        self.assertEqual((self.work/'train.jsonl').read_bytes(), self.raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
