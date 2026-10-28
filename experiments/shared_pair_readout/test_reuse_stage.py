"""Synthetic file/hash tests only, no HDF5 reader, Torch or real cache values."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from reuse_stage import *


class Checks(unittest.TestCase):
    def fixture(self,root):
        source=root/'old';source.mkdir();out=root/'new'
        for n in PAYLOADS:(source/n).write_bytes(('fixture:'+n).encode())
        sources={n:'a'*64 for n in PREPARATION_SOURCES}
        save(source/'protocol.json',dict(policy=POLICY,source_assets=sources,source_commit='old-model-version'))
        expected=dict(cache=FILES,checkpoint=CHECKPOINT_SHA,data=DATA_SHA,detections=DETECTION_SHA,legacy=LEGACY_SHA,dimensions=DIMENSIONS_SHA)
        receipt=dict(status='PREPARATION_COMPLETE',input_hashes=expected,payload_sha256={n:sha(source/n) for n in PAYLOADS},
            tensor_content_sha256='b'*64,seconds=504.)
        save(source/'preparation.json',receipt);(source/'failure_rank0.json').write_text('old attempt preserved')
        cfg=dict(previous_prepared=str(source),output=str(out),policy=POLICY,source_assets={**sources,
            'experiments/shared_pair_readout/readout_model.py':'c'*64},source_commit='corrected-model-version')
        return source,out,cfg

    def test_reuse_keeps_old_attempt_and_new_model_provenance(self):
        with tempfile.TemporaryDirectory(prefix='reuse-test-',dir=HERE) as temp:
            source,out,cfg=self.fixture(Path(temp));old=sha(source/'protocol.json')
            with contextlib.redirect_stdout(io.StringIO()):reuse(cfg)
            self.assertEqual(old,sha(source/'protocol.json'));self.assertTrue((source/'failure_rank0.json').is_file())
            self.assertFalse((out/'failure_rank0.json').exists());self.assertEqual(json.loads((out/'protocol.json').read_bytes()),cfg)
            receipt=json.loads((out/'preparation.json').read_bytes())
            self.assertFalse(receipt['reused_preparation']['model_updates_resumed'])
            for n,h in receipt['payload_sha256'].items():self.assertEqual(sha(out/n),h)

    def test_changed_payload_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='reuse-test-',dir=HERE) as temp:
            source,out,cfg=self.fixture(Path(temp));(source/'inputs.h5').write_bytes(b'changed')
            with self.assertRaises(ValueError):reuse(cfg)
            self.assertFalse(out.exists())

    def test_changed_preparation_source_rejected(self):
        with tempfile.TemporaryDirectory(prefix='reuse-test-',dir=HERE) as temp:
            source,out,cfg=self.fixture(Path(temp));cfg['source_assets'][PREPARATION_SOURCES[0]]='d'*64
            with self.assertRaises(ValueError):reuse(cfg)
            self.assertFalse(out.exists())

    def test_copy_fallback_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory(prefix='reuse-test-',dir=HERE) as temp:
            source,out,cfg=self.fixture(Path(temp))
            with patch('reuse_stage.os.link',side_effect=OSError('fixture no hardlinks')),contextlib.redirect_stdout(io.StringIO()):reuse(cfg)
            methods=json.loads((out/'reuse_lineage.json').read_bytes())['methods']
            self.assertEqual(set(methods.values()),{'verified_copy'})
            with self.assertRaises(FileExistsError):reuse(cfg)


if __name__=='__main__':unittest.main(verbosity=2)
