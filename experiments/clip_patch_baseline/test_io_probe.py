"""Pure NumPy/mock IO checks; no Kaggle, H5 files or neural runtime."""
import ast
import hashlib
from pathlib import Path
import queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import io_probe as probe

class Checks(unittest.TestCase):
    def test_same_global_batches_partition_across_four_readers(self):
        ev=np.random.default_rng(43).permutation(5000)[:2048];assignment=probe.tasks(ev)
        self.assertEqual(len(assignment),4)
        flat=[e for worker in assignment for batch in worker for e in batch]
        self.assertEqual(sorted(flat),sorted(ev.tolist()));self.assertTrue(all(len(w)==32 and all(len(b)==16 for b in w) for w in assignment))
        np.testing.assert_array_equal(assignment[0][0],ev[:16]);np.testing.assert_array_equal(assignment[3][0],ev[48:64])
        with self.assertRaises(ValueError):probe.tasks(np.zeros(2048,dtype=np.int64))
    def test_physical_stage_is_lossless_and_local_lru(self):
        rng=np.random.default_rng(1);raw=rng.normal(size=(4,197,768)).astype(np.float16);requests=[]
        class Features:
            def __getitem__(self,rows):requests.append(rows);return raw[rows]
        class Handle:
            def __getitem__(self,key):return Features()
            def close(self):pass
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=base/'fake.h5';source.write_bytes(b'metadata fixture, not H5')
            # Image order deliberately differs from physical row order.
            data=dict(caption_image=np.array([0,1,2,0]),locations=np.array([[0,3],[0,0],[0,2]]))
            meta=dict(sources=[dict(path=str(source),bytes=source.stat().st_size)])
            with patch.object(probe,'load_stage',return_value=(data,[],meta)),patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *a,**k:Handle())}):
                info=probe.stage_sample(base,base/'local',np.arange(4),time.monotonic()+30)
            self.assertEqual(requests,[[0,2,3]]);self.assertEqual(info['unique_TRAIN_images'],3)
            reader=probe.LocalReader(base/'local')
            try:
                for image,row in [(0,3),(1,0),(2,2)]:np.testing.assert_array_equal(reader.get(image),raw[row,1:])
                self.assertIs(reader.get(0),reader.get(0))
            finally:reader.close()
    def test_worker_parity_and_nonTRAIN_rejection(self):
        class Reader:
            def __init__(self,*a):pass
            def get(self,i):return np.full((196,768),i,np.float16)
            def close(self):pass
        class Go:
            def wait(self,*a):return True
        reports=[]
        with patch.object(probe,'load_stage',return_value=(dict(caption_image=np.arange(16)),[],{})),patch.object(probe,'FeatureReader',Reader),patch.object(probe,'LocalReader',Reader):
            for mode in ('mounted','local'):
                q=queue.Queue();probe.run_worker(0,mode,'stage','local',[list(range(16))],q,Go(),time.monotonic()+30);self.assertEqual(q.get()['kind'],'ready');reports.append(q.get())
        self.assertEqual(reports[0]['feature_stream_sha256'],reports[1]['feature_stream_sha256']);self.assertEqual(reports[0]['events'],16)
        with patch.object(probe,'load_stage',return_value=(dict(caption_image=np.array([probe.TRAIN_IMAGES])),[],{})),patch.object(probe,'FeatureReader',Reader):
            q=queue.Queue();probe.run_worker(0,'mounted','stage','local',[[0]],q,Go(),time.monotonic()+30);q.get();self.assertIn('NonTRAIN',q.get()['error'])
    def test_source_parse_and_no_neural_or_api_calls(self):
        source=Path(__file__).with_name('io_probe.py').read_text(encoding='utf-8');tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(x.split('.')[0] in ('torch','transformers','kaggle','requests') for x in imports))
        self.assertNotIn('beam_search',source);self.assertIn('feature_streams_equal=True',source)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
