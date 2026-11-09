"""Lossless local-stage NumPy/mock checks; no real H5/model execution."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import full_data as d
import local_stage as l

@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);stage=root/'stage';out=root/'out';stage.mkdir();out.mkdir();reads=[]
        ids=np.array([91,7,33,44],np.int64);locations=np.array([[0,4],[1,1],[0,0],[1,3]],np.int32)
        sources=[];raws=[]
        for file in range(2):
            path=root/f'raw{file}.h5';path.write_bytes(b'mock metadata, not actual H5')
            raw=np.full((5,197,768),np.nan,np.float16)
            idrows=np.full(5,-1,np.int64)
            for image in np.flatnonzero(locations[:,0]==file):
                row=locations[image,1];raw[row]=image+1;raw[row,0]=19;raw[row,2,3]=-0.;idrows[row]=ids[image]
            raws.append(raw);sources.append(dict(path=str(path),bytes=path.stat().st_size,shape=list(raw.shape),dtype='float16'))
        arrays=dict(ids=ids,locations=locations,caption_image=np.arange(4,dtype=np.int32),sequences=np.ones((4,3),np.int32))
        for name,value in arrays.items():np.save(stage/(name+'.npy'),value)
        l.save(stage/'vocabulary.json',['<pad>','<bos>','<eos>','<unk>'])
        meta=dict(status='FULL_PATCH_METADATA_COMPLETE',train_caption_events=566747,test_features_used=False,feature_arrays_staged=False,sources=sources,files_sha256={p.name:l.sha(p) for p in stage.iterdir()})
        l.save(stage/'manifest.json',meta)
        class Features:
            def __init__(self,file):self.file=file;self.shape=raws[file].shape;self.dtype=raws[file].dtype
            def __getitem__(self,rows):
                rows=np.asarray(rows);reads.append((self.file,rows.tolist()))
                if not np.isfinite(raws[self.file][rows]).all():raise AssertionError('TEST/nonselected features read')
                return raws[self.file][rows]
        class Handle:
            def __init__(self,file):self.file=file
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def __getitem__(self,key):
                if key=='features':return Features(self.file)
                x=np.full(5,-1,np.int64)
                for i in np.flatnonzero(locations[:,0]==self.file):x[locations[i,1]]=ids[i]
                return x
        original=l.read
        golden={str(int(ids[i])):hashlib.sha256(raws[int(locations[i,0])][int(locations[i,1])].tobytes()).hexdigest() for i in (0,1)}
        def fake_read(p):return dict(raw_selected_row_sha256=golden) if Path(p).name=='SMOKE_SOURCE.json' else original(p)
        with patch.object(l,'TRAIN_IMAGES',3),patch.object(l,'VALID_IMAGES',1),patch.object(l,'read',side_effect=fake_read),patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda p,*a,**k:Handle(int(Path(p).stem[-1]))) }):
            yield dict(stage=str(stage),output=str(out)),root,raws,reads

class Checks(unittest.TestCase):
    def test_full_local_stage_bytes_mapping_and_no_TEST(self):
        with fixture() as (cfg,root,raw,reads):
            l.stage_local(cfg,time.monotonic()+60)
            meta=l.read(root/'stage/manifest.json');self.assertTrue(meta['feature_arrays_staged']);self.assertEqual(meta['local_stage']['smoke_raw_rows_replayed'],2)
            self.assertEqual(reads,[(0,[0,4]),(1,[1,3])]);self.assertEqual(np.load(root/'stage/local_rows.npy').tolist(),[1,2,0,3])
            reader=d.FeatureReader(root/'stage',capacity=1)
            try:
                for image,(file,row) in enumerate([[0,4],[1,1],[0,0],[1,3]]):
                    np.testing.assert_array_equal(reader.get(image).view(np.uint16),raw[file][row,1:].view(np.uint16))
                self.assertIs(reader.get(3),reader.get(3));self.assertLessEqual(len(reader.cache),1)
            finally:
                reader.close()
                for value in reader.data.values():value._mmap.close()
            self.assertEqual(reads,[(0,[0,4]),(1,[1,3])])
    def test_insufficient_disk_refuses_before_allocation(self):
        with fixture() as (cfg,root,raw,reads):
            with patch.object(l.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):l.stage_local(cfg,time.monotonic()+60)
            self.assertFalse((root/'stage/local_patches.npy').exists());self.assertEqual(reads,[])
    def test_deadline_never_seals_partial_stage(self):
        with fixture() as (cfg,root,raw,reads):
            with self.assertRaises(TimeoutError):l.stage_local(cfg,time.monotonic()-1)
            self.assertFalse(l.read(root/'stage/manifest.json')['feature_arrays_staged']);self.assertEqual(l.read(root/'out/local_stage_progress.json')['status'],'LOCAL_PATCH_STAGE_INCOMPLETE');self.assertEqual(reads,[])
    def test_changed_sealed_map_rejected(self):
        with fixture() as (cfg,root,raw,reads):
            l.stage_local(cfg,time.monotonic()+60)
            (root/'stage/local_rows.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):d.load_stage(root/'stage')
    def test_sources_parse_and_loader_only_change(self):
        here=Path(__file__).parent
        for name in ('local_stage.py','full_data.py','run_full.py','make_full_notebook.py'):ast.parse((here/name).read_text(encoding='utf-8'))
        self.assertNotIn('torch',sys.modules)
        code=(here/'local_stage.py').read_text(encoding='utf-8');self.assertNotIn('PatchCaptioner',code);self.assertIn('np.uint16',code)
        self.assertIn('do not fall back to mounted H5',(here/'run_full.py').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main(verbosity=2)
