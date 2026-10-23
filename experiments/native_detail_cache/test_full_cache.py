"""Metadata/resume tests run locally; HDF5 tests run on Kaggle before extraction."""
import ast
import base64
import zlib
import copy
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
if (Path(__file__).resolve().parent.parent/'native_detail_benchmark').is_dir():
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'native_detail_benchmark'))
from cache_data import plan,spec_for,identity,validate_spec,verify_receipt,check_shard
from run_benchmark import digest
from run_full_cache import import_resume
from tile_data import tile_centers,save


def fixture():
    groups={};data=[]
    for group,count in [('train',10000),('val',5000)]:
        rows=[]
        for _ in range(count):
            i=len(data);name=f'COCO_train2014_{i:012d}.jpg';data.append(dict(filename=name,filepath='train2014'))
            rows.append(dict(image_id=i,coco_id=i,filename=name))
        groups[group]=rows
    return groups,data


class Metadata(unittest.TestCase):
    def test_locked_plan_coverage_and_gpu_balance(self):
        g,d=fixture();shards=plan(g,d)
        self.assertEqual(len(shards),30);self.assertEqual([sum(len(s['rows']) for s in shards if s['rank']==r) for r in range(2)],[7500,7500])
        self.assertEqual(len({r['coco_id'] for s in shards for r in s['rows']}),15000)
        self.assertEqual([s['name'] for s in shards][19:22],['train_019','val_000','val_001'])
        self.assertEqual(shards,plan(g,d))

    def test_cross_split_duplicate_and_wrong_mapping_fail(self):
        g,d=fixture();g['val'][0]=g['train'][0].copy()
        with self.assertRaises(ValueError):plan(g,d)
        g,d=fixture();g['train'][0]['coco_id']=99999
        with self.assertRaises(ValueError):plan(g,d)

    def test_resume_protocol_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            out=Path(tmp);save(out/'cache_spec.json',dict(schema='one'))
            validate_spec(out,dict(schema='one'))
            with self.assertRaises(ValueError):validate_spec(out,dict(schema='two'))
            self.assertEqual(json.loads((out/'cache_spec.json').read_text()),dict(schema='one'))

    def test_resume_rejects_payload_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            out=Path(tmp);p=out/'train_000.h5';p.write_bytes(b'original')
            record=dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p));verify_receipt(p,record)
            p.write_bytes(b'corrupt!')
            with self.assertRaises(ValueError):verify_receipt(p,record)

    def test_spec_pins_geometry_sources_and_full_order(self):
        g,d=fixture();shards=plan(g,d);a=spec_for(shards,dict(code='hash'),'groups','clip')
        b=spec_for(list(reversed(shards)),dict(code='hash'),'groups','clip')
        self.assertNotEqual(identity(a),identity(b));self.assertEqual(a['shape_per_condition'],[196,512]);self.assertEqual(a['images'],15000)
        b=copy.deepcopy(a);b['implementation_sha256']['code']='other';self.assertNotEqual(identity(a),identity(b))

    def test_python_and_notebook_sources_are_portable(self):
        root=Path(__file__).resolve().parent
        for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
        notebook=root/'NATIVE_DETAIL_CACHE_15K_2GPU.ipynb'
        if notebook.exists():
            self.assertLess(notebook.stat().st_size,1000000)
            for c in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
                if c['cell_type']=='code':
                    tree=ast.parse(''.join(c['source']));self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
                    for node in tree.body:
                        if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in node.targets):continue
                        packed=next(n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000)
                        embedded=json.loads(zlib.decompress(base64.b64decode(packed)))
                        for name,source in embedded.items():
                            path=root/name if (root/name).exists() else root.parent/'native_detail_benchmark'/name
                            if name.startswith('captioning/'):path=root.parents[1]/name
                            self.assertEqual(source,path.read_text(encoding='utf-8'),name)


@unittest.skipUnless(importlib.util.find_spec('h5py'),'HDF5 checks execute in Kaggle, not this metadata-only local runtime.')
class Hdf5(unittest.TestCase):
    def setUp(self):
        import h5py
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.out=Path(self.tmp.name)
        self.shard=dict(name='train_000',rank=0,rows=[dict(coco_id=i,image_id=i) for i in range(4)])
        self.spec=dict(schema='synthetic-only');self.models=['clip','projection'];self.path=self.out/'train_000.h5'
        save(self.out/'cache_spec.json',self.spec)
        with h5py.File(self.path,'w') as f:
            f.attrs.update(complete=True,benchmark_only=False,spec_sha256=identity(self.spec),rows_sha256=identity(self.shard['rows']),model_sha256=json.dumps(self.models),runtime_sha256='synthetic-runtime')
            f.create_dataset('coco_ids',data=np.arange(4));f.create_dataset('image_ids',data=np.arange(4));f.create_dataset('xy',data=tile_centers())
            for c in ('H','L'):f.create_dataset(c,data=np.zeros((4,196,512),dtype='f2'))

    def tearDown(self):self.tmp.cleanup()

    def test_valid_committed_shard_can_resume(self):
        record=check_shard(self.path,self.shard,self.spec,self.models);save(self.out/'train_000.json',record)
        target=self.out/'destination';target.mkdir()
        import_resume(self.out,target,[self.shard],self.spec)
        verify_receipt(target/'train_000.h5',record)
        self.assertEqual(check_shard(target/'train_000.h5',self.shard,self.spec,self.models),record)

    def test_partial_marker_rejected(self):
        import h5py
        with h5py.File(self.path,'r+') as f:f.attrs['complete']=False
        with self.assertRaises(ValueError):check_shard(self.path,self.shard,self.spec,self.models)

    def test_bad_feature_and_wrong_model_rejected(self):
        import h5py
        with self.assertRaises(ValueError):check_shard(self.path,self.shard,self.spec,['other','projection'])
        with self.assertRaises(ValueError):check_shard(self.path,self.shard,self.spec,self.models,'other-runtime')
        with h5py.File(self.path,'r+') as f:f['L'][1,2,3]=np.nan
        with self.assertRaises(ValueError):check_shard(self.path,self.shard,self.spec,self.models)

    def test_unreceipted_input_not_imported(self):
        target=self.out/'destination';target.mkdir()
        import_resume(self.out,target,[self.shard],self.spec)
        self.assertFalse((target/'train_000.h5').exists())


if __name__=='__main__':unittest.main(verbosity=2)
