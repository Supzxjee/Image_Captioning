"""Host metadata/NumPy/mocks only; never import Torch or a caption metric scorer."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import symtable
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    """Just the arithmetic surface needed to check the saved-score objective."""
    def __init__(self, value): self.value = np.asarray(value, dtype=np.float64)
    @property
    def ndim(self): return self.value.ndim
    @property
    def shape(self): return self.value.shape
    def detach(self): return self
    def mean(self, axis, keepdim=False): return Tensor(self.value.mean(axis, keepdims=keepdim))
    def __sub__(self, other): return Tensor(self.value-other.value)
    def __mul__(self, other): return Tensor(self.value*other.value)
    def __neg__(self): return Tensor(-self.value)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((HERE/'BEAM_SCST_FULL_EPOCH1_VALID5000_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup = ''.join(cls.nb['cells'][1]['source'])
        cls.values = {}
        for n in ast.parse(cls.setup).body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
                try:cls.values[n.targets[0].id] = ast.literal_eval(n.value)
                except (ValueError,TypeError):pass
        cls.payloads = cls.values['SOURCE_PAYLOADS']
        cls.policy = load_module('beam_policy_host', HERE/'beam_policy.py')

    def test_cells_payload_hashes_and_global_dependencies(self):
        for c in self.nb['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']),'<cell>','exec')
                self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
        for name,code in self.payloads.items():
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):ast.parse(code,filename=name)
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'notebook','exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__'}
        def check(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced():self.assertIn(symbol.get_name(),bound)
            for child in scope.get_children():check(child)
        check(table)

    def test_dependency_closure_and_no_local_torch(self):
        local={str(Path(n).with_suffix('')).replace('\\','/').replace('/','.') for n in self.payloads if n.endswith('.py')}
        known={'csa_patch_model','memory_fusion','bank_layers','semantic_patch_model','semantic_data',
               'captioning.scst','scst_cache','scst_dataset_mount','csa_per_caption_test_data','test199_data',
               'patch_data','patch_beam','real_bank_data','run_beam_epoch','beam_policy','full_epoch_data'}
        for name,code in self.payloads.items():
            if not name.endswith('.py'):continue
            for n in ast.walk(ast.parse(code)):
                target=n.module if isinstance(n,ast.ImportFrom) else None
                if target in known:self.assertIn(target,local)
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pycocoevalcap',sys.modules)

    def test_one_epoch_coverage_no_replacement_and_tail(self):
        real=[]
        for start in range(0,113287,32):
            this=[]
            for micro in range(4):
                for rank in (0,1):
                    rows,keep=self.policy.rank_microbatch(start,rank,micro)
                    self.assertEqual(len(rows),4)
                    this.extend(i for i,w in zip(rows,keep) if w)
            self.assertEqual(this,list(range(start,min(113287,start+32))))
            real.extend(this)
        self.assertEqual(real,list(range(113287)))
        self.assertEqual(math.ceil(len(real)/32),3541)
        self.assertEqual(len(this),7)
        with self.assertRaises(ValueError):self.policy.rank_microbatch(0,2,0)

    def test_objective_mean_baseline_fixed_denominator_and_gradient_weights(self):
        rng=np.random.default_rng(42)
        logps=rng.normal(size=(32,5,20))
        rewards=rng.normal(size=(32,5))
        actual=self.policy.image_losses(Tensor(logps),Tensor(rewards)).value
        expected=np.array([-sum((rewards[i,j]-sum(rewards[i])/5)*sum(logps[i,j])/20 for j in range(5))/5 for i in range(32)])
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        np.testing.assert_allclose(self.policy.image_losses(Tensor(logps),Tensor(rewards+100)).value,actual,atol=1e-14)
        for n in (7,32):
            grads=np.zeros_like(logps)
            for rank in (0,1):
                for micro in range(4):
                    rows,keep=self.policy.rank_microbatch(0,rank,micro,n)
                    for row,weight in zip(rows,keep):
                        if weight:grads[row]+=-(rewards[row]-rewards[row].mean())[:,None]/(5*20)*(2/n)/2
            ref=np.zeros_like(logps);ref[:n]=-(rewards[:n]-rewards[:n].mean(-1,keepdims=True))[:,:,None]/(5*20*n)
            np.testing.assert_allclose(grads,ref,atol=1e-15)
        equal=self.policy.image_losses(Tensor(logps),Tensor(np.ones((32,5)))).value
        np.testing.assert_array_equal(equal,np.zeros(32))

    def test_resume_validation_rejects_foreign_source_recipe_and_cursor(self):
        source=self.payloads['run_beam_epoch.py']
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='validate_resume')
        scope=dict(math=math,IMAGES=113287,GLOBAL_BATCH=32,SOURCE_SHA='a'*64)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<metadata>','exec'),scope)
        identity={'recipe':'fixed','torch':'same','cache':'sealed'}
        r=dict(cursor_images=3200,updates=100,identity=identity,source_sha256='a'*64)
        scope['validate_resume'](r,identity)
        scope['validate_resume'](dict(r,cursor_images=113287,updates=3541),identity)
        for changes in ({'cursor_images':3201},{'updates':99},{'source_sha256':'b'*64},{'identity':{'recipe':'sampled_greedy'}}):
            with self.assertRaises(ValueError):scope['validate_resume'](dict(r,**changes),identity)

    def test_real_guards_training_recipe_and_validation_scope(self):
        source=self.payloads['run_beam_epoch.py']
        for required in ('model.to(device).train()', 'mechanism_guard(device)', 'real_eval_paths_match=True',
                         'update(IMAGES-7)', 'cfg.get(\'resume_root\')', 'optimizer.load_state_dict',
                         "torch.cuda.set_rng_state(state['cuda'], device)", 'if updates % 100 == 0',
                         'float(\'inf\')', "'preflight':", 'torch.optim.Adam(wrapped.parameters(), lr=4e-6'):
            self.assertIn(required,source)
        self.assertNotIn('multinomial',source)
        self.assertNotIn('cross_entropy',source)
        policy=self.payloads['beam_policy.py']
        self.assertIn('logp.detach()',policy)
        self.assertIn('chosen = logp.gather',policy)
        self.assertIn('history = torch.cat',policy)
        runner=''.join(self.nb['cells'][3]['source'])
        self.assertLess(runner.index("'preflight',CONFIG"),runner.index("'train',CONFIG"))
        self.assertIn("eval_update=3541",runner)
        self.assertNotIn("'test'",runner)
        self.assertIn("test_used=False",runner)
        self.assertEqual(self.values['CHECKPOINT_ROOT'],'/kaggle/input/datasets/ducanh3/upt-coco-full-cache')
        self.assertEqual(self.values['CACHE_ROOT'],'/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1')
        self.assertEqual(self.values['RESUME_ROOT'],'')
        self.assertEqual(self.values['WALL_HOURS'],11.5)

    def test_cache_shard_provenance_and_model_bytes(self):
        p=json.loads(self.payloads['CACHE_PARTS.json'])
        self.assertEqual([(r['train_start'],r['train_stop']) for r in p['parts']],[(0,40000),(40000,80000),(80000,113287)])
        self.assertEqual([r['notebook_version'].split('/')[-1] for r in p['parts']],['82','83','84'])
        manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        for name in ('csa_patch_model.py','bank_layers.py','semantic_patch_model.py','semantic_data.py','patch_beam.py','captioning/scst.py'):
            self.assertEqual(hashlib.sha256(self.payloads[name].encode()).hexdigest(),manifest['origins'][name]['sha256'])
        cache=self.payloads['full_epoch_data.py']
        self.assertIn('target.symlink_to(source)',cache)
        self.assertIn("set(all_ids) & set(valid)",cache)
        self.assertIn('len(set(all_ids)) != 113287',cache)
        self.assertIn('patch_full_content_SHA_rehashed=False',cache)

    def test_nested_metadata_root_exists_before_original_part0_materializer(self):
        class StopHere(Exception): pass
        bank=ModuleType('real_bank_data')
        bank.read=lambda path: {'parts':[],'ids_sha256':'sealed'} if Path(path).name=='CACHE_PARTS.json' else {}
        bank.save=lambda *args:None
        bank.sha=lambda path:'sealed'
        mounting=ModuleType('scst_dataset_mount')
        def materialize(dataset,local,parts,ids):
            self.assertTrue(Path(local).parent.is_dir())
            self.assertFalse(Path(local).exists())
            raise StopHere
        mounting.materialize_part0=materialize
        cache=ModuleType('scst_cache');cache.mount=lambda *args:None
        semantic=ModuleType('semantic_data');semantic.ARRAYS={}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,
                {'real_bank_data':bank,'scst_dataset_mount':mounting,'scst_cache':cache,'semantic_data':semantic}):
            module=load_module('full_epoch_data_metadata_host',HERE/'full_epoch_data.py')
            cfg=dict(cache_root=temporary,output=temporary,metadata_root=str(Path(temporary)/'nested'/'metadata'))
            with self.assertRaises(StopHere):module.prepare(cfg)
        self.assertNotIn('torch',sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
