"""Host checks use NumPy fixtures/static/mocks only; no local Torch/model calls."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from make_speed_notebook import HERE,NAME,SHA
from make_notebook import literals
import test_notebook as previous


class ArrayTensor:
    def __init__(self,value):self.value=np.asarray(value)
    def to(self,device):return self
    def index_select(self,dim,index):return ArrayTensor(np.take(self.value,index.value,axis=dim))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((HERE/NAME).read_text(encoding='utf-8'))
        cls.setup=''.join(cls.nb['cells'][1]['source']);cls.values=literals(cls.setup)
        cls.payloads=cls.values['SOURCE_PAYLOADS'];cls.main=''.join(cls.nb['cells'][3]['source'])

    def test_syntax_hash_global_and_import_closure(self):
        previous.Checks.test_cells_payload_hashes_and_global_dependencies(self)
        previous.Checks.test_dependency_closure_and_no_local_torch(self)
        self.assertEqual(self.values['WALL_HOURS'],1)
        self.assertEqual(self.values['CHECKPOINT_SHA'],SHA)
        self.assertEqual(self.values['SOURCE_VERSION'],'ducanh3/notebook7bc83c4c76/199')

    def test_vectorized_batch_matches_original_order_dtype_and_values(self):
        source=self.payloads['full_epoch_data.py']
        klass=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FullTrain')
        method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
        ns=dict(np=np);exec(compile(ast.Module(body=[method],type_ignores=[]),'<original batch>','exec'),ns)
        candidate=previous.load_module('speed_candidate_host',HERE/'speed_candidate.py')
        fake=ModuleType('torch');fake.from_numpy=ArrayTensor
        fake.cat=lambda values: ArrayTensor(np.concatenate([v.value for v in values],axis=0))
        class Context:
            def __init__(self,part):self.part=part;self.calls=0
            def images(self,offsets,device):
                self.calls+=1;offsets=np.asarray(offsets)
                return dict(value=ArrayTensor((offsets+self.part*100)[:,None].astype(np.float32)),
                    valid=ArrayTensor((offsets%2==0)[:,None]))
        data=SimpleNamespace(stops=np.array([3,6,9]),
            patches=[np.arange(i*18,(i+1)*18,dtype=np.float16).reshape(3,2,3) for i in range(3)],
            ids=[np.arange(i*3,(i+1)*3)+1000 for i in range(3)],context=[Context(i) for i in range(3)])
        with patch.dict(sys.modules,{'torch':fake}):
            for rows in ([0,1,2,0],[2,3,0,8],[6,5,6,0],[8,0,0,0]):
                a,pa,ia=ns['batch'](data,rows,'fixture')
                b,pb,ib=candidate.batch_rows(data,rows,'fixture')
                np.testing.assert_array_equal(a.value,b.value);self.assertEqual(a.value.dtype,b.value.dtype)
                self.assertEqual(ia,ib);self.assertEqual(pa.keys(),pb.keys())
                for key in pa:
                    np.testing.assert_array_equal(pa[key].value,pb[key].value)
                    self.assertEqual(pa[key].value.dtype,pb[key].value.dtype)
            for rows in ([],[-1],[9]):
                with self.assertRaises(ValueError):candidate.batch_rows(data,rows,'fixture')
            for c in data.context:c.calls=0
            candidate.batch_rows(data,[0,1,2,0],'fixture')
            self.assertEqual([c.calls for c in data.context],[1,0,0])
        self.assertNotIn('torch',sys.modules)

    def test_original_model_reward_and_effective_recipe_preserved(self):
        old=json.loads((HERE/'V2_SOURCE_MANIFEST.json').read_text())
        for name in ('csa_patch_model.py','bank_layers.py','memory_fusion.py','semantic_patch_model.py',
                     'semantic_data.py','beam_policy.py','full_epoch_data.py','captioning/scst.py','scst_cache.py'):
            self.assertEqual(hashlib.sha256(self.payloads[name].encode()).hexdigest(),old['sha256'][name])
        worker=self.payloads['run_speed_probe.py']
        for text in ("optimizer.load_state_dict(ck['optimizer'])","torch.set_rng_state(state['cpu'])",
                     "torch.cuda.set_rng_state(state['cuda'],device)","np.random.set_state(state['numpy'])",
                     'range(4)','weights).sum()*(2/real)','find_unused_parameters=True',
                     'lr=4e-6,betas=(.9,.99),eps=1e-8,weight_decay=0','error_if_nonfinite=True'):
            self.assertIn(text,worker)
        for forbidden in ('autocast','GradScaler','torch.compile','torch.save','cross_entropy'):
            self.assertNotIn(forbidden,worker)

    def test_synchronization_change_rejects_nonfinite_before_loss_and_keeps_EOS(self):
        source=self.payloads['speed_candidate.py']
        self.assertIn('finite.logical_and_(torch.isfinite(logits).all())',source)
        self.assertIn("if not finite:raise FloatingPointError('Nonfinite beam logits')",source)
        self.assertIn('if done.all():break',source)
        self.assertIn('logp.detach()',source);self.assertIn('chosen=logp.gather',source)
        self.assertIn('if not torch.isfinite(history).all()',source)
        worker=self.payloads['run_speed_probe.py']
        self.assertIn('Candidate silently accepted nonfinite logits',worker)
        self.assertIn("for key in ('logps','loss'):torch.testing.assert_close",worker)
        self.assertIn("for key in ('gradients','parameters')",worker)
        self.assertIn("for key in ('cpu_rng','cuda_rng')",worker)
        self.assertLess(worker.index("save(out/f'equivalence_rank"),worker.index('blocks=[]'))

    def test_ABBA_schedule_shards_tail_measurements_and_no_production_output(self):
        source=self.payloads['run_speed_probe.py']
        self.assertIn('enumerate((False,True,True,False))',source)
        self.assertIn('POSITIONS=(64,96,40000,40032,80000,80032,113280)',source)
        self.assertIn('for position in (0,32)',source)
        self.assertEqual(2+4*(2+7),38)
        for text in ('torch.cuda.Event(enable_timing=True)','torch.cuda.synchronize(device)',
                     "'beam_forward_span'","'CPU_reward_and_transfer'","'backward_span'","'optimizer_span'",
                     'if position==113280:continue','statistics.median','max(r[\'blocks\']'):
            self.assertIn(text,source)
        self.assertNotIn('torch.save',source)
        self.assertNotIn("'worker','--config'",self.main)
        self.assertNotIn("'train',CONFIG",self.main)
        self.assertIn('all_optimizer_updates_discarded=True',self.main)

    def test_mock_API_full_state_before_setup(self):
        start=self.main.index('    sys.path.insert(0,str(SOURCE))')
        end=self.main.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'",start)
        block=textwrap.dedent(self.main[start:end]);calls=[]
        fake_secret=ModuleType('kaggle_secrets');fake_secret.UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:'fixture-token')
        fake_checkpoint=ModuleType('checkpoint_input')
        def resolver(path,digest,report):
            self.assertEqual(digest,SHA);result=Path(path)/'checkpoint.bin';self.assertTrue(result.is_file());return result
        fake_checkpoint.resolve_mounted=resolver
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            def run(args,timeout,env=None):
                args=list(map(str,args));calls.append((args,env))
                if 'kernels' in args:(Path(args[args.index('-p')+1])/'checkpoint.bin').write_bytes(b'fixture')
            scope=dict(sys=sys,os=__import__('os'),Path=Path,json=json,SOURCE=root,OUTPUT=out,CLI=root/'cli',
                CHECKPOINT_SHA=SHA,SOURCE_VERSION=self.values['SOURCE_VERSION'],SEED=43,
                SECRET_NAMES=('codex','KAGGLE_API_TOKEN'),TOKENS=[],CHECKPOINT_ROOT='',cfg={},write_config=lambda:None,
                run=run,ROUTING_PREFLIGHT=self.values['ROUTING_PREFLIGHT'])
            with patch.dict(sys.modules,{'kaggle_secrets':fake_secret,'checkpoint_input':fake_checkpoint}),patch.object(sys,'path',list(sys.path)):
                exec(block,scope)
        args,env=next(c for c in calls if 'kernels' in c[0])
        self.assertEqual(args[args.index('output')+1],'ducanh3/notebook7bc83c4c76/199')
        self.assertEqual(args[args.index('--file-pattern')+1],'(^|/)beam_scst_epoch1_seed43_v179_v2_run1/checkpoint[.]bin$')
        self.assertNotIn('fixture-token',' '.join(args));self.assertNotIn('KAGGLE_KEY',env)
        self.assertLess(start,self.main.index("'nltk==3.9.2'"))


if __name__=='__main__':unittest.main(verbosity=2)
