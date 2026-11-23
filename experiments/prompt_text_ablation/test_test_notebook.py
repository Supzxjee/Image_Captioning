"""Frozen zero-text TEST static/mock checks. No Torch, network or scorers."""
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import symtable
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from make_test_notebook import literals

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):Path(p).write_text(json.dumps(obj),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class FakeApi:
    def build_kaggle_client(self):raise AssertionError('Network must be mocked')
    def kernels_output(self,source,path):
        owner,slug,version=source.split('/')
        with self.build_kaggle_client() as client:
            client.kernels.kernels_api_client.list_kernel_session_output(SimpleNamespace(user_name=owner,kernel_slug=slug,version_label='v'+version))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=read(HERE/'FULL_ZERO_PROMPT_TEXT_V193_TEST5000_API_2GPU.ipynb')
        cls.setup=''.join(cls.nb['cells'][1]['source']);cls.run_source=''.join(cls.nb['cells'][3]['source'])
        cls.values=literals(cls.setup);cls.payloads=cls.values['SOURCE_PAYLOADS']
        cls.worker=cls.payloads['run_csa_per_caption_test.py'];cls.data=cls.payloads['csa_per_caption_test_data.py']

    def metadata(self):
        original=json.loads(self.payloads['TEST199_LOCK.json']);sealed=json.loads(self.payloads['ZERO_TEXT_TRAINING_LOCK.json'])
        env=dict(Path=Path,HERE=Path('fixture'),shared=SimpleNamespace(lock=lambda:copy.deepcopy(original)),
                 read=lambda path:sealed,sha=sha)
        for node in ast.parse(self.data).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CHECKPOINT_SHA','VERSION'):
                env[node.targets[0].id]=ast.literal_eval(node.value)
            if isinstance(node,ast.FunctionDef) and node.name in ('lock','validate_lock','checkpoint','runtime_policy','validate_checkpoint_state'):
                exec(compile(ast.Module(body=[node],type_ignores=[]),'metadata_only','exec'),env)
        return env

    def test_compile_hashes_closure_and_missing_globals(self):
        sources={}
        for i,c in enumerate(self.nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec');self.assertEqual(c['outputs'],[])
        for name,code in self.payloads.items():
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):sources[Path(name).stem]=ast.parse(code)
        local={p.stem for folder in (HERE,ROOT/'experiments/csa_paper',ROOT/'experiments/clip_patch_baseline') for p in folder.glob('*.py')}
        pending=['run_csa_per_caption_test','csa_per_caption_test_data','zero_prompt_model','test_test199','test_semantic_test'];seen=set()
        while pending:
            name=pending.pop()
            if name in seen:continue
            self.assertIn(name,sources);seen.add(name)
            tree=sources[name]
            if name=='zero_prompt_model':
                # The byte-identical training source also defines a train-only guard.
                # TEST imports only ZeroTextCaptioner/zero_prompt_text and never calls it.
                self.assertFalse('contract_guard(' in self.worker)
                tree=ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='contract_guard')],type_ignores=[])
            for node in ast.walk(tree):
                names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
                for target in names:
                    if target in local:self.assertIn(target,sources);pending.append(target)
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'notebook','exec');bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__','__builtins__'}
        def inspect(t):
            for s in t.get_symbols():
                if s.is_global() and s.is_referenced():self.assertIn(s.get_name(),bound)
            for c in t.get_children():inspect(c)
        inspect(table)

    def test_exact_api_route_and_source_version(self):
        nodes=[n for c in self.nb['cells'] if c['cell_type']=='code' for n in ast.parse(''.join(c['source'])).body
               if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROUTING_PREFLIGHT' for t in n.targets)]
        self.assertEqual(len(nodes),1);self.assertEqual(self.values['SOURCE_VERSION'],'ducanh3/notebook7bc83c4c76/193')
        packages={n:ModuleType(n) for n in ('kaggle','kaggle.api','kaggle.api.kaggle_api_extended')}
        packages['kaggle.api.kaggle_api_extended'].KaggleApi=FakeApi
        with patch.dict(sys.modules,packages),contextlib.redirect_stdout(io.StringIO()):exec(ast.literal_eval(nodes[0].value),{'SOURCE_VERSION':self.values['SOURCE_VERSION']})
        self.assertTrue("r'(^|/)best[.]bin$'" in self.run_source)
        self.assertEqual(self.values['SECRET_NAMES'],('codex','KAGGLE_API_TOKEN'))

    def test_zero_checkpoint_contract_rejects_wrong_epoch_mode_source(self):
        env=self.metadata();locked=env['lock']();sealed=json.loads(self.payloads['ZERO_TEXT_TRAINING_LOCK.json'])
        self.assertEqual((locked['epoch'],locked['updates'],locked['seed'],locked['prompt_text_mode']),(8,141688,42,'zero'))
        self.assertEqual(len(locked['expected_test_ids']),5000)
        state=dict(epoch=9,batch_cursor=0,phase='train',updates=141688,caption_events=4533976,best_epoch=8,
                   history=[dict(epoch=i) for i in range(1,9)],best_CIDEr=sealed['best_VALID_CIDEr'])
        identity=dict(source=dict(commit=locked['training_commit'],sha256=copy.deepcopy(sealed['source_sha256'])),
                      method='full_zero_prompt_text_xe10_v1',epochs=10,global_batch=32,seed=42,prompt_text_mode='zero')
        ck=dict(state=state,identity=identity,model={'core.nouns':object(),'core.verbs':object()})
        self.assertEqual(env['validate_checkpoint_state'](ck),locked)
        for group,key,value in [('state','updates',177110),('state','best_epoch',10),('identity','seed',43),('identity','prompt_text_mode','normal'),('identity','method','csa_structure_patch_prompt_grouped_per_caption_accelerate_v1')]:
            wrong=copy.deepcopy(ck);wrong[group][key]=value
            with self.assertRaises(ValueError):env['validate_checkpoint_state'](wrong)
        wrong=copy.deepcopy(ck);wrong['identity']['source']['sha256']['zero_prompt_model.py']='0'*64
        with self.assertRaises(ValueError):env['validate_checkpoint_state'](wrong)
        for key,value in [('beam_width',4),('reranking',True),('prompt_text_mode','normal'),('source_version','ducanh3/notebook7bc83c4c76/194')]:
            wrong=copy.deepcopy(locked);wrong[key]=value
            with self.assertRaises(ValueError):env['validate_lock'](wrong)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'best.bin';p.write_bytes(b'fixture weights')
            env['CHECKPOINT_SHA']=sha(p);self.assertEqual(env['checkpoint'](td),p)
            p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):env['checkpoint'](td)

    def test_zero_model_sources_factory_and_inference_hooks(self):
        training=literals(''.join(read(HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb')['cells'][1]['source']))['SOURCE_PAYLOADS']
        for name in ('zero_prompt_model.py','csa_patch_model.py','bank_layers.py','semantic_patch_model.py'):
            self.assertEqual(self.payloads[name],training[name])
        calls=[n.func.id for n in ast.walk(ast.parse(self.worker)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('ZeroTextCaptioner'),1)
        self.assertNotIn('CSAPatchCaptioner',calls);self.assertNotIn('PatchCaptioner',calls)
        for expression in ("cfg.get('prompt_text_mode')!='zero'",'torch.full_like(prompts[key]', 'torch.testing.assert_close(memory,perturbed',
                           'ROI_geometry_masks_unchanged=True',"save(out/f'beams_rank{rank}.json',beams)"):
            self.assertTrue(expression in self.worker,msg=expression)
        self.assertTrue('PROMPT_TEST_ROOT' in self.setup.split('\nSOURCE_PAYLOADS=',1)[0])
        self.assertTrue("prompt_text_mode='zero',semantic=True" in self.setup.split('\nSOURCE_PAYLOADS=',1)[0])

    def test_scoring_unchanged_and_no_training(self):
        parent=read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_SEED43_TEST5000_API_2GPU.ipynb');original=None
        for node in ast.walk(ast.parse(''.join(parent['cells'][3]['source']))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
                try:value=ast.literal_eval(node.args[0])
                except (TypeError,ValueError):continue
                if isinstance(value,str) and 'def checked_weights(cfg)' in value:original=value
        self.assertIsNotNone(original)
        functions=lambda source:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        a,b=functions(original),functions(self.worker)
        for name in ('score','tokenize','inputs'):self.assertEqual(a[name],b[name])
        for bad in ('backward(', 'torch.optim', 'optimizer.step', 'model.train('):self.assertFalse(bad in self.worker)
        self.assertTrue('if set(metrics)!=required' in self.worker)
        self.assertTrue("'SPICE','CHAIRs','CHAIRi','Recall'" in self.worker)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)

    def test_all_metrics_required_and_output_stays_zero_mode(self):
        node=next(n for n in ast.parse(self.worker).body if isinstance(n,ast.FunctionDef) and n.name=='combine')
        meta=self.metadata();d=SimpleNamespace(CHECKPOINT_SHA=meta['CHECKPOINT_SHA'],lock=meta['lock'])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);save(out/'scoring_lock.json',dict(files_sha256={}));save(out/'chair_counts.json',{});save(out/'environment.json',{})
            phases=dict(basic=dict(BLEU_1=.7,BLEU_4=.3,CIDEr=1.2,ROUGE_L=.5),meteor=dict(METEOR=.2),spice=dict(SPICE=.2),chair=dict(CHAIRs=.05,CHAIRi=.03,Recall=.4))
            for mode,metrics in phases.items():save(out/(mode+'.json'),dict(images=5000,metrics=metrics,input_sha256={}))
            env=dict(read=read,save=save,sha=sha,math=math,contracts=lambda cfg:d,
                     inputs=lambda cfg:(out,[],[],[dict(NLL_sum=6,NLL_tokens=3,runtime_policy={})]))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'combine_fixture','exec'),env)
            with contextlib.redirect_stdout(io.StringIO()):env['combine'](dict(output=str(out)))
            result=read(out/'result.json')
            self.assertEqual(result['status'],'FULL_ZERO_PROMPT_TEXT_TEST5000_COMPLETE')
            self.assertEqual(result['prompt_text_mode'],'zero');self.assertEqual(result['epoch'],8)
            self.assertEqual(result['metrics']['NLL'],2.)
            save(out/'spice.json',dict(images=5000,metrics={},input_sha256={}))
            with self.assertRaises(ValueError):env['combine'](dict(output=str(out)))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
