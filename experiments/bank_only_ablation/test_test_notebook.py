"""Offline TEST locks/routing/source tests: no Torch, network or metric scorers."""
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import math
import symtable
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from make_test_notebook import literal_assignments

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeApi:
    def build_kaggle_client(self):
        raise AssertionError('Network must be mocked')

    def kernels_output(self, source, path):
        owner, slug, version = source.split('/')
        with self.build_kaggle_client() as client:
            client.kernels.kernels_api_client.list_kernel_session_output(SimpleNamespace(user_name=owner,kernel_slug=slug,version_label='v'+version))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = read(HERE/'M2_ONLY_BANK_ALIGNMENT_V245_TEST5000_API_2GPU.ipynb')
        cls.setup = ''.join(cls.nb['cells'][1]['source'])
        cls.header = cls.setup.split('\nSOURCE_PAYLOADS = ',1)[0]
        cls.execution = ''.join(cls.nb['cells'][3]['source'])
        cls.values = literal_assignments(cls.setup)
        cls.payloads = cls.values['SOURCE_PAYLOADS']

    def data_module(self):
        helper = ModuleType('real_bank_data')
        helper.read, helper.save, helper.sha = read, save, sha
        shared = ModuleType('test199_data')
        shared.lock = lambda: copy.deepcopy(json.loads(self.payloads['TEST199_LOCK.json']))
        env = {}
        tree = ast.parse(self.payloads['test199_data.py'])
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='runtime_policy')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'metadata_runtime','exec'),env)
        shared.runtime_policy = env['runtime_policy']
        spec = importlib.util.spec_from_file_location('baseline_metadata',HERE/'bank_only_test_data.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'real_bank_data':helper,'test199_data':shared}):
            spec.loader.exec_module(module)
        original_read = module.read
        sealed = json.loads(self.payloads['M2_ONLY_TRAINING_LOCK.json'])
        module.read = lambda p: sealed if Path(p).name=='M2_ONLY_TRAINING_LOCK.json' else original_read(p)
        return module

    def test_compile_hashes_closure_and_missing_globals(self):
        sources = {}
        for i,c in enumerate(self.nb['cells']):
            if c['cell_type']=='code':
                compile(''.join(c['source']),f'cell{i}','exec')
                self.assertEqual(c['outputs'],[])
                self.assertIsNone(c['execution_count'])
        for name,code in self.payloads.items():
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):
                sources[Path(name).stem]=ast.parse(code,filename=name)
        local={p.stem for folder in (ROOT/'experiments/clip_patch_baseline',ROOT/'experiments/csa_paper',HERE) for p in folder.glob('*.py')}
        pending=['run_csa_per_caption_test','bank_only_test_data','bank_only_model','bank_only_inference_guard','test_test199'];seen=set()
        while pending:
            name=pending.pop()
            if name in seen:continue
            self.assertIn(name,sources);seen.add(name)
            tree=copy.deepcopy(sources[name])
            if name=='bank_only_model':
                # The byte-identical model carries a train-only lazy guard, unused by TEST.
                tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='contract_guard')]
            for node in ast.walk(tree):
                names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
                for target in names:
                    if target in local:self.assertIn(target,sources);pending.append(target)
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'notebook','exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__','__builtins__'}
        def check(scope):
            for s in scope.get_symbols():
                if s.is_global() and s.is_referenced():self.assertIn(s.get_name(),bound)
            for child in scope.get_children():check(child)
        check(table)

    def test_exact_effective_api_route_no_stale_overwrite(self):
        nodes=[n for c in self.nb['cells'] if c['cell_type']=='code' for n in ast.parse(''.join(c['source'])).body
               if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROUTING_PREFLIGHT' for t in n.targets)]
        self.assertEqual(len(nodes),1)
        code=ast.literal_eval(nodes[0].value)
        self.assertEqual(self.values['SOURCE_VERSION'],'ducanh2403/2-vi-cai-kia-qua-nhieu/245')
        packages={name:ModuleType(name) for name in ('kaggle','kaggle.api','kaggle.api.kaggle_api_extended')}
        packages['kaggle.api.kaggle_api_extended'].KaggleApi=FakeApi
        with patch.dict(sys.modules,packages),contextlib.redirect_stdout(io.StringIO()):
            exec(code,{'SOURCE_VERSION':self.values['SOURCE_VERSION']})
        self.assertIn("r'(^|/)best[.]bin$'",self.execution)
        self.assertEqual(self.values['SECRET_NAMES'],('codex','KAGGLE_API_TOKEN'))

    def test_locks_and_exact_file_selection_reject_changed_inputs(self):
        d=self.data_module();gold=d.lock()
        self.assertEqual((gold['epoch'],gold['updates'],gold['seed']),(9,159399,42))
        self.assertEqual(len(gold['expected_test_ids']),5000)
        for key,value in [('source_version',d.VERSION.replace('/245','/246')),('checkpoint_sha256','0'*64),
                          ('beam_width',4),('max_new_tokens',21),('reranking',True),('epoch',10),('seed',43)]:
            wrong=copy.deepcopy(gold);wrong[key]=value
            with self.assertRaises(ValueError):d.validate_lock(wrong)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);best=root/'best.bin';best.write_bytes(b'fixture')
            with patch.object(d,'CHECKPOINT_SHA',sha(best)):
                self.assertEqual(d.checkpoint(root),best)
                best.write_bytes(b'wrong')
                with self.assertRaises(ValueError):d.checkpoint(root)
        for build in ('2.11.0+cpu','2.12.0+cu130'):
            with self.assertRaises(RuntimeError):d.runtime_policy('2.11.0+cu128',build)

    def test_checkpoint_metadata_acceptance_and_rejections(self):
        d=self.data_module();sealed=json.loads(self.payloads['M2_ONLY_TRAINING_LOCK.json'])
        state=dict(epoch=10,batch_cursor=0,phase='train',updates=159399,caption_events=5100723,
                   best_epoch=9,best_CIDEr=sealed['best_VALID_CIDEr'],history=[dict(epoch=i) for i in range(1,10)])
        identity=dict(method=d.METHOD,seed=42,epochs=10,global_batch=32,dropout=.2,
                      decoder_layers=3,semantic_conditioning=True,image_specific_prompts=False,sharing=False,
                      sampler='image_grouped_42_plus_epoch',
                      source=dict(commit=d.lock()['training_commit'],sha256=copy.deepcopy(sealed['source_sha256'])))
        weights={f'core.decoder.0.fixture{i}':SimpleNamespace(shape=(1,)) for i in range(156)}
        for prefix in ('project','noun_agg','verb_agg','object1','object2','relation','head'):
            weights['core.'+prefix+'.fixture']=SimpleNamespace(shape=(1,))
        weights['core.embedding.weight']=SimpleNamespace(shape=(10000,512))
        weights['core.nouns']=SimpleNamespace(shape=(5269,512))
        weights['core.verbs']=SimpleNamespace(shape=(1594,512))
        ck=dict(state=state,identity=identity,model=weights)
        self.assertEqual(d.validate_checkpoint_state(ck)['epoch'],9)
        for group,key,value in [('state','updates',177110),('identity','seed',43),('identity','method','old_baseline'),('identity','decoder_layers',4),('identity','semantic_conditioning',False),('identity','image_specific_prompts',True)]:
            wrong=copy.deepcopy(ck);wrong[group][key]=value
            with self.assertRaises(ValueError):d.validate_checkpoint_state(wrong)
        wrong=copy.deepcopy(ck);wrong['identity']['source']['sha256']['bank_only_model.py']='0'*64
        with self.assertRaises(ValueError):d.validate_checkpoint_state(wrong)
        wrong=copy.deepcopy(ck);wrong['model']['object_prompt.fixture']=wrong['model'].pop('core.decoder.0.fixture0')
        with self.assertRaises(ValueError):d.validate_checkpoint_state(wrong)
        wrong=copy.deepcopy(ck);wrong['model']['core.nouns'].shape=(1,512)
        with self.assertRaises(ValueError):d.validate_checkpoint_state(wrong)

    def test_training_model_and_scorers_unchanged_no_training(self):
        training=literal_assignments(''.join(read(HERE/'M2_ONLY_BANK_ALIGNMENT_SEED42_XE10_2GPU.ipynb')['cells'][1]['source']))['SOURCE_PAYLOADS']
        for name in ('bank_only_model.py','csa_patch_model.py','bank_layers.py','semantic_patch_model.py'):
            self.assertEqual(self.payloads[name],training[name])
        parent=read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_SEED43_TEST5000_API_2GPU.ipynb')
        original=None
        for node in ast.walk(ast.parse(''.join(parent['cells'][3]['source']))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
                try:code=ast.literal_eval(node.args[0])
                except (TypeError,ValueError):continue
                if isinstance(code,str) and 'def checked_weights(cfg)' in code:original=code
        self.assertIsNotNone(original)
        functions=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        before,after=functions(original),functions(self.payloads['run_csa_per_caption_test.py'])
        for name in ('score','tokenize','inputs'):self.assertEqual(before[name],after[name])
        for bad in ('backward(', 'torch.optim', 'optimizer.step', 'model.train('):
            self.assertNotIn(bad,self.payloads['run_csa_per_caption_test.py'])
        self.assertNotIn('PROMPT_TEST_ROOT',self.header)
        self.assertIn('bank_only=True,semantic=False',self.header)
        self.assertIn("save(out/f'beams_rank{rank}.json',beams)",self.payloads['run_csa_per_caption_test.py'])
        worker=self.payloads['run_csa_per_caption_test.py']
        self.assertIn("BankOnlyCaptioner(len(words),ck['model']['core.nouns'],ck['model']['core.verbs'])",worker)
        self.assertIn('inference_guard(model,x,memory,rank,out)',worker)
        self.assertNotIn('VisualOnlyCaptioner',worker)
        guard=self.payloads['bank_only_inference_guard.py']
        for anchor in ('count != 49239288', 'torch.cat(model.core.encode(patches, valid), 1)',
                       'model.encode(patches, poisoned)', 'atol=2e-6', 'M3_absent=True'):
            self.assertIn(anchor,guard)
        self.assertNotIn('contract_guard(',worker)

    def test_all_eight_metrics_required_and_no_inference_for_combine(self):
        tree=ast.parse(self.payloads['run_csa_per_caption_test.py'])
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='combine')
        d=self.data_module()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);save(out/'scoring_lock.json',dict(files_sha256={}));save(out/'chair_counts.json',{});save(out/'environment.json',{})
            parts=dict(basic=dict(BLEU_1=.7,BLEU_4=.3,CIDEr=1.1,ROUGE_L=.5),meteor=dict(METEOR=.2),spice=dict(SPICE=.15),chair=dict(CHAIRs=.05,CHAIRi=.03,Recall=.4))
            for mode,metrics in parts.items():save(out/(mode+'.json'),dict(images=5000,metrics=metrics,input_sha256={}))
            env=dict(read=read,save=save,sha=sha,math=math,contracts=lambda cfg:d,
                     inputs=lambda cfg:(out,[],[],[dict(NLL_sum=6,NLL_tokens=3,runtime_policy={})]))
            exec(compile(ast.Module(body=[function],type_ignores=[]),'combine_fixture','exec'),env)
            with contextlib.redirect_stdout(io.StringIO()):env['combine'](dict(output=str(out)))
            self.assertEqual(read(out/'result.json')['status'],'M2_ONLY_BANK_ALIGNMENT_TEST5000_COMPLETE')
            self.assertEqual(read(out/'result.json')['metrics']['NLL'],2.)
            save(out/'spice.json',dict(images=5000,metrics={},input_sha256={}))
            with self.assertRaises(ValueError):env['combine'](dict(output=str(out)))
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pycocoevalcap',sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)
