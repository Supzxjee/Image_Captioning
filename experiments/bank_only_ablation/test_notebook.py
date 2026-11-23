"""Host static and mock checks only; never load Torch or run a model/scorer."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import symtable
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from make_notebook import literals, NAME, METHOD, PARENT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((HERE/NAME).read_text(encoding='utf-8'))
        cls.setup = ''.join(cls.nb['cells'][1]['source'])
        cls.values = literals(cls.setup)
        cls.sources = cls.values['SOURCE_PAYLOADS']
        cls.header = cls.setup.split('\nSOURCE_PAYLOADS = ', 1)[0]
        cls.run_source = ''.join(cls.nb['cells'][3]['source'])
        parent = json.loads(PARENT.read_text(encoding='utf-8'))
        cls.parent = literals(''.join(parent['cells'][1]['source']))['SOURCE_PAYLOADS']

    def test_compile_hashes_globals_and_dependency_closure(self):
        for i, cell in enumerate(self.nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell{i}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        self.assertEqual(len(self.sources), 42)
        parsed = {}
        for name, code in self.sources.items():
            self.assertEqual(hashlib.sha256(code.encode('utf-8')).hexdigest(), self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):
                parsed[Path(name).stem] = ast.parse(code, filename=name)
        combined = '\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table = symtable.symtable(combined, 'notebook', 'exec')
        bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
        bound |= set(vars(builtins)) | {'__name__', '__builtins__'}
        missing = set()
        def inspect(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in bound:
                    missing.add(symbol.get_name())
            for child in scope.get_children():
                inspect(child)
        inspect(table)
        self.assertEqual(missing, set())
        inventory = {p.stem for folder in (HERE, ROOT/'experiments/clip_patch_baseline',
                     ROOT/'experiments/csa_paper', PARENT.parent) for p in folder.glob('*.py')}
        pending = ['run_csa_full', 'score_csa_full', 'bank_only_model', 'bank_only_data',
                   'test_full_data', 'test_local_stage', 'test_semantic_data', 'test_csa_full_data', 'test_gradient_agreement']
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            self.assertIn(name, parsed)
            seen.add(name)
            for node in ast.walk(parsed[name]):
                imports = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else []
                for target in imports:
                    if target in inventory:
                        self.assertIn(target, parsed, msg=f'{name} imports missing {target}')
                        pending.append(target)
        self.assertGreater(len(seen), 20)
        self.assertNotIn('torch', sys.modules)

    def test_same_core_data_optimizer_selection_and_score(self):
        for name in ('bank_layers.py', 'csa_patch_model.py', 'full_data.py', 'csa_full_data.py',
                     'semantic_data.py', 'patch_beam.py', 'score_full.py'):
            self.assertEqual(self.sources[name], self.parent[name], msg=name)
        def nested(code, name):
            return ast.dump(next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef) and n.name==name))
        for name in ('validate', 'checkpoint', 'predict', 'memory_for'):
            self.assertEqual(nested(self.sources['run_csa_full.py'], name), nested(self.parent['run_csa_full.py'], name), msg=name)
        current = ast.parse(self.sources['run_csa_full.py'])
        prior = ast.parse(self.parent['run_csa_full.py'])
        loop = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.While))
        self.assertEqual(ast.dump(loop(current)), ast.dump(loop(prior)))
        text = self.sources['run_csa_full.py']
        for anchor in ("mixed_precision='no'", "global_batch=32,epochs=10", "point['CIDEr']>state['best_CIDEr']",
                       'torch.optim.Adam(model.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)',
                       "if ck['identity']!=identity", "model.load_state_dict(ck['model'],strict=True)"):
            self.assertTrue(anchor in text, msg=anchor)

    def test_configuration_scope_resume_and_runtime_guards(self):
        self.assertEqual((self.values['SEED'], self.values['RESUME_ROOT'], self.values['WALL_HOURS']), (42, '', 11.5))
        self.assertIn('semantic=False,bank_alignment=True', self.header)
        self.assertIn('for p in (VISUAL_CACHE,COCO_JSON,BANK_ROOT):', self.header)
        self.assertNotIn('PROMPT_ROOT', self.header)
        self.assertIn("identity.get('method')!=cfg['method']", self.header)
        runner = self.sources['run_csa_full.py']
        self.assertIn("stage.get('bank_alignment',{}).get('method')!='bank_only_v130'", runner)
        self.assertIn("schema='"+METHOD+"'", runner)
        self.assertIn('stage_banks(cfg,deadline)', runner)
        self.assertIn('BankOnlyContext as SemanticContext', runner)
        self.assertIn('BankOnlyContext as SemanticContext, contract_guard', self.sources['csa_full_runtime.py'])
        self.assertIn('contract_guard(model,len(words),cfg,rank,device,out)', self.sources['csa_full_runtime.py'])
        self.assertNotIn('visual_only_model', self.run_source)
        self.assertNotIn('raw.githubusercontent.com/Supzxjee', self.run_source)
        self.assertIn("SOURCE/'cuda_runtime_guard.py'", self.run_source)
        self.assertIn("'--nproc_per_node=2'", self.run_source)
        self.assertIn('bank_only_contract_rank0.json', self.sources['score_csa_full.py'])
        guard = self.sources['bank_only_model.py']
        for anchor in ('count != 49239288', 'full_count != 53183736', "torch.Generator(device='cpu').manual_seed(cfg['seed'])",
                       'reference.core.nouns.zero_()', 'sensitivity <= 1e-7', 'tensor_digest(model) != initial'):
            self.assertIn(anchor, guard)

    def test_mock_constructor_removes_only_M3_and_routes_both_M2_streams(self):
        calls = []
        vo, vr, valid = object(), object(), object()
        class Parent:
            def __init__(self, vocab, nouns, verbs):
                self.core = SimpleNamespace(encode=lambda x, v: (calls.append((x, v)) or (vo, vr)),
                    noun_agg=object(), verb_agg=object(), object1=object(), object2=object(), relation=object(),
                    nouns=nouns, verbs=verbs, decoder=object(), project=object(), embedding=object(), head=object())
                self.object_prompt, self.relation_prompt = object(), object()
        upstream, torch_mock = ModuleType('csa_patch_model'), ModuleType('torch')
        upstream.CSAPatchCaptioner = Parent
        torch_mock.bool = object()
        torch_mock.ones = lambda shape, **kw: valid
        spec = importlib.util.spec_from_file_location('bank_model_mock', HERE/'bank_only_model.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch':torch_mock, 'csa_patch_model':upstream}):
            spec.loader.exec_module(module)
            nouns, verbs = object(), object()
            model = module.BankOnlyCaptioner(10000, nouns, verbs)
            self.assertEqual(set(vars(model)), {'core'})
            self.assertIs(model.core.nouns, nouns)
            self.assertIs(model.core.verbs, verbs)
            self.assertTrue(all(hasattr(model.core, n) for n in ('noun_agg','verb_agg','object1','object2','relation','decoder')))
            x = SimpleNamespace(ndim=3, shape=(2,196,768), device='mock')
            for p in ({}, None, {'object_text':'poison', 'object_weights':'poison'}):
                self.assertEqual(model.streams(x,p), (vo,vr))
                self.assertEqual(calls[-1], (x,valid))
            with self.assertRaisesRegex(ValueError, '196'):
                model.streams(SimpleNamespace(ndim=3,shape=(2,197,768)), {})
            self.assertEqual(module.BankOnlyContext('path-must-not-be-read').images(None,None), {})
            self.assertEqual(module.BankOnlyContext(None).events(None,None,None), {})
        self.assertNotIn('torch', sys.modules)

    def test_factory_accepts_only_M2_only_seed42(self):
        code = self.sources['csa_full_runtime.py']
        node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name=='make_model')
        torch_mock, model_mock = ModuleType('torch'), ModuleType('bank_only_model')
        torch_mock.from_numpy = lambda x:x
        model_mock.BankOnlyCaptioner = lambda vocab,*banks:(vocab,banks)
        namespace = {'np':SimpleNamespace(load=lambda path,**kw:Path(path).name), 'Path':Path}
        cfg = dict(method=METHOD,semantic=False,bank_alignment=True,seed=42,stage='no-read')
        with patch.dict(sys.modules, {'torch':torch_mock,'bank_only_model':model_mock}):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'factory','exec'),namespace)
            factory=namespace['make_model']
            self.assertEqual(factory(10000,cfg),(10000,('sem_noun_bank.npy','sem_verb_bank.npy')))
            for changed in ({'method':'full_zero_prompt_text_xe10_v1'},{'semantic':True},{'bank_alignment':False},
                            {'seed':43},{'hybrid_sharing':True},{'semantic_strength_gate':True}):
                with self.assertRaises(ValueError):factory(10000,dict(cfg,**changed))

    def test_bank_stage_only_two_files_and_sealed_manifest(self):
        real_mock, sem_mock, np_mock = ModuleType('real_bank_data'), ModuleType('semantic_data'), ModuleType('numpy')
        real_mock.read = lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
        real_mock.save = lambda path,value:Path(path).write_text(json.dumps(value),encoding='utf-8')
        real_mock.sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        np_mock.save = lambda path,array:Path(path).write_text(str(array.shape),encoding='utf-8')
        sem_mock.fixed_banks = lambda root:({'noun':SimpleNamespace(shape=(5269,512)),'verb':SimpleNamespace(shape=(1594,512))},
            {'files':{'semantic_banks.npz':'mock_hash'}},{'model':'fixed_clip','revision':'fixed_revision'})
        spec=importlib.util.spec_from_file_location('bank_data_mock',HERE/'bank_only_data.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'real_bank_data':real_mock,'semantic_data':sem_mock,'numpy':np_mock}):
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);stage=root/'stage';out=root/'out';stage.mkdir();out.mkdir()
                cfg=dict(bank_alignment=True,semantic=False,bank_root='mock',stage=str(stage),output=str(out))
                manifest={'files_sha256':{'ids.npy':'prior'},'test_features_used':False}
                real_mock.save(stage/'manifest.json',manifest)
                module.stage_banks(cfg,float('inf'))
                staged=real_mock.read(stage/'manifest.json')
                self.assertEqual(set(p.name for p in stage.glob('*.npy')),{'sem_noun_bank.npy','sem_verb_bank.npy'})
                for n,digest in staged['bank_alignment']['files_sha256'].items():
                    self.assertEqual(real_mock.sha(stage/n),digest)
                    self.assertEqual(staged['files_sha256'][n],digest)
                self.assertFalse(staged['bank_alignment']['image_specific_prompts'])
                self.assertEqual(staged,real_mock.read(out/'stage_summary.json'))
                with self.assertRaisesRegex(ValueError,'fresh'):module.stage_banks(cfg,float('inf'))
                with self.assertRaisesRegex(ValueError,'M2-only'):module.stage_banks(dict(cfg,prompt_root='forbidden'),float('inf'))
                real_mock.save(stage/'manifest.json',manifest)
                with self.assertRaises(TimeoutError):module.stage_banks(cfg,0)
        self.assertNotIn('torch',sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
