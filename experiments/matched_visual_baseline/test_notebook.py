"""Host static/mock checks only. Do not import Torch or execute notebook cells."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import symtable
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def value_assignments(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((HERE/'BASELINE_MATCHED_PARALLEL3_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup = ''.join(cls.notebook['cells'][1]['source'])
        cls.header = cls.setup.split('\nSOURCE_PAYLOADS = ', 1)[0]
        cls.run_source = ''.join(cls.notebook['cells'][3]['source'])
        cls.values = value_assignments(cls.setup)
        cls.payloads = cls.values['SOURCE_PAYLOADS']

    def test_cells_sources_hashes_and_dependency_closure(self):
        self.assertEqual(len(self.payloads), 41)
        for index, cell in enumerate(self.notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell{index}', 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        sources = {}
        for name, code in self.payloads.items():
            self.assertEqual(hashlib.sha256(code.encode('utf-8')).hexdigest(), self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):
                sources[Path(name).stem] = ast.parse(code, filename=name)
        local_names = {p.stem for folder in (ROOT/'experiments/clip_patch_baseline', ROOT/'experiments/csa_paper', HERE) for p in folder.glob('*.py')}
        roots = ('run_csa_full', 'score_csa_full', 'visual_only_model', 'test_full_data',
                 'test_local_stage', 'test_semantic_data', 'test_csa_full_data', 'test_gradient_agreement')
        pending = list(roots)
        reached = set()
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            self.assertIn(name, sources)
            reached.add(name)
            for node in ast.walk(sources[name]):
                names = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                         else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
                for target in names:
                    if target in local_names:
                        self.assertIn(target, sources, msg=f'{name} imports unstaged {target}')
                        pending.append(target)
        self.assertGreater(len(reached), 20)

    def test_visual_only_recipe_and_resume_separation(self):
        self.assertEqual(self.values['SEED'], 42)
        self.assertEqual(self.values['RESUME_ROOT'], '')
        self.assertEqual(self.values['WALL_HOURS'], 11.5)
        self.assertNotIn('BANK_ROOT', self.header)
        self.assertNotIn('PROMPT_ROOT', self.header)
        self.assertIn('for p in (VISUAL_CACHE,COCO_JSON):', self.header)
        self.assertIn("semantic=False", self.header)
        self.assertIn("identity.get('method')!=cfg['method']", self.header)
        runner = self.payloads['run_csa_full.py']
        runtime = self.payloads['csa_full_runtime.py']
        self.assertIn("if cfg.get('semantic') or stage.get('semantic')", runner)
        self.assertIn("method='matched_visual_parallel3_xe10_v1'", runner)
        self.assertIn('decoder_layers=3,semantic_conditioning=False', runner)
        factory = ast.get_source_segment(runtime, next(n for n in ast.parse(runtime).body if isinstance(n, ast.FunctionDef) and n.name=='make_model'))
        self.assertNotIn('np.load', factory)
        self.assertIn('return VisualOnlyCaptioner(vocab)', factory)
        self.assertIn('contract_guard(model,len(words),cfg,rank,device,out)', runtime)
        self.assertNotIn('raw.githubusercontent.com/Supzxjee', self.run_source)
        self.assertIn("SOURCE/'cuda_runtime_guard.py'", self.run_source)
        self.assertIn("'--nproc_per_node=2'", self.run_source)

    def test_no_missing_notebook_global_names(self):
        combined = '\n'.join(''.join(c['source']) for c in self.notebook['cells'] if c['cell_type']=='code')
        table = symtable.symtable(combined, 'notebook', 'exec')
        bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
        bound.update(vars(builtins))
        bound.update(('__name__', '__builtins__'))
        missing = set()

        def inspect(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in bound:
                    missing.add(symbol.get_name())
            for child in scope.get_children():
                inspect(child)

        inspect(table)
        self.assertEqual(missing, set())

    def test_training_and_evaluation_contract_preserved(self):
        runner = self.payloads['run_csa_full.py']
        data = self.payloads['full_data.py']
        self.assertIn('TRAIN_IMAGES=113287;TRAIN_EVENTS=566747;VALID_IMAGES=5000;VALID_EVENTS=25010', data)
        self.assertIn('STEPS=(TRAIN_EVENTS+31)//32;EPOCHS=10', data)
        self.assertIn('8e-6+(1e-4-8e-6)', data)
        self.assertTrue("Accelerator(mixed_precision='no'" in runner)
        self.assertNotIn("pilot_precision=", self.header)
        self.assertIn('torch.optim.Adam(model.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)', runner)
        self.assertIn("point['CIDEr']>state['best_CIDEr']", runner)
        self.assertIn('model.load_state_dict(ck[\'model\'],strict=True)', runner)
        self.assertIn("if ck['identity']!=identity", runner)
        self.assertIn('checkpoint();raise TimeoutError', runner)
        self.assertNotIn('test_references.json', runner)
        self.assertIn('METEOR', self.payloads['score_full.py'])

    def test_same_decoder_and_bank_independent_constructor(self):
        source = self.payloads['bank_layers.py']
        tree = ast.parse(source)
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Config')
        defaults = {n.target.id: ast.literal_eval(n.value) for n in config.body if isinstance(n, ast.AnnAssign) and n.value is not None}
        self.assertEqual({k: defaults[k] for k in ('layers', 'dim', 'heads', 'hidden', 'dropout')},
                         dict(layers=3, dim=512, heads=8, hidden=2048, dropout=.2))
        captioner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='CSACaptioner')
        init = next(n for n in captioner.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        # Banks are validated/copied as buffers; none of the parameter constructors use them.
        constructors = ('Linear', 'Embedding', 'ModuleList', 'SemanticAggregation', 'Fusion', 'DecoderLayer')
        for call in calls:
            name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ''
            if name in constructors:
                used = {n.id for arg in call.args for n in ast.walk(arg) if isinstance(n, ast.Name)}
                self.assertTrue(used.isdisjoint({'nouns', 'verbs', 'bank'}))
        model = ast.parse(self.payloads['visual_only_model.py'])
        cls = next(n for n in model.body if isinstance(n, ast.ClassDef) and n.name=='VisualOnlyCaptioner')
        self.assertEqual({n.name for n in cls.body if isinstance(n, ast.FunctionDef)}, {'__init__', 'streams'})
        # encode/decode/forward are inherited, including the exact original decoder.
        original = self.payloads['csa_patch_model.py']
        self.assertIn('return self.core.decode(prefix, vo, vr, valid)', original)

    def test_mock_pruning_and_prompt_independent_routing(self):
        class FakeTensor:
            ndim = 3
            shape = (2, 196, 768)

        class AllFinite:
            def all(self):
                return True

        visual = object()

        class Base:
            def __init__(self, vocab, nouns, verbs):
                self.core = types.SimpleNamespace(project=lambda x: visual, decoder=object(), embedding=object(), head=object())
                for name in ('noun_agg', 'verb_agg', 'object1', 'object2', 'relation', 'nouns', 'verbs'):
                    setattr(self.core, name, object())
                self.object_prompt, self.relation_prompt = object(), object()

        torch_mock = types.ModuleType('torch')
        torch_mock.zeros = lambda *shape: object()
        torch_mock.isfinite = lambda x: AllFinite()
        upstream = types.ModuleType('csa_patch_model')
        upstream.CSAPatchCaptioner = Base
        spec = importlib.util.spec_from_file_location('mock_visual_only', HERE/'visual_only_model.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch_mock, 'csa_patch_model': upstream}):
            spec.loader.exec_module(module)
            model = module.VisualOnlyCaptioner(10000)
            self.assertEqual(set(vars(model.core)), {'project', 'decoder', 'embedding', 'head'})
            self.assertFalse(hasattr(model, 'object_prompt'))
            self.assertFalse(hasattr(model, 'relation_prompt'))
            a, b = model.streams(FakeTensor(), object())
            self.assertIs(a, visual)
            self.assertIs(a, b)
            with self.assertRaisesRegex(ValueError, '196'):
                model.streams(types.SimpleNamespace(ndim=3, shape=(2, 197, 768)), {})
            self.assertEqual(module.VisualOnlyContext('no-read').events(None, None, None), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
