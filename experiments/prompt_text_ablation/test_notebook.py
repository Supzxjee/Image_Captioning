"""Offline metadata/mock checks; never import Torch or run a research model."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import symtable
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def literals(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            try:result[node.targets[0].id]=ast.literal_eval(node.value)
            except (TypeError,ValueError):pass
    return result


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup=''.join(cls.nb['cells'][1]['source'])
        cls.header=cls.setup.split('\nSOURCE_PAYLOADS = ',1)[0]
        cls.values=literals(cls.setup)
        cls.payloads=cls.values['SOURCE_PAYLOADS']
        cls.parent=json.loads((ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_H5_SEED43_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.old=literals(''.join(cls.parent['cells'][1]['source']))

    def test_all_cells_and_sources_compile_hashes_and_no_missing_globals(self):
        for i,c in enumerate(self.nb['cells']):
            if c['cell_type']=='code':
                compile(''.join(c['source']),f'cell{i}','exec')
                self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
        for name,code in self.payloads.items():
            self.assertEqual(hashlib.sha256(code.encode('utf-8')).hexdigest(),self.values['SOURCE_SHA256'][name])
            if name.endswith('.py'):ast.parse(code,filename=name)
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'notebook','exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__','__builtins__'}
        def inspect(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced():self.assertIn(symbol.get_name(),bound)
            for child in scope.get_children():inspect(child)
        inspect(table)

    def test_local_dependency_closure(self):
        sources={Path(n).stem:ast.parse(code) for n,code in self.payloads.items() if n.endswith('.py')}
        inventory={p.stem for folder in (ROOT/'experiments/clip_patch_baseline',ROOT/'experiments/csa_paper',HERE) for p in folder.glob('*.py')}
        pending=['run_csa_full','csa_full_runtime','score_csa_full','zero_prompt_model','test_semantic_data','test_csa_full_data']
        seen=set()
        while pending:
            name=pending.pop()
            if name in seen:continue
            self.assertIn(name,sources);seen.add(name)
            for node in ast.walk(sources[name]):
                names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
                for target in names:
                    if target in inventory:self.assertIn(target,sources);pending.append(target)
        self.assertGreater(len(seen),20)

    def test_full_architecture_and_metadata_loader_unchanged(self):
        # Hash equality ties the unmodified sources to the known full training recipe.
        by_name={name:path for path,name in self.old['MAPPING'].items()}
        for name in ('csa_patch_model.py','bank_layers.py','semantic_patch_model.py','semantic_data.py','full_data.py','patch_beam.py','score_full.py'):
            self.assertEqual(hashlib.sha256(self.payloads[name].encode()).hexdigest(),self.old['HASHES'][by_name[name]])
        for name in ('csa_full_data.py',):self.assertEqual(self.payloads[name],self.old['SEED_SOURCES'][name])
        tree=ast.parse(self.payloads['zero_prompt_model.py'])
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ZeroTextCaptioner')
        self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['streams'])
        self.assertEqual(cls.bases[0].id,'CSAPatchCaptioner')
        self.assertFalse(any(isinstance(n,ast.If) for n in ast.walk(cls)))
        self.assertNotIn('self.training',ast.get_source_segment(self.payloads['zero_prompt_model.py'],cls))
        original=self.payloads['csa_patch_model.py']
        self.assertIn('torch.cat(self.streams(patches, p), dim=1)',original)
        self.assertIn('self.decode(self.encode(patches, p), prefix)',original)

    def test_mock_zeroing_preserves_every_nontext_field_in_train_and_eval(self):
        class Tensor:
            def __init__(self,value,shape):self.value,self.shape=value,shape
        torch_mock=ModuleType('torch')
        torch_mock.zeros_like=lambda value:Tensor(0,value.shape)
        upstream=ModuleType('csa_patch_model')
        class Parent:
            def streams(self,patches,prompts):return patches,prompts
        upstream.CSAPatchCaptioner=Parent
        spec=importlib.util.spec_from_file_location('zero_mock',HERE/'zero_prompt_model.py')
        module=importlib.util.module_from_spec(spec)
        prompts={key:Tensor(3,(2,8,512)) for key in ('object_text','relation_text')}
        prompts.update({key:object() for key in ('object_weights','object_box','subject_weights','target_weights','relation_geometry','object_valid','relation_valid')})
        originals=dict(prompts)
        with patch.dict(sys.modules,{'torch':torch_mock,'csa_patch_model':upstream}):
            spec.loader.exec_module(module)
            for train in (True,False):
                model=module.ZeroTextCaptioner();model.training=train
                x=object();actual_x,actual=model.streams(x,prompts)
                self.assertIs(actual_x,x);self.assertIsNot(actual,prompts)
                self.assertEqual(set(actual),set(prompts))
                for key,value in prompts.items():
                    if key in ('object_text','relation_text'):
                        self.assertEqual(actual[key].value,0);self.assertEqual(actual[key].shape,value.shape)
                        self.assertEqual(value.value,3)
                    else:self.assertIs(actual[key],value)
                self.assertEqual(prompts,originals)
        self.assertNotIn('torch',sys.modules)

    def test_train_recipe_selection_and_resume_isolation(self):
        self.assertEqual((self.values['SEED'],self.values['RESUME_ROOT'],self.values['WALL_HOURS']),(42,'',11.5))
        self.assertIn("prompt_text_mode='zero'",self.header)
        self.assertIn('semantic=True',self.header)
        self.assertIn("prior.get('method')!=cfg['method']",self.header)
        self.assertIn("prior.get('prompt_text_mode')!='zero'",self.header)
        runner=self.payloads['run_csa_full.py']
        for expression in ("torch.manual_seed(seed)","global_batch=32,epochs=10", "mixed_precision='no'", "prompt_text_mode='zero'", "point['CIDEr']>state['best_CIDEr']", "if ck['identity']!=identity"):
            self.assertTrue(expression in runner,msg=expression)
        self.assertIn('return ZeroTextCaptioner(vocab,*banks)',self.payloads['csa_full_runtime.py'])
        self.assertIn("('noun','verb')",self.payloads['csa_full_runtime.py'])
        execution=''.join(self.nb['cells'][3]['source'])
        self.assertNotIn("'TEST'",execution)
        self.assertIn("SOURCE/'run_csa_full.py','preflight'",execution)
        self.assertIn("SOURCE/'run_csa_full.py','worker'",execution)

    def test_guard_avoids_vacuous_initial_zero_adapter_and_preserves_original(self):
        source=self.payloads['zero_prompt_model.py']
        tree=ast.parse(source)
        guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='contract_guard')
        text=ast.get_source_segment(source,guard)
        self.assertIn('probe = copy.deepcopy(model)',text)
        self.assertIn('torch.nn.init.eye_(branch.attention.out_proj.weight)',text)
        self.assertIn('reference.load_state_dict(probe.state_dict(), strict=True)',text)
        self.assertIn('reference.encode(x, zeroed)',text)
        self.assertIn('sensitive <= 1e-7',text)
        self.assertIn('tensor_digest(model) != initial',text)
        self.assertIn('count != 53183736',text)
        self.assertIn('contract_guard(model,len(words),cfg,rank,device,out)',self.payloads['csa_full_runtime.py'])
        self.assertNotIn('pycocoevalcap',sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)
