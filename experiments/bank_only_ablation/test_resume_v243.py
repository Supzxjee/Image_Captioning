"""Resume source/metadata/API mocks; no Torch, real checkpoint or network."""
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import symtable
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from make_resume_v243 import literals
from resume_v243_data import verify_resume,digest

HERE=Path(__file__).resolve().parent


class FakeApi:
    def build_kaggle_client(self):raise AssertionError('Network must be mocked')
    def kernels_output(self,source,path):
        owner,slug,version=source.split('/')
        with self.build_kaggle_client() as client:
            client.kernels.kernels_api_client.list_kernel_session_output(SimpleNamespace(user_name=owner,kernel_slug=slug,version_label='v'+version))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((HERE/'M2_ONLY_BANK_ALIGNMENT_RESUME_V243_API_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.first=json.loads((HERE/'M2_ONLY_BANK_ALIGNMENT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup=''.join(cls.nb['cells'][1]['source'])
        cls.execution=''.join(cls.nb['cells'][3]['source'])
        cls.values=literals(cls.setup)
        cls.run_values=literals(cls.execution)

    def fixture(self,root):
        folder=root/'m2_only_bank_alignment_seed42_run1';folder.mkdir()
        latest,best=folder/'checkpoint.bin',folder/'best.bin'
        latest.write_bytes(b'fake latest fixture, not Torch');best.write_bytes(b'fake best fixture, not Torch')
        identity=dict(seed=42,image_specific_prompts=False,method='bank_only_parallel3_xe10_v1',source={'sha256':{'bank_only_model.py':'immutable'}})
        history=[]
        for e in range(1,9):
            path=folder/'evaluations'/f'epoch_{e:02d}';path.mkdir(parents=True)
            (path/'result.json').write_text(json.dumps(dict(epoch=e)),encoding='utf-8')
            history.append(dict(epoch=e,files_sha256={'result.json':digest(path/'result.json')}))
        state=dict(epoch=9,batch_cursor=5616,phase='train',updates=147304,caption_events=4713688,best_epoch=8,best_CIDEr=1.1933623617626516,history=history)
        (folder/'training.jsonl').write_text('{"updates":147277}\n',encoding='utf-8')
        receipt=dict(updates=147304,sha256=digest(latest),training_log_sha256=digest(folder/'training.jsonl'))
        bp=dict(epoch=8,sha256=digest(best),CIDEr=state['best_CIDEr'])
        for name,value in [('identity.json',identity),('progress.json',state),('checkpoint_receipt.json',receipt),('best_receipt.json',bp)]:
            (folder/name).write_text(json.dumps(value),encoding='utf-8')
        locked=dict(identity=identity,checkpoint_sha256=digest(latest),best_sha256=digest(best),metadata_sha256={str(p.relative_to(folder)).replace('\\','/'):digest(p) for p in folder.rglob('*') if p.is_file() and p.suffix!='.bin'})
        return folder,locked

    def test_verifier_accepts_and_rejects_changed_weight_metadata_and_cursor(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder,locked=self.fixture(root)
            self.assertEqual(verify_resume(root,locked),folder)
            latest=folder/'checkpoint.bin';raw=latest.read_bytes();latest.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'weights'):verify_resume(root,locked)
            latest.write_bytes(raw)
            metadata=folder/'training.jsonl';data=metadata.read_bytes();metadata.write_bytes(b'wrong log')
            with self.assertRaisesRegex(ValueError,'metadata'):verify_resume(root,locked)
            metadata.write_bytes(data)
            for name,key,value in [('progress.json','batch_cursor',0),('identity.json','image_specific_prompts',True),('identity.json','seed',43)]:
                target=folder/name;original=target.read_bytes();obj=json.loads(original);obj[key]=value
                target.write_text(json.dumps(obj),encoding='utf-8')
                # Reseal that file to test semantic guards beyond the hash guard.
                resealed=copy.deepcopy(locked);resealed['metadata_sha256'][name]=digest(target)
                with self.assertRaises(ValueError):verify_resume(root,resealed)
                target.write_bytes(original)

    def test_all_cells_hashes_unchanged_sources_and_no_missing_names(self):
        for i,c in enumerate(self.nb['cells']):
            if c['cell_type']=='code':
                compile(''.join(c['source']),f'cell{i}','exec');self.assertEqual(c['outputs'],[])
        prior=literals(''.join(self.first['cells'][1]['source']))
        for key in ('SOURCE_PAYLOADS','SOURCE_SHA256','SOURCE_ORIGINS','REVISION'):
            self.assertEqual(self.values[key],prior[key])
        self.assertEqual(self.values['RESUME_LOCK']['identity']['source']['sha256'],self.values['SOURCE_SHA256'])
        self.assertEqual(hashlib.sha256(self.values['RESUME_HELPER'].encode()).hexdigest(),self.values['RESUME_HELPER_SHA'])
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'resume','exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__','__builtins__'}
        def inspect(t):
            for s in t.get_symbols():
                if s.is_global() and s.is_referenced():self.assertIn(s.get_name(),bound)
            for child in t.get_children():inspect(child)
        inspect(table)

    def test_effective_routing_command_compiles_and_targets_v243(self):
        assignments=[n for c in self.nb['cells'] if c['cell_type']=='code' for n in ast.parse(''.join(c['source'])).body
                     if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROUTING_PREFLIGHT' for t in n.targets)]
        self.assertEqual(len(assignments),1)
        route=ast.literal_eval(assignments[0].value)
        self.assertEqual(self.values['SOURCE_VERSION'],'ducanh2403/2-vi-cai-kia-qua-nhieu/243')
        calls=[n for n in ast.walk(ast.parse(self.execution)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run']
        command=None
        for call in calls:
            if not call.args or not isinstance(call.args[0],ast.List):continue
            entries=call.args[0].elts
            if len(entries)>3 and isinstance(entries[2],ast.Constant) and entries[2].value=='-c' and isinstance(entries[3],ast.BinOp):
                command=eval(compile(ast.Expression(entries[3]),'command','eval'),dict(SOURCE_VERSION=self.values['SOURCE_VERSION'],ROUTING_PREFLIGHT=route))
        self.assertIsNotNone(command);compile(command,'api_mock_command','exec')
        packages={n:ModuleType(n) for n in ('kaggle','kaggle.api','kaggle.api.kaggle_api_extended')}
        packages['kaggle.api.kaggle_api_extended'].KaggleApi=FakeApi
        with patch.dict(sys.modules,packages),contextlib.redirect_stdout(io.StringIO()):exec(command,{})

    def test_output_file_filter_and_resume_order(self):
        pattern=self.run_values['RESUME_FILE_PATTERN']
        for suffix in ('checkpoint.bin','best.bin','identity.json','training.jsonl','progress.json','source_receipt.json','evaluations/epoch_07/val_rank1.json'):
            self.assertRegex('m2_only_bank_alignment_seed42_run1/'+suffix,pattern)
        for name in ('other_run/checkpoint.bin','m2_only_bank_alignment_seed42_run1/source/arbitrary.py','m2_only_bank_alignment_seed43_run1/best.bin'):
            self.assertIsNone(re.search(pattern,name))
        self.assertLess(self.execution.index("SOURCE/'cuda_runtime_guard.py'"),self.execution.index("'kernels','output'"))
        self.assertLess(self.execution.index("cfg['resume_root']=str(resume)"),self.execution.index("SOURCE/'run_csa_full.py','worker'"))
        self.assertEqual(self.values['WALL_HOURS'],6)
        self.assertEqual(self.values['RESUME_ROOT'],'')
        self.assertIn('optimizer.load_state_dict',self.values['SOURCE_PAYLOADS']['run_csa_full.py'])
        self.assertIn("rng=ck['rng_by_rank'][rank]",self.values['SOURCE_PAYLOADS']['run_csa_full.py'])
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
