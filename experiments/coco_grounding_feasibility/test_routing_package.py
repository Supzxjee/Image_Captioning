"""Local data/packaging checks only. GPU checks remain user-run in the notebooks."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'experiments/upt_alignment_pilot'))
from routing_data import tensors_for


class Checks(unittest.TestCase):
    def test_recovery_reuses_stage_and_never_runs_prepare(self):
        script=(HERE/'resume_preflight.py').read_text(encoding='utf-8');ast.parse(script)
        source=(HERE/'routing_preflight.py').read_text(encoding='utf-8')
        sha=hashlib.sha256(source.encode()).hexdigest()
        (HERE/'review').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'review') as tmp:
            root=Path(tmp);repo=root/'repo';out=root/'output';repo.mkdir();out.mkdir()
            (repo/'routing_preflight.py').write_bytes(source.encode('utf-8'))
            stage=root/'stage.h5';stage.write_bytes(b'preserve existing stage')
            cfg=dict(stage=str(stage),implementation_sha256={'routing_preflight.py':sha},arm='R',aux_weight=.05)
            protocol=out/'protocol.json';protocol.write_text(json.dumps(cfg),encoding='utf-8')
            calls=[]
            ns=dict(REPO=repo,OUTPUT=out,LOG=root/'missing.log',run=lambda args,cwd:calls.append(args))
            exec(compile(script,'recovery','exec'),ns)
            self.assertEqual(json.loads(protocol.read_text()),cfg)
            self.assertEqual(json.loads((out/'protocol_before_preflight_fix.json').read_text()),cfg)
            self.assertEqual(stage.read_bytes(),b'preserve existing stage')
            self.assertEqual([args[args.index('run_routing.py')+1] for args in calls],['preflight','worker','evaluate','archive'])
            (out/'history_rank0.json').write_text('[]')
            with self.assertRaises(AssertionError):exec(compile(script,'recovery','exec'),ns)

    def test_actual_five_reference_expansion_rejects_wrong_order(self):
        tree=ast.parse((HERE/'routing_preflight.py').read_text(encoding='utf-8'))
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_teacher_expansion')
        namespace=dict(torch=SimpleNamespace(equal=np.array_equal))
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'expansion_check','exec'),namespace)
        check=namespace['check_teacher_expansion']
        memory=np.arange(12).reshape(2,3,2);padding=np.array([[False,False,True],[False,True,True]])
        targets=np.arange(40).reshape(10,4);masks=np.zeros((10,4),dtype=bool)
        correct=dict(memory=np.repeat(memory,5,axis=0),padding=np.repeat(padding,5,axis=0),targets=targets,masks=masks)
        check(correct,memory,padding,targets,masks)
        wrong=dict(correct,memory=np.tile(memory,(5,1,1)),padding=np.tile(padding,(5,1)))
        with self.assertRaises(ValueError):check(wrong,memory,padding,targets,masks)
        wrong=dict(correct,targets=targets[::-1])
        with self.assertRaises(ValueError):check(wrong,memory,padding,targets,masks)

    def test_prepare_creates_output_before_writes_and_preserves_existing_stage(self):
        # Exercise the filesystem setup before any ML imports or model work.
        tree=ast.parse((HERE/'routing_stage.py').read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
        prefix=[]
        for node in function.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='begin' for t in node.targets):break
            if not isinstance(node,(ast.Import,ast.ImportFrom)):prefix.append(node)
        function.body=prefix
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        namespace=dict(json=json,Path=Path,verify_sources=lambda cfg:None)
        exec(compile(module,'routing_stage_setup','exec'),namespace)
        (HERE/'review').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'review') as tmp:
            root=Path(tmp);output=root/'nested'/'output';stage=root/'stage.h5';config=root/'config.json'
            config.write_text(json.dumps(dict(output=str(output),stage=str(stage))))
            namespace['prepare'](config)
            self.assertTrue(output.is_dir())
            (output/'protocol.json').write_text('preserve')
            with self.assertRaises(FileExistsError):namespace['prepare'](config)
            self.assertEqual((output/'protocol.json').read_text(),'preserve')
            (output/'protocol.json').unlink();stage.write_bytes(b'preserve stage')
            with self.assertRaises(FileExistsError):namespace['prepare'](config)
            self.assertEqual(stage.read_bytes(),b'preserve stage')

    def test_multitoken_and_empty_image_mapping(self):
        masks={'a':dict(patch_area_fraction=np.ones((14,14)).tolist())}
        rows=[dict(decoder_logit_positions=[3,4],caption_index=4,category='person',mask_key='a')]
        value=tensors_for(rows,masks,3,2)
        self.assertEqual(value['span_caption'].tolist(),[4,0,0])
        self.assertEqual(value['span_token_valid'].tolist(),[[True,True],[False,False],[False,False]])
        np.testing.assert_array_equal(value['span_positions'][0],[3,4])
        self.assertEqual(value['span_masks'].shape,(3,196))
        self.assertFalse(tensors_for([],masks,3,2)['span_valid'].any())

    def test_sealed_dataset(self):
        ready=json.loads((HERE/'FULL_DATA_READY.json').read_text(encoding='utf-8'))
        archive=HERE/'review/coco_noun_box_train10k_v11_dataset.zip'
        if not archive.exists():self.skipTest('Raw local dataset intentionally not in Git.')
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),ready['dataset_archive_sha256'])
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip());self.assertEqual(len(z.namelist()),7)
            self.assertTrue(all(Path(n).name==n for n in z.namelist()))
            receipt=json.loads(z.read('manifest_receipt.json'))
            for name,expected in receipt['payload_sha256'].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),expected)
            self.assertEqual(receipt['retained_spans'],17904)
            self.assertEqual(receipt['person_spans'],4476)

    def test_notebook_sources_and_locked_recipe(self):
        bundles=[]
        for arm in ('B','R'):
            path=HERE/f'COCO_ROUTING_{arm}_2GPU.ipynb'
            self.assertLess(path.stat().st_size,1_000_000)
            notebook=json.loads(path.read_text(encoding='utf-8'))
            texts=[]
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                text=''.join(cell['source']);ast.parse(text);texts.append(text)
            combined='\n'.join(texts)
            packed=re.search(r"b64decode\('([^']+)'\)",combined).group(1)
            bundle=json.loads(zlib.decompress(base64.b64decode(packed)))
            for name,source in bundle.items():
                ast.parse(source,filename=name)
                if (HERE/name).exists():self.assertEqual(source,(HERE/name).read_text(encoding='utf-8'))
            for item in ("warmup_updates=200","base_lr=5e-6","aux_weight=.05","updates=1250",
                         "--nproc_per_node=2","('preflight','worker')","nltk==3.9.2"):
                self.assertIn(item,combined)
            self.assertNotIn('HIDDEN_CACHE=',combined)
            bundles.append(bundle)
        self.assertEqual(*bundles)


if __name__=='__main__':unittest.main(verbosity=2)
