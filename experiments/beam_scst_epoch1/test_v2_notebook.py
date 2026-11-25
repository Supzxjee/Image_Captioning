"""Offline regression for source179 acquisition; no API/Torch/model calls."""
import contextlib
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import ModuleType
import unittest
from unittest.mock import patch

import test_notebook as prior
from make_notebook import literals

HERE = Path(__file__).resolve().parent
SHA = 'a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'
VERSION = 'ducanh3/notebook7bc83c4c76/179'


class Checks(prior.Checks):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.original_values = cls.values
        cls.nb = json.loads((HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup = ''.join(cls.nb['cells'][1]['source'])
        cls.values = literals(cls.setup)
        cls.payloads = cls.values['SOURCE_PAYLOADS']

    def test_real_guards_training_recipe_and_validation_scope(self):
        self.assertEqual((self.values['SEED'],self.values['SOURCE_VERSION'],self.values['CHECKPOINT_SHA']), (43,VERSION,SHA))
        self.assertEqual(self.values['CHECKPOINT_ROOT'],'')
        self.assertEqual(self.values['RESUME_ROOT'],'')
        self.assertEqual(self.values['WALL_HOURS'],11.5)
        source=self.payloads['run_beam_epoch.py']
        restored=source.replace(SHA,'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef')
        restored=restored.replace('4342+rank','4242+rank').replace('seed=4342, XE_seed=43, order=','seed=4242, order=')
        restored=restored.replace('Need seed43/v179 CE best8, not an SCST checkpoint','Need original CE best9, not an SCST checkpoint')
        self.assertEqual(restored,self.original_values['SOURCE_PAYLOADS']['run_beam_epoch.py'])
        for name in ('beam_policy.py','full_epoch_data.py','captioning/scst.py'):
            self.assertEqual(self.payloads[name],self.original_values['SOURCE_PAYLOADS'][name])
        runner=''.join(self.nb['cells'][3]['source'])
        source_check=runner.index('weights=checkpoint(')
        self.assertLess(source_check,runner.index("SOURCE/'cuda_runtime_guard.py'"))
        self.assertLess(source_check,runner.index("'pycocoevalcap==1.2'"))
        self.assertLess(source_check,runner.index('prepare(cfg)'))
        self.assertIn('bind_verified_cache(cfg,\'b0c5bde',runner)
        self.assertIn('eval_update=3541',runner)
        self.assertNotIn("'test'",runner)
        self.assertIn('redact(traceback.format_exc())',runner)
        self.assertAlmostEqual(self.values['REFERENCE_VALID']['CIDEr'],1.2177780747765262)
        self.assertAlmostEqual(self.values['REFERENCE_VALID']['CHAIRs'],.0418)
        self.assertAlmostEqual(self.values['REFERENCE_VALID']['NLL'],2.1552617615644585)
        manifest=json.loads((HERE/'V2_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['notebook_sha256'],hashlib.sha256((HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_bytes()).hexdigest())

    def test_mounted_renamed_file_hash_and_rejection_without_pickle_load(self):
        module=prior.load_module('checkpoint_input_host',HERE/'checkpoint_input.py')
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);sub=root/'nested';sub.mkdir()
            renamed=sub/'renamed weights.pth';renamed.write_bytes(b'raw bytes fixture')
            digest=hashlib.sha256(renamed.read_bytes()).hexdigest()
            report=root/'report.json'
            self.assertEqual(module.resolve_mounted(root,digest,report),renamed.resolve())
            self.assertEqual(json.loads(report.read_text())['matched_file'],str(renamed.resolve()))
            with self.assertRaises(FileNotFoundError):module.resolve_mounted(root,'0'*64,report)
            unpacked=root/'archive';unpacked.mkdir();(unpacked/'data.pkl').write_bytes(b'never unpickle this')
            with self.assertRaisesRegex(FileNotFoundError,'appears extracted'):
                module.resolve_mounted(root,'0'*64,report)
        self.assertNotIn('torch',sys.modules)

    def acquisition(self, temporary, mounted=False, secret=True):
        root=Path(temporary);out=root/'output';out.mkdir();source=root/'source';source.mkdir()
        calls=[]
        runner=''.join(self.nb['cells'][3]['source'])
        start=runner.index('    sys.path.insert(0,str(SOURCE))')
        end=runner.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'",start)
        acquisition=textwrap.dedent(runner[start:end])
        secrets=ModuleType('kaggle_secrets')
        class Client:
            def get_secret(self,name):
                if not secret:raise KeyError(name)
                return 'test_only_token'
        secrets.UserSecretsClient=Client
        checkpoint=ModuleType('checkpoint_input')
        def resolve(path,expected,report):
            self.assertEqual(expected,SHA)
            result=Path(path) if Path(path).is_file() else Path(path)/'best.bin'
            self.assertTrue(result.is_file())
            return result
        checkpoint.resolve_mounted=resolve
        def run(args,timeout,environment=None):
            args=list(map(str,args));calls.append((args,timeout,environment))
            if 'kernels' in args:
                target=Path(args[args.index('-p')+1]);(target/'best.bin').write_bytes(b'mocked API transport')
        cfg={}
        scope=dict(sys=sys,os=os,Path=Path,json=json,SOURCE=source,OUTPUT=out,CLI=source/'cli',
            CHECKPOINT_ROOT='',CHECKPOINT_SHA=SHA,SOURCE_VERSION=VERSION,SEED=43,
            SECRET_NAMES=('codex','KAGGLE_API_TOKEN'),TOKENS=[],cfg=cfg,write_config=lambda:None,
            ROUTING_PREFLIGHT=self.values['ROUTING_PREFLIGHT'],run=run)
        if mounted:
            path=root/'mounted renamed.pth';path.write_bytes(b'mock transport')
            scope['CHECKPOINT_ROOT']=str(path)
        with patch.dict(sys.modules,{'kaggle_secrets':secrets,'checkpoint_input':checkpoint}),patch.object(sys,'path',list(sys.path)):
            exec(compile(acquisition,'<API acquisition metadata>','exec'),scope)
        return calls,cfg,json.loads((out/'checkpoint_source.json').read_text())

    def test_API_route_pins_179_hash_and_isolated_credentials_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls,cfg,receipt=self.acquisition(temporary)
        command=next(c for c in calls if 'kernels' in c[0])
        args,timeout,env=command
        self.assertEqual(args[args.index('output')+1],VERSION)
        self.assertEqual(args[args.index('--file-pattern')+1],'(^|/)best[.]bin$')
        self.assertEqual(env['KAGGLE_API_TOKEN'],'test_only_token')
        self.assertNotIn('KAGGLE_KEY',env);self.assertNotIn('KAGGLE_USERNAME',env)
        self.assertNotIn('test_only_token',' '.join(args))
        route=next(c for c in calls if c[0][1:2]==['-c'])[0][2]
        parsed=ast.parse(route)
        self.assertEqual(ast.literal_eval(parsed.body[0].value),VERSION)
        self.assertIn("request.version_label",route)
        self.assertIn("'v'+version",route)
        self.assertEqual((receipt['seed'],receipt['epoch'],receipt['version'],receipt['sha256']),(43,8,VERSION,SHA))
        self.assertTrue(cfg['checkpoint_root'].endswith('input_source/best.bin') or cfg['checkpoint_root'].endswith('input_source\\best.bin'))

    def test_mounted_source_skips_API_and_missing_secret_stops_before_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls,_,_=self.acquisition(temporary,mounted=True)
        self.assertEqual(calls,[])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError,'No cache/model run yet'):
                self.acquisition(temporary,secret=False)


if __name__=='__main__':
    unittest.main(verbosity=2)
