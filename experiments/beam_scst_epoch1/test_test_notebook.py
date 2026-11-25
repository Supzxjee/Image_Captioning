"""Static/metadata/mock checks; never call API, Torch, captioner or scorers."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from make_test_notebook import HERE, NAME, SHA, XE_SHA, VERSION, build
from make_notebook import literals
import test_notebook as previous
from test_seed_test_notebook_routing import FakeApi


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((HERE/NAME).read_text(encoding='utf-8'))
        cls.values=literals(''.join(cls.nb['cells'][1]['source']))
        cls.payloads=cls.values['SOURCE_PAYLOADS']
        cls.main=''.join(cls.nb['cells'][3]['source'])
        cls.setup=''.join(cls.nb['cells'][1]['source'])

    def test_cells_hashes_globals_and_dependency_closure(self):
        previous.Checks.test_cells_payload_hashes_and_global_dependencies(self)
        previous.Checks.test_dependency_closure_and_no_local_torch(self)
        self.assertEqual(self.values['SOURCE_VERSION'],VERSION)
        self.assertEqual(self.values['ENDPOINT_SHA'],SHA)
        self.assertEqual(self.values['WALL_HOURS'],3)
        self.assertEqual(self.values['CHECKPOINT_ROOT'],'')
        self.assertNotIn('CACHE_ROOT',self.values)

    def test_model_beam_and_selector_unchanged(self):
        v2=json.loads((HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_text(encoding='utf-8'))
        old=literals(''.join(v2['cells'][1]['source']))['SOURCE_PAYLOADS']
        for name in ('csa_patch_model.py','memory_fusion.py','bank_layers.py','semantic_patch_model.py','semantic_data.py','patch_model.py','patch_beam.py'):
            self.assertEqual(self.payloads[name],old[name])
        self.assertEqual(json.loads(self.payloads['RERANK_TEST_INPUTS.json'])['frozen_selector'],self.values['FROZEN_SELECTOR'])
        self.assertEqual(self.values['FROZEN_SELECTOR']['decoder_weight'],.3)
        self.assertEqual(self.values['FROZEN_SELECTOR']['clip_weight'],.4)
        self.assertEqual(self.values['FROZEN_SELECTOR']['object_weight'],.3)
        runner=self.payloads['run_csa_per_caption_test.py']
        self.assertIn('model.load_state_dict(ck[\'model\'],strict=True)',runner)
        self.assertIn('model.eval()',runner);self.assertIn('torch.inference_mode()',runner)
        self.assertIn('beams_sha256=sha',runner)
        self.assertNotIn('torch.optim',runner)
        self.assertEqual(self.main.count("SOURCE/'run_csa_per_caption_test.py','worker'"),1)
        self.assertNotIn("'train',CONFIG",self.main)
        self.assertNotIn('evaluate_pilot',self.main)

    def metadata(self):
        source=self.payloads['csa_per_caption_test_data.py']
        nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.Assign,ast.FunctionDef))]
        original=dict(expected_test_ids=[999999],test_id_sha256='test-only',beam_width=5,max_new_tokens=20,
            length_normalization=False,reranking=False)
        env=dict(Path=Path,__file__='/tmp/csa_per_caption_test_data.py',shared=SimpleNamespace(lock=lambda:copy.deepcopy(original)))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<endpoint contract>','exec'),env)
        return env

    def test_endpoint_binary_shape_and_TEST_contract_reject_foreign_source(self):
        env=self.metadata();ck=dict(updates=3541,source_sha256=XE_SHA,model={'core.nouns':[],'core.verbs':[]})
        result=env['validate_checkpoint_state'](ck)
        self.assertEqual(result['checkpoint_sha256'],SHA)
        self.assertEqual(result['source_version'],VERSION)
        self.assertEqual(result['expected_test_ids'],[999999])
        self.assertEqual(result['updates'],3541)
        self.assertEqual(result['epoch'],1)
        for change in ({'updates':1500},{'source_sha256':'other'},{'model':{}}):
            with self.assertRaises(ValueError):env['validate_checkpoint_state'](dict(ck,**change))
        wrong=dict(result,expected_test_ids=[42])
        with self.assertRaises(ValueError):env['validate_lock'](wrong)
        self.assertNotIn('configure_valid',self.payloads['csa_per_caption_test_data.py'])

    def test_effective_API_routing_and_acquisition_before_heavy_phases(self):
        route=literals(self.main)['ROUTING_PREFLIGHT']
        names=('kaggle','kaggle.api','kaggle.api.kaggle_api_extended')
        modules={name:ModuleType(name) for name in names};modules[names[-1]].KaggleApi=FakeApi
        with patch.dict(sys.modules,modules),contextlib.redirect_stdout(io.StringIO()):exec(route,dict(SOURCE_VERSION=VERSION))
        start=self.main.index('    sys.path.insert(0,str(SOURCE))')
        end=self.main.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'",start)
        block=textwrap.dedent(self.main[start:end]);calls=[]
        secret=ModuleType('kaggle_secrets');secret.UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:'fixture-token')
        resolver=ModuleType('checkpoint_input')
        def check(path,expected,report):
            self.assertEqual(expected,SHA)
            result=Path(path)/'pilot_checkpoint.bin';self.assertTrue(result.is_file());return result
        resolver.resolve_mounted=check
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            def run(args,timeout,env=None):
                args=list(map(str,args));calls.append((args,env))
                if 'kernels' in args:(Path(args[args.index('-p')+1])/'pilot_checkpoint.bin').write_bytes(b'API fixture')
            scope=dict(sys=sys,os=__import__('os'),Path=Path,json=json,SOURCE=root,OUTPUT=out,CLI=root/'cli',
                ENDPOINT_SHA=SHA,SOURCE_VERSION=VERSION,SEED=43,SECRET_NAMES=('codex','KAGGLE_API_TOKEN'),TOKENS=[],
                CHECKPOINT_ROOT='',cfg={},write_config=lambda:None,run=run,ROUTING_PREFLIGHT=route)
            with patch.dict(sys.modules,{'kaggle_secrets':secret,'checkpoint_input':resolver}),patch.object(sys,'path',list(sys.path)):
                exec(block,scope)
            receipt=json.loads((out/'checkpoint_source.json').read_text())
            self.assertEqual(receipt['version'],VERSION)
            self.assertEqual(receipt['sha256'],SHA)
        args,env=next(c for c in calls if 'kernels' in c[0])
        self.assertEqual(args[args.index('output')+1],VERSION)
        pattern=args[args.index('--file-pattern')+1]
        self.assertEqual(pattern,'(^|/)beam_scst_epoch1_seed43_v179_v2_run1/pilot_checkpoint[.]bin$')
        self.assertNotIn('fixture-token',' '.join(args))
        self.assertNotIn('KAGGLE_KEY',env);self.assertNotIn('KAGGLE_USERNAME',env)
        self.assertLess(start,self.main.index("'pycocoevalcap==1.2'"))
        self.assertLess(start,self.main.index("'prepare','--config'"))
        route_command=next(args[2] for args,_ in calls if args[1:2]==['-c'])
        self.assertEqual(ast.literal_eval(ast.parse(route_command).body[0].value),VERSION)

    def test_all_metrics_saved_candidates_recovery_and_same_seed_reference(self):
        scorer=self.payloads['run_csa_per_caption_rerank.py']
        for key in ('CIDEr','BLEU_1','BLEU_4','METEOR','SPICE','ROUGE_L','CHAIRs','CHAIRi'):
            self.assertIn(key,scorer+self.payloads['saved_quality_data.py'])
        for token in ("'_captions.json'","'_cider_per_image.json'","'_basic_chair.json'"):
            self.assertLess(scorer.index(token),scorer.index('meteor = Meteor()'))
        self.assertLess(scorer.index("'_meteor.json'"),scorer.index('spice_score, spice_per'))
        self.assertIn("'selected_ranks.json'",scorer)
        self.assertIn('selection replay differs',scorer)
        self.assertIn('TEST references differ from same-seed CE',self.payloads['csa_per_caption_rerank_data.py'])
        ref=self.values['CE_REFERENCE']
        self.assertAlmostEqual(ref['decoder_rank0']['CIDEr'],1.2198431675475085)
        self.assertAlmostEqual(ref['clip_object_fixed']['CIDEr'],1.2436387660847645)
        self.assertIn('delta_vs_same_seed_CE',self.main)
        self.assertIn('test_tuning=False',self.main)
        self.assertNotIn('expected_top1',self.main) # New checkpoint, no CE caption replay requirement.
        manifest=json.loads((HERE/'TEST_SOURCE_MANIFEST.json').read_text())
        self.assertEqual(manifest['notebook_sha256'],hashlib.sha256((HERE/NAME).read_bytes()).hexdigest())


if __name__=='__main__':unittest.main(verbosity=2)
