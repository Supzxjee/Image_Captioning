"""Host syntax/metadata/mock checks only; never run Torch, models or scorers."""
import ast
import builtins
import contextlib
import copy
import hashlib
import json
import io
import math
from pathlib import Path
import symtable
import sys
import tempfile
import textwrap
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from make_notebook import HERE, OLD, NAME, literals
import epoch2_contract as contract


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((HERE/NAME).read_text(encoding='utf-8'))
        cls.setup=''.join(cls.nb['cells'][1]['source']);cls.values=literals(cls.setup)
        cls.train=cls.values['TRAIN_PAYLOADS'];cls.test=cls.values['TEST_PAYLOADS']
        cls.main=''.join(cls.nb['cells'][3]['source'])
        cls.worker=cls.train['run_epoch2.py']
        cls.expected=json.loads(cls.train['PARENT_CHECKPOINT_RECEIPT.json'])

    def test_syntax_hashes_unexecuted_and_global_dependencies(self):
        for i,c in enumerate(self.nb['cells']):
            if c['cell_type']=='code':
                compile(''.join(c['source']),f'cell{i}','exec')
                self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
        for label,payload in (('TRAIN',self.train),('TEST',self.test)):
            for n,s in payload.items():
                if n.endswith('.py'):compile(s,n,'exec')
                self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),self.values[label+'_SHA256'][n])
        combined='\n'.join(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code')
        table=symtable.symtable(combined,'notebook','exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}|set(vars(builtins))|{'__name__'}
        def check(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced():self.assertIn(symbol.get_name(),bound)
            for child in scope.get_children():check(child)
        check(table)
        self.assertEqual(self.values['WALL_HOURS'],5.75)
        self.assertEqual(self.values['EVALUATION_RESERVE_SECONDS'],6000)
        self.assertEqual(self.values['CHECKPOINT_SHA'],contract.PARENT_SHA)
        self.assertEqual(self.values['SOURCE_VERSION'],contract.PARENT_VERSION)

    def parent_state(self):
        return dict(identity=self.expected['identity'],updates=3541,cursor_images=113287,
                    source_sha256=contract.XE_SHA,model={},optimizer={},
                    rng=[dict(cpu='fixture',cuda='fixture',numpy='fixture') for _ in (0,1)])

    def identity(self):
        old=self.expected['identity']
        return contract.make_identity(dict(source_hashes=self.values['TRAIN_SHA256'],
            cache_manifest_sha256=old['cache_manifest_sha256']),self.expected,old['torch'],old['numpy'],old['vocabulary_sha256'])

    def receipt(self,cursor):
        return dict(identity=self.identity(),epoch=2,parent_sha256=contract.PARENT_SHA,source_sha256=contract.XE_SHA,
            cursor_images=cursor,updates=3541+math.ceil(cursor/32),complete_epoch=cursor==113287)

    def test_parent_full_state_and_software_lineage_reject_weights_only(self):
        contract.validate_parent(self.parent_state(),self.expected)
        for change in ({'updates':3540},{'cursor_images':113280},{'source_sha256':'wrong'},{'rng':[]}):
            with self.assertRaises(ValueError):contract.validate_parent(dict(self.parent_state(),**change),self.expected)
        with self.assertRaises((KeyError,ValueError)):contract.validate_parent(dict(model={},updates=3541),self.expected)
        old=self.expected['identity'];cfg=dict(source_hashes={},cache_manifest_sha256=old['cache_manifest_sha256'])
        with self.assertRaises(ValueError):contract.make_identity(cfg,self.expected,'2.11.0+cpu',old['numpy'],old['vocabulary_sha256'])
        with self.assertRaises(ValueError):contract.make_identity(cfg,self.expected,old['torch'],old['numpy'],'changed')
        self.assertEqual(self.identity()['endpoint_updates'],7082)

    def test_epoch_cursor_transition_resume_boundaries_and_final_tail(self):
        for cursor in (0,32,3200,113280,113287):contract.validate_resume(self.receipt(cursor),self.identity())
        self.assertEqual(self.receipt(0)['updates'],3541)
        self.assertEqual(self.receipt(113287)['updates'],7082)
        for change in ({'epoch':1},{'updates':0},{'cursor_images':31},{'source_sha256':'wrong'},
                       {'parent_sha256':'wrong'},{'complete_epoch':True},{'cursor_images':False}):
            with self.assertRaises(ValueError):contract.validate_resume(dict(self.receipt(0),**change),self.identity())
        with self.assertRaises(ValueError):contract.validate_resume(dict(self.receipt(113280),updates=7082),self.identity())
        self.assertIn('cursor, updates = 0, PARENT_STEPS',self.worker)
        self.assertIn('if updates != END_STEPS:',self.worker)
        self.assertIn('optimizer.load_state_dict(prior[\'optimizer\'])',self.worker)
        restore=self.worker.index("torch.set_rng_state(state['cpu'])")
        self.assertLess(self.worker.index('optimizer.load_state_dict'),restore)
        self.assertNotIn('manual_seed',self.worker[restore:])
        self.assertIn("if mode == 'train':checkpoint_now()",self.worker)
        self.assertIn('os.replace(temp, target)',self.worker)
        self.assertIn('epoch=2, parent_sha256=PARENT_SHA, rng=states',self.worker)

    def test_original_model_reward_update_math_and_audited_fast_functions_preserved(self):
        prior=literals(''.join(json.loads((OLD/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_text(encoding='utf-8'))['cells'][1]['source']))['SOURCE_PAYLOADS']
        for name in ('csa_patch_model.py','bank_layers.py','memory_fusion.py','semantic_patch_model.py',
                     'semantic_data.py','patch_model.py','beam_policy.py','full_epoch_data.py','captioning/scst.py','scst_cache.py'):
            self.assertEqual(self.train[name],prior[name])
        self.assertEqual(self.train['speed_candidate.py'],(OLD/'speed_candidate.py').read_text(encoding='utf-8'))
        find=lambda s:next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='update')
        original,new=find(prior['run_beam_epoch.py']),find(self.worker)
        class NormalizeBatch(ast.NodeTransformer):
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='batch_rows':
                    return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='data',ctx=ast.Load()),attr='batch',ctx=ast.Load()),args=node.args[1:],keywords=[]),node)
                return self.generic_visit(node)
        self.assertEqual(ast.dump(original),ast.dump(NormalizeBatch().visit(new)))
        for text in ('lr=4e-6, betas=(.9, .99), eps=1e-8, weight_decay=0','find_unused_parameters=True',
                     'broadcast_buffers=False','torch.backends.cuda.matmul.allow_tf32 = False'):
            self.assertIn(text,self.worker)
        for forbidden in ('autocast','GradScaler','torch.compile','cross_entropy'):
            self.assertNotIn(forbidden,self.worker)
        self.assertIn('return train_beams_fast',self.worker)

    def test_effective_API_acquisition_exact_full_checkpoint_before_heavy_setup(self):
        start=self.main.index('    sys.path.insert(0,str(SOURCE))')
        end=self.main.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'",start)
        block=textwrap.dedent(self.main[start:end]);calls=[]
        secret=ModuleType('kaggle_secrets');secret.UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:'fixture-token')
        resolver=ModuleType('checkpoint_input')
        def resolve(path,digest,report):
            self.assertEqual(digest,contract.PARENT_SHA)
            target=Path(path)/'checkpoint.bin';self.assertTrue(target.is_file());return target
        resolver.resolve_mounted=resolve
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            def run(args,timeout,env=None):
                args=list(map(str,args));calls.append((args,env))
                if 'kernels' in args:(Path(args[args.index('-p')+1])/'checkpoint.bin').write_bytes(b'fixture full state')
            scope=dict(sys=sys,os=__import__('os'),Path=Path,json=json,SOURCE=root,OUTPUT=out,CLI=root/'cli',
                CHECKPOINT_SHA=contract.PARENT_SHA,SOURCE_VERSION=contract.PARENT_VERSION,
                SECRET_NAMES=('codex','KAGGLE_API_TOKEN'),TOKENS=[],CHECKPOINT_ROOT='',cfg={},write_config=lambda:None,
                run=run,ROUTING_PREFLIGHT=self.values['ROUTING_PREFLIGHT'])
            with patch.dict(sys.modules,{'kaggle_secrets':secret,'checkpoint_input':resolver}),patch.object(sys,'path',list(sys.path)):
                exec(block,scope)
        args,env=next(c for c in calls if 'kernels' in c[0])
        self.assertEqual(args[args.index('output')+1],contract.PARENT_VERSION)
        self.assertEqual(args[args.index('--file-pattern')+1],'(^|/)beam_scst_epoch1_seed43_v179_v2_run1/checkpoint[.]bin$')
        self.assertNotIn('fixture-token',' '.join(args));self.assertNotIn('KAGGLE_KEY',env)
        route=ast.parse(self.values['ROUTING_PREFLIGHT'])
        self.assertTrue(any(isinstance(n,ast.Attribute) and n.attr=='version_label' for n in ast.walk(route)))
        self.assertLess(end,self.main.index("'pycocoevalcap==1.2'"))

    def binding(self):
        return dict(sha256='c'*64,epoch=2,updates=7082,seed=43,cursor_images=113287,
                    parent_sha256=contract.PARENT_SHA,source_sha256=contract.XE_SHA,complete_epoch=True,version='local_epoch2_endpoint')

    def test_dynamic_TEST_binding_rejects_partial_or_epoch1_no_stale_SHA(self):
        b=self.binding();contract.validate_eval_binding(b)
        for change in ({'epoch':1},{'updates':3541},{'complete_epoch':False},{'sha256':contract.PARENT_SHA},{'source_sha256':'wrong'}):
            with self.assertRaises(ValueError):contract.validate_eval_binding(dict(b,**change))
        nodes=[n for n in ast.parse(self.test['csa_per_caption_test_data.py']).body if isinstance(n,(ast.Assign,ast.FunctionDef))]
        base=dict(expected_test_ids=[42],test_id_sha256='fixture',beam_width=5,max_new_tokens=20,length_normalization=False,reranking=False)
        env=dict(Path=Path,__file__='/tmp/test_source/csa_per_caption_test_data.py',read=lambda _:b,
                 shared=SimpleNamespace(lock=lambda:copy.deepcopy(base)),validate_eval_binding=contract.validate_eval_binding,
                 END_STEPS=7082,PARENT_SHA=contract.PARENT_SHA)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<TEST dynamic binding>','exec'),env)
        ck=dict(updates=7082,source_sha256=contract.XE_SHA,epoch=2,parent_sha256=contract.PARENT_SHA,model={'core.nouns':[],'core.verbs':[]})
        locked=env['validate_checkpoint_state'](ck)
        self.assertEqual(locked['checkpoint_sha256'],'c'*64);self.assertEqual(locked['epoch'],2)
        self.assertEqual(locked['source_version'],'local_epoch2_endpoint')
        with self.assertRaises(ValueError):env['validate_checkpoint_state'](dict(ck,updates=3541))
        self.assertNotIn('REPLAY_RECEIPTS.json',self.test)
        self.assertLess(self.main.index("training_complete=True"),self.main.index("phase='TEST'"))
        self.assertIn("test_env()",self.main)
        self.assertIn("str(DEPS)+os.pathsep+str(TEST_SOURCE)",self.main)

    def test_fixed_policies_metrics_recovery_and_no_TEST_selection(self):
        ref=self.values['REFERENCES'];self.assertEqual(ref['selector']['decoder_weight'],.3)
        self.assertEqual(ref['selector']['clip_weight'],.4);self.assertEqual(ref['selector']['object_weight'],.3)
        self.assertAlmostEqual(ref['epoch1_TEST']['clip_object_fixed']['CIDEr'],1.301785,places=5)
        original=literals(''.join(json.loads((OLD/'BEAM_SCST_V199_TEST5000_RERANK_API_2GPU.ipynb').read_text(encoding='utf-8'))['cells'][1]['source']))['SOURCE_PAYLOADS']
        for n in ('run_csa_per_caption_test.py','run_csa_per_caption_rerank.py','csa_per_caption_rerank_data.py',
                  'captioning/object_consistency.py','captioning/multiscore_reranking.py','RERANK_TEST_INPUTS.json'):
            self.assertEqual(self.test[n],original[n])
        scorer=self.test['run_csa_per_caption_rerank.py']
        for n in ("'_captions.json'","'_basic_chair.json'","'_cider_per_image.json'"):
            self.assertLess(scorer.index(n),scorer.index('meteor = Meteor()'))
        for n in ('BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall','NLL'):
            self.assertIn(n,self.main)
        self.assertIn('no TEST selection',self.main);self.assertIn('test_tuning=False',self.main)
        self.assertEqual(self.main.count("evaluator,'worker','--config'"),2) # VALID once + TEST once.
        self.assertIn("phase='CPU_metrics'",self.main)
        self.assertIn("except BaseException:",self.main);self.assertIn("finally:",self.main)
        self.assertIn("EVALUATION_RESERVE_SECONDS",self.main)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)

    def test_runtime_import_closure_for_both_isolated_sources(self):
        external={'numpy','torch','h5py','PIL','nltk','pycocoevalcap','transformers','huggingface_hub'}
        for payload,entries in ((self.train,['run_epoch2','run_csa_per_caption_test','full_epoch_data','checkpoint_input']),
                                (self.test,['run_csa_per_caption_test','run_csa_per_caption_rerank','csa_per_caption_rerank_data'])):
            pending=list(entries);seen=set()
            while pending:
                name=pending.pop()
                if name in seen:continue
                seen.add(name);filename=name.replace('.','/')+'.py'
                if filename not in payload:filename=name.replace('.','/')+'/__init__.py'
                self.assertIn(filename,payload,name)
                for node in ast.walk(ast.parse(payload[filename])):
                    imports=[]
                    if isinstance(node,ast.Import):imports=[n.name for n in node.names]
                    elif isinstance(node,ast.ImportFrom) and node.module:
                        imports=[('.'.join(name.split('.')[:-node.level])+'.' if node.level else '')+node.module]
                    for module in imports:
                        root=module.split('.')[0]
                        if root in sys.stdlib_module_names or root in external:continue
                        pending.append(module)
            self.assertNotIn('make_test199_notebook',seen) # Packaging-only inherited file is not a runtime dependency.

    def simulated_pipeline(self,root,failure=None):
        """Fake child outputs exercise orchestration only; no model/metric call."""
        out=root/'out';out.mkdir();source=root/'train';source.mkdir();test_source=root/'test_source';test_source.mkdir()
        (out/'source'/'test').mkdir(parents=True);(out/'test').mkdir()
        config=root/'config.json';test_config=root/'test_config.json';test_config.write_text('{}')
        cfg=dict(output=str(out),source_hashes=self.values['TRAIN_SHA256'])
        calls=[];references=self.values['REFERENCES']
        save=lambda p,v:Path(p).write_text(json.dumps(v),encoding='utf-8')
        def run(args,timeout,env=None):
            args=list(map(str,args));calls.append((args,env));self.assertGreater(timeout,0)
            script=next((Path(a).name for a in args if a.endswith('.py')),'pip')
            if script=='run_epoch2.py':
                if 'preflight' in args:
                    for r in (0,1):save(out/f'beam_preflight_rank{r}.json',dict(passed=True,projected_training_seconds=10800))
                else:
                    (out/'checkpoint.bin').write_bytes(b'fixture full checkpoint, not model weights')
                    if failure=='train_timeout':raise TimeoutError('fixture training timeout')
                    endpoint=b'fixture endpoint, not model weights';(out/'pilot_checkpoint.bin').write_bytes(endpoint)
                    receipt=dict(self.receipt(113287),sha256=hashlib.sha256(endpoint).hexdigest())
                    save(out/'checkpoint_receipt.json',receipt);save(out/'pilot_checkpoint_receipt.json',receipt)
                    save(out/'training_complete.json',dict(complete=True,epochs=2,train_images=113287,
                        updates=7081 if failure=='partial' else 7082,cumulative_image_events=226574))
            elif script=='run_csa_per_caption_test.py':
                is_test=args[args.index('--config')+1]==str(test_config)
                dest=out/'test' if is_test else out
                if 'worker' in args:
                    for r in (0,1):save(dest/f'worker{r}.json',dict(NLL_sum=10,NLL_tokens=4))
                    if is_test:save(dest/'beams_rank0.json',[dict(fixture=True)])
                if 'basic' in args:save(dest/'basic.json',dict(metrics={k:references['epoch1_VALID'][k] for k in ('BLEU_1','BLEU_4','CIDEr','ROUGE_L')}))
                if 'chair' in args:save(dest/'chair.json',dict(metrics={k:references['epoch1_VALID'][k] for k in ('CHAIRs','CHAIRi','Recall')}))
            elif script=='run_csa_per_caption_rerank.py' and 'score' in args:
                if failure=='score':raise TimeoutError('fixture scorer timeout')
                binding=json.loads((out/'evaluation_binding.json').read_text())
                save(out/'test'/'rerank'/'result.json',dict(status='CSA_PER_CAPTION_FIXED_RERANK_TEST5000_COMPLETE',
                    images=5000,checkpoint_sha256=binding['sha256'],selector=references['selector'],metrics=references['epoch1_TEST']))
        scope=dict(OUTPUT=out,SOURCE=source,TEST_SOURCE=test_source,WORK=root,CONFIG=config,CHAIR=root/'chair',
            COCO_JSON=root/'coco.json',DETECTIONS=root/'detections.json',DEPS=root/'deps',test_out=out/'test',test_cfg={},
            test_config=test_config,evaluator=test_source/'run_csa_per_caption_test.py',test_env=lambda:dict(PYTHONPATH='test-only'),cfg=cfg,run=run,
            write_config=lambda:save(config,cfg),sys=sys,os=__import__('os'),Path=Path,json=json,hashlib=hashlib,
            REFERENCES=references,launch=['fixture_distributed_launch'],PARENT_DEADLINE=__import__('time').monotonic()+20700,
            EVALUATION_RESERVE_SECONDS=6000,time=__import__('time'),training_complete=False,complete=False)
        start=self.main.index("    phase='preflight'");end=self.main.index('\nexcept BaseException:')
        try:
            with contextlib.redirect_stdout(io.StringIO()):exec(textwrap.dedent(self.main[start:end]),scope)
        except (ValueError,TimeoutError):
            if failure is None:raise
            return scope,calls
        if failure:raise AssertionError('Expected pipeline failure was accepted')
        return scope,calls

    def test_mock_full_pipeline_and_failures_never_select_partial_or_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for label in ('success','partial','train_timeout','score'):
                dest=root/label;dest.mkdir()
                scope,calls=self.simulated_pipeline(dest,None if label=='success' else label)
                self.assertEqual(sum('train' in a and any(Path(x).name=='run_epoch2.py' for x in a) for a,_ in calls),1)
                out=dest/'out'
                if label=='success':
                    self.assertTrue(scope['complete'])
                    result=json.loads((out/'comparison.json').read_text())
                    self.assertEqual(result['updates'],7082);self.assertEqual(set(result['epoch2_metrics']),{'decoder_rank0','clip_object_fixed'})
                elif label in ('partial','train_timeout'):
                    self.assertFalse(scope['complete']);self.assertFalse((out/'evaluation_binding.json').exists())
                    self.assertFalse(any('checkpoint_preflight' in a for a,_ in calls))
                else:
                    self.assertTrue(scope['training_complete']);self.assertFalse(scope['complete'])
                    self.assertTrue((out/'checkpoint.bin').is_file());self.assertTrue((out/'test'/'beams_rank0.json').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
