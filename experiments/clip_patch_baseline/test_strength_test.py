"""Pure metadata/mock contracts for fixed epoch8 strength TEST."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import strength_test_data as d
import run_test199 as runner
from real_bank_data import save,read,sha


class Checks(unittest.TestCase):
    def test_fixed_epoch8_rule_and_same_TEST_ids(self):
        value=d.lock();self.assertEqual(value['expected_test_ids'],d.shared.lock()['expected_test_ids']);self.assertEqual(value['updates'],141688)
        for key,v in (('epoch',9),('updates',177110),('source_version','ducanh2403/kltnnnnnff/9'),('checkpoint_sha256','0'*64),('beam_width',4),('reranking',True),('method','bank_prompt_patch_v1')):
            bad=copy.deepcopy(value);bad[key]=v
            with self.assertRaises(ValueError):d.validate_lock(bad)

    def test_snapshot_requires_learned_gate_and_correct_cursor(self):
        ck=dict(state=dict(epoch=9,batch_cursor=0,phase='train',updates=141688,caption_events=4533976,best_epoch=8,history=[{}]*8),
            identity=dict(source=dict(commit=d.lock()['training_commit']),method='bank_prompt_strength_v1',epochs=10,global_batch=32),
            model={k:None for k in ('noun_bank','verb_bank','strength_gate.weight','strength_gate.bias')})
        d.validate_checkpoint_state(ck)
        bad=copy.deepcopy(ck);bad['model'].pop('strength_gate.weight')
        with self.assertRaises(ValueError):d.validate_checkpoint_state(bad)
        bad=copy.deepcopy(ck);bad['state']['updates']=177110
        with self.assertRaises(ValueError):d.validate_checkpoint_state(bad)

    def test_exact_best_file_and_no_final_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'best.bin').write_bytes(b'best8');(root/'checkpoint.bin').write_bytes(b'final10')
            with patch.object(d,'CHECKPOINT_SHA',sha(root/'best.bin')):
                self.assertEqual(d.checkpoint(root),root/'best.bin')
                with self.assertRaises(ValueError):d.checkpoint(root/'checkpoint.bin')

    def test_prepare_reuses_unchanged_prompt_and_patch_paths(self):
        self.assertIs(d.prompt_inputs,d.semantic.prompt_inputs)
        with patch.object(d,'prompt_inputs',return_value=([],[],[],{})),patch.object(d,'checkpoint',return_value=Path('/fixed/best.bin')),patch.object(d.shared,'prepare') as stage,patch.object(d.semantic,'stage_prompts') as prompts:
            d.prepare(dict(checkpoint_root='/fixed'))
            self.assertEqual(stage.call_args.kwargs['locked']['epoch'],8);self.assertEqual(stage.call_args.kwargs['weight_path'],Path('/fixed/best.bin'))
            prompts.assert_called_once()

    def test_all8_metrics_and_epoch8_output(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);save(out/'scoring_lock.json',dict(files_sha256={}));save(out/'chair_counts.json',{});save(out/'environment.json',{})
            parts={'basic':dict(BLEU_1=.7,BLEU_4=.3,CIDEr=1.2,ROUGE_L=.5),'meteor':dict(METEOR=.2),'spice':dict(SPICE=.2),'chair':dict(CHAIRs=.04,CHAIRi=.03,Recall=.4)}
            for mode,metrics in parts.items():save(out/(mode+'.json'),dict(images=5000,metrics=metrics,input_sha256={}))
            with patch.object(runner,'inputs',return_value=(out,[],[],[dict(NLL_sum=6,NLL_tokens=3,runtime_policy={})])),patch('builtins.print'):
                cfg=dict(output=str(out),semantic=True,semantic_strength_gate=True);runner.combine(cfg);result=read(out/'result.json')
                self.assertEqual(result['status'],'CLIP_PATCH_STRENGTH_TEST5000_COMPLETE');self.assertEqual(result['epoch'],8);self.assertEqual(result['checkpoint_sha256'],d.CHECKPOINT_SHA)
                save(out/'spice.json',dict(images=5000,metrics={},input_sha256={}))
                with self.assertRaises(ValueError):runner.combine(cfg)

    def test_model_flag_and_no_training_or_local_neural_import(self):
        here=Path(__file__).parent
        for n in ('strength_test_data.py','run_test199.py','make_test199_notebook.py'):ast.parse((here/n).read_text(encoding='utf-8'))
        source=(here/'run_test199.py').read_text(encoding='utf-8');self.assertIn("strength_gate=cfg.get('semantic_strength_gate',False)",source)
        self.assertIn('model.load_state_dict(ck[\'model\'],strict=True)',source)
        for bad in ('backward(', 'torch.optim', 'optimizer.step', 'model.train('):self.assertNotIn(bad,source)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
