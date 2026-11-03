"""Metadata/NumPy-free text fixtures. No NLP/scorer/model/API execution."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import contextlib
import io
import sys
import tempfile
import types
import zipfile
from types import SimpleNamespace
from saved_quality_data import check_rows,common_refs,chair_counts,align_cider_scores

HERE=Path(__file__).parent

class Checks(unittest.TestCase):
    def test_same_cohort_and_first_five_reference_policy(self):
        refs=[dict(coco_id=i,references=[str(i)+' word']*5) for i in range(4000)]
        baseline=dict(cohort_ids=list(range(4000)),reference_sha256=hashlib.sha256(json.dumps(refs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
        expanded=copy.deepcopy(refs)
        for r in expanded[:8]:r['references'].append('extra new reference')
        self.assertEqual(common_refs(expanded,baseline),refs)
        baseline['legacy_order']=list(reversed(range(4000)))
        self.assertEqual(common_refs(expanded,baseline),list(reversed(refs)))
        bad=copy.deepcopy(expanded);bad[0]['references'][0]='changed'
        with self.assertRaises(ValueError):common_refs(bad,baseline)
        with self.assertRaises(ValueError):common_refs(expanded[:-1],baseline)

    def test_duplicate_or_missing_caption_rejected(self):
        rows=[dict(coco_id=1,caption='a dog'),dict(coco_id=2,caption='')]
        check_rows(rows,{1,2})
        for bad in (rows[:1],rows+[rows[0]],[rows[0],rows[0]]):
            with self.assertRaises(ValueError):check_rows(bad,{1,2})

    def test_chair_counts_not_confused_with_denominator(self):
        rows=[dict(coco_id=1,caption='a dog'),dict(coco_id=2,caption='a cat and a bus')]
        sentences=[dict(image_id=1,caption='a dog',mscoco_hallucinated_words=[],mscoco_generated_words=['dog'],mscoco_gt_words=['dog']),
            dict(image_id=2,caption='a cat and a bus',mscoco_hallucinated_words=[['cat','cat']],mscoco_generated_words=['cat','bus'],mscoco_gt_words=['bus'])]
        chair=dict(sentences=sentences,overall_metrics=dict(CHAIRs=.5,CHAIRi=1/3,Recall=1.))
        metric,counts=chair_counts(chair,rows)
        self.assertEqual(counts['hallucinated_captions'],1);self.assertEqual(counts['hallucinated_mentions'],1);self.assertEqual(counts['object_mentions'],3)
        bad=copy.deepcopy(chair);bad['overall_metrics']['CHAIRi']=.2
        with self.assertRaises(ValueError):chair_counts(bad,rows)
        bad=copy.deepcopy(chair);bad['sentences'][0]['caption']='not original'
        with self.assertRaises(ValueError):chair_counts(bad,rows)

    def test_baseline_bundle_identity_and_coverage(self):
        manifest=json.loads((HERE/'COMPARISON_INPUTS.json').read_text(encoding='utf-8'));raw=(HERE/'comparison_baseline4000.json.gz').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest['payload_sha256'])
        bundle=json.loads(gzip.decompress(raw));self.assertEqual(set(bundle['models']),{'A','C0','C'})
        for rows in bundle['models'].values():check_rows(rows,set(bundle['cohort_ids']))
        self.assertEqual(manifest['shared_word_types'],6916)
        self.assertEqual(set(bundle['legacy_order']),set(bundle['cohort_ids']))
        self.assertEqual([bundle['legacy_image_ids'][str(i)] for i in bundle['legacy_order']],sorted(bundle['legacy_image_ids'].values()))

    def test_cpu_source_no_model_execution(self):
        for name in ('saved_quality_data.py','score_saved_quality.py','make_saved_quality_notebook.py'):
            text=(HERE/name).read_text(encoding='utf-8');ast.parse(text);self.assertNotIn('import torch',text)
        text=(HERE/'score_saved_quality.py').read_text(encoding='utf-8')
        self.assertIn('replay_mismatches.json',text);self.assertIn('METRIC_REPLAY_FAILED',text)
        self.assertIn('if mismatches:raise ValueError',text);self.assertIn('if cider_valid else []',text)
        self.assertIn('old_comparable_val_nll=None',text)

    def test_per_image_scores_follow_reference_ids_not_hypothesis_order(self):
        refs=[dict(coco_id=2),dict(coco_id=1)];rows=[dict(coco_id=1),dict(coco_id=2)]
        self.assertEqual(align_cider_scores(refs,[.2,.9],rows),[.9,.2])
        with self.assertRaises(ValueError):align_cider_scores(refs,[.2],rows)
        with self.assertRaises(ValueError):align_cider_scores(refs,[.2,.9],rows[:1])

    def test_replay_failure_keeps_all_chair_outputs_without_false_completion(self):
        import score_saved_quality as scorer
        class Tokenizer:
            def tokenize(self,rows):return {k:[x['caption'] for x in v] for k,v in rows.items()}
        class Cider:
            def compute_score(self,refs,hyp):return .5,[.5]*len(refs)
        class Bleu:
            def __init__(self,*args):pass
            def compute_score(self,refs,hyp):return [0.,0.,0.,0.],None
        class Rouge:
            def compute_score(self,refs,hyp):return 0.,None
        modules={}
        for name,attribute,value in [('tokenizer.ptbtokenizer','PTBTokenizer',Tokenizer),('cider.cider','Cider',Cider),
                                     ('cider.cider_scorer','CiderScorer',Cider),('bleu.bleu','Bleu',Bleu),('rouge.rouge','Rouge',Rouge)]:
            parts=('pycocoevalcap.'+name).split('.')
            for i in range(1,len(parts)+1):
                key='.'.join(parts[:i])
                if key not in modules:modules[key]=types.ModuleType(key);modules[key].__path__=[]
            setattr(modules['.'.join(parts)],attribute,value)
        with tempfile.TemporaryDirectory(dir=HERE,prefix='quality-mock-') as tmp:
            root=Path(tmp);self.assertTrue(root.resolve().is_relative_to(HERE.resolve()))
            refs=[dict(coco_id=i,references=['reference']*5) for i in range(5000)]
            rows=[dict(coco_id=i,caption='a dog') for i in range(5000)]
            bundle=dict(cohort_ids=list(range(4000)),legacy_order=list(range(4000)),legacy_image_ids={str(i):i+10000 for i in range(4000)},
                reference_sha256=hashlib.sha256(json.dumps(refs[:4000],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                models={name:rows[:4000] for name in ('A','C0','C')},expected_metrics={name:dict(CIDEr=.7,CHAIRs=0.,CHAIRi=0.,Recall=0.,Bleu_4=0.,ROUGE_L=0.,mean_words=2.,
                    counts=dict(hallucinated_captions=0,hallucinated_mentions=0,object_mentions=0,recalled_unique=0,reference_unique=0)) for name in ('A','C0','C')})
            review=root/'review.zip'
            with zipfile.ZipFile(review,'w') as z:
                z.writestr('result.json',json.dumps(dict(status='FULL_XE_10_COMPLETE',best_epoch=9,history=[dict(cider=.7)]*10)))
                z.writestr('validation_references.json',json.dumps(refs));z.writestr('validation_ptb.json',json.dumps({str(x['coco_id']):x['references'] for x in refs}))
                for epoch in (9,10):
                    for rank in (0,1):z.writestr(f'val_epoch{epoch:02d}_rank{rank}.json',json.dumps(rows[rank::2]))
            baseline=root/'baseline.gz';baseline.write_bytes(gzip.compress(json.dumps(bundle).encode()))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(new_review_zip_sha256=scorer.sha(review),payload_sha256=scorer.sha(baseline),shared_word_types=6916)))
            chair=root/'chair';chair.mkdir();(chair/'chair.pkl').write_bytes(b'not-a-real-cache');calls=[]
            def fake_chair(command,**kwargs):
                calls.append(command);data=json.loads(Path(command[command.index('--cap_file')+1]).read_text())
                result=dict(sentences=[dict(image_id=x['image_id'],caption=x['caption'],mscoco_hallucinated_words=[],mscoco_generated_words=[],mscoco_gt_words=[]) for x in data],
                            overall_metrics=dict(CHAIRs=0.,CHAIRi=0.,Recall=0.))
                Path(command[command.index('--save_path')+1]).write_text(json.dumps(result))
            out=root/'output';args=SimpleNamespace(review=review,baseline=baseline,manifest=manifest,chair=chair,output=out)
            with patch.dict(sys.modules,modules),patch.object(scorer,'CHAIR_SHA',scorer.sha(chair/'chair.pkl')), \
                 patch('score_saved_quality.subprocess.run',fake_chair),patch('score_saved_quality.importlib.metadata.version',lambda k:'mock'), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'Metric replay mismatch'):scorer.run(args)
            result=json.loads((out/'comparison.json').read_text())
            self.assertEqual(len(calls),7);self.assertEqual(result['status'],'METRIC_REPLAY_FAILED')
            self.assertFalse(result['cider_replay_valid']);self.assertTrue(result['chair_replay_valid']);self.assertEqual(result['paired'],{})
            self.assertEqual(set(result['metrics']['common4000']),{'A','C0','C','CSA9','CSA10'})
            self.assertTrue((out/'full5000_CSA10_chair_output.json').is_file())
            first=result['replay_mismatches'][0];self.assertEqual((first['expected'],first['actual']),(.7,.5))

    def auth_fixture(self,version='139',secret='FAKE_TOKEN_FOR_TEST'):
        nb=json.loads((HERE/'CSA_SAVED_QUALITY_V139_CPU.ipynb').read_text(encoding='utf-8'))
        source=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
        calls=[];output=io.StringIO()
        class FakeProcess:
            def __init__(self,args,**kwargs):
                calls.append((args,kwargs));self.stdout=iter([secret+' safe log\n'])
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self):return 0
        env=dict(KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/'+version,SECRET_NAME='codex')
        with patch.dict(sys.modules,{'kaggle_secrets':SimpleNamespace(UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:secret))}), \
             patch('subprocess.Popen',FakeProcess),contextlib.redirect_stdout(output):
            exec(compile(source,'auth-cell','exec'),env)
        return calls,output.getvalue()

    def test_auth_cell_executes_without_dataset_upload_variables(self):
        calls,output=self.auth_fixture()
        self.assertEqual(len(calls),2)
        self.assertNotIn('KAGGLE_API_TOKEN',calls[0][1]['env'])
        self.assertEqual(calls[1][1]['env']['KAGGLE_API_TOKEN'],'FAKE_TOKEN_FOR_TEST')
        self.assertNotIn('FAKE_TOKEN_FOR_TEST',output);self.assertIn('[REDACTED]',output)
        self.assertNotIn('KAGGLE_KEY',calls[1][1]['env']);self.assertNotIn('KAGGLE_USERNAME',calls[1][1]['env'])

    def test_auth_keeps_exact_version_and_empty_secret_guards(self):
        with self.assertRaises(ValueError):self.auth_fixture(version='138')
        with self.assertRaises(RuntimeError):self.auth_fixture(secret='')

if __name__=='__main__':unittest.main(verbosity=2)
