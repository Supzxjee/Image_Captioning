"""Metadata/NumPy-free text fixtures. No NLP/scorer/model/API execution."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import unittest
from saved_quality_data import check_rows,common_refs,chair_counts

HERE=Path(__file__).parent

class Checks(unittest.TestCase):
    def test_same_cohort_and_first_five_reference_policy(self):
        refs=[dict(coco_id=i,references=[str(i)+' word']*5) for i in range(4000)]
        baseline=dict(cohort_ids=list(range(4000)),reference_sha256=hashlib.sha256(json.dumps(refs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
        expanded=copy.deepcopy(refs)
        for r in expanded[:8]:r['references'].append('extra new reference')
        self.assertEqual(common_refs(expanded,baseline),refs)
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

    def test_cpu_source_no_model_execution(self):
        for name in ('saved_quality_data.py','score_saved_quality.py','make_saved_quality_notebook.py'):
            text=(HERE/name).read_text(encoding='utf-8');ast.parse(text);self.assertNotIn('import torch',text)
        text=(HERE/'score_saved_quality.py').read_text(encoding='utf-8')
        self.assertIn('Historical evaluator replay differs:',text);self.assertIn('CSA stored CIDEr replay differs:',text)
        self.assertIn('old_comparable_val_nll=None',text)

if __name__=='__main__':unittest.main(verbosity=2)
