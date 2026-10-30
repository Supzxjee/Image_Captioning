"""Metadata/statistics/transport tests; no model, HDF5, API or GPU execution."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from expanded_data import *
from expanded_metrics import quantities,paired_contrasts,exact_mcnemar,gates
from package_eval_checkpoints import package

class Checks(unittest.TestCase):
 def test_exact_current_dev_complement_and_manifest_hash(self):
  self.assertEqual(sha(HERE/'EXPANDED_EVAL_MANIFEST.json'),MANIFEST_SHA)
  old=pinned_metadata(HERE.parents[1]/'experiments/semantic_grounding_method/data/gain_metadata.json.gz')
  manifest=json.loads((HERE/'EXPANDED_EVAL_MANIFEST.json').read_bytes());rows,sentinels=rows_and_sentinels(manifest,old)
  self.assertEqual(len(rows),4000);self.assertEqual(len(sentinels),16)
  bad=dict(manifest,rows=rows[:-1]+[old['groups']['val'][0]])
  with self.assertRaises(ValueError):rows_and_sentinels(bad,old)
 def test_prediction_identity_rejects_missing_duplicate_and_wrong_mapping(self):
  rows=[dict(image_id=5,coco_id=901),dict(image_id=6,coco_id=902)]
  good=[dict(r,caption='a cat') for r in rows];self.assertEqual(validate_predictions(good,rows),good)
  for bad in [good[:1],[good[0],good[0]],[dict(good[0],coco_id=7),good[1]]]:
   with self.assertRaises(ValueError):validate_predictions(bad,rows)
 def test_aggregate_ratios_not_mean_image_ratios(self):
  v=np.array([[1,1,1,1,1,2,8],[2,0,0,9,8,10,10]],dtype=float);q=quantities(v)
  self.assertAlmostEqual(q['CHAIRi'],.1);self.assertAlmostEqual(q['Recall'],9/12)
  self.assertNotEqual(q['CHAIRi'],.5)
 def test_exact_sparse_discordants(self):
  self.assertEqual(exact_mcnemar([0,0],[0,0])['p_two_sided'],1.)
  self.assertEqual(exact_mcnemar([0],[1])['p_two_sided'],1.)
  m=exact_mcnemar(np.zeros(8),np.ones(8));self.assertEqual(m['better'],8);self.assertAlmostEqual(m['p_two_sided'],2/256)
 def test_bootstrap_paired_identical_and_deterministic(self):
  v=np.array([[1,0,0,1,1,2,8],[2,1,1,2,1,3,10]],dtype=float);vectors={n:v.copy() for n in ['A','C0','C']}
  a=paired_contrasts(vectors,draws=240,seed=7);b=paired_contrasts(vectors,draws=240,seed=7);self.assertEqual(a,b)
  for pair in a:
   for k in ['CIDEr','CHAIRs','CHAIRi','Recall','mean_words']:self.assertEqual(a[pair][k]['paired95_ci'],[0.,0.])
  self.assertFalse(gates(a)['joint_pass'])
 def test_joint_gate_does_not_accept_cider_only(self):
  def c():return dict(CIDEr=dict(paired95_ci=[.01,.02]),CHAIRs=dict(paired95_ci=[-.02,-.001]),Recall=dict(paired95_ci=[0.,.01]),McNemar=dict(p_two_sided=.01))
  contrasts={'C-A':c(),'C-C0':c()};self.assertTrue(gates(contrasts)['joint_pass'])
  contrasts['C-C0']['CHAIRs']['paired95_ci']=[-.01,0.];self.assertFalse(gates(contrasts)['joint_pass'])
 def test_transport_preserves_bytes_and_fails_before_bad_copy(self):
  parent=HERE/'review';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as temp:
   root=Path(temp);assert root.resolve().is_relative_to(parent.resolve());paths={};expected={}
   for name,version in [('C',111),('C0',112)]:
    p=root/(name+'.source');p.write_bytes((name*20).encode());paths[name]=p;expected[name]=dict(version=version,sha256=sha(p))
   receipt=package(paths,root/'good',expected)
   self.assertFalse(receipt['model_loaded'])
   for n,p in paths.items():self.assertEqual((root/'good'/(n+'.bin')).read_bytes(),p.read_bytes())
   paths['C'].write_bytes(b'wrong')
   with self.assertRaises(ValueError):package(paths,root/'bad',expected)
   self.assertFalse((root/'bad').exists())
   with self.assertRaises(FileExistsError):package(paths,root/'good',expected)
 def test_eval_worker_has_no_training_or_new_research_probe(self):
  source=(HERE/'run_expanded_eval.py').read_text(encoding='utf-8');tree=ast.parse(source)
  self.assertNotIn('AdamW',source);self.assertNotIn('.backward(',source);self.assertNotIn('class_logmass(',source)
  self.assertNotIn('diagnostic_concept',source);self.assertIn('strict=True',source)
  self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
