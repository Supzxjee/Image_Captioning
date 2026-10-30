"""Reproduce Linux blob mismatch and mock the exact in-session repair; no model."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REL='experiments/concept_mass_grounding/expanded_data.py'
MANIFEST='experiments/concept_mass_grounding/EXPANDED_EVAL_MANIFEST.json'

class Checks(unittest.TestCase):
 def test_original_failure_is_only_line_endings(self):
  raw=subprocess.check_output(['git','show','13daa4f:'+MANIFEST],cwd=ROOT)
  self.assertEqual(hashlib.sha256(raw).hexdigest(),'8f68ab9c45ab9c2d11618e837b8d945f20c0c3148c3b1705ff352cdf244a6fab')
  old=raw.replace(b'\n',b'\r\n')
  self.assertEqual(hashlib.sha256(old).hexdigest(),'31559d3f68a133fe9d923489875df21b641b84b8238d3a805ec54de9a99af661')
  self.assertEqual(json.loads(old),json.loads(raw))
 def test_in_session_fix_preserves_science_and_original_receipt(self):
  parent=HERE/'review';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as temp:
   repo=Path(temp);assert repo.resolve().is_relative_to(parent.resolve());here=repo/'experiments/concept_mass_grounding';here.mkdir(parents=True)
   assets={}
   for relative in [REL,MANIFEST]:
    raw=subprocess.check_output(['git','show','13daa4f:'+relative],cwd=ROOT);p=repo/relative;p.write_bytes(raw);assets[relative]=hashlib.sha256(raw).hexdigest()
   cfg=dict(source_commit='13daa4f',source_assets=assets,policy=dict(count=4000,training=False),checkpoints=dict(C='fixedC',C0='fixedC0'))
   config=repo/'config.json';config.write_text(json.dumps(cfg),encoding='utf-8');old_config=config.read_bytes();output=repo/'output';review=repo/'failed_review.zip';review.write_bytes(b'preserved failure')
   env=dict(SOURCE_COMMIT='13daa4f',SOURCE_ASSETS=dict(assets),REPO=repo,HERE=here,CONFIG=config,OUTPUT=output,REVIEW=review)
   code=(HERE/'FIX_EXPANDED_MANIFEST_LF_CELL.py').read_text(encoding='utf-8')
   with contextlib.redirect_stdout(io.StringIO()):exec(code,env)
   fixed=json.loads(config.read_text(encoding='utf-8'))
   self.assertEqual(fixed['policy'],cfg['policy']);self.assertEqual(fixed['checkpoints'],cfg['checkpoints'])
   self.assertFalse(fixed['technical_amendment']['cohort_changed'])
   self.assertEqual(review.read_bytes(),b'preserved failure');self.assertEqual(json.loads(config.with_name(config.name+'.before_manifest_lf_fix.json').read_text()),cfg)
   self.assertIn(b"MANIFEST_SHA='8f68",(repo/REL).read_bytes())
   self.assertEqual(env['SOURCE_ASSETS'][REL],hashlib.sha256((repo/REL).read_bytes()).hexdigest())
   before=config.read_bytes()
   with contextlib.redirect_stdout(io.StringIO()):exec(code,env)
   self.assertEqual(config.read_bytes(),before)
   output.mkdir()
   with self.assertRaisesRegex(RuntimeError,'Preparation already'):exec(code,env)

if __name__=='__main__':unittest.main()
