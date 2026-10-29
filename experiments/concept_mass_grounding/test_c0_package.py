"""C0 package/ablation checks only; no neural imports or execution."""
import ast,hashlib,json,pathlib,subprocess,sys,unittest
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class Checks(unittest.TestCase):
 def test_pinned_assets_and_unexecuted_notebook(self):
  p=json.loads((HERE/'NOTEBOOK_PACKAGE_C0.json').read_text())
  for name,h in p['source_assets'].items():self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',p['source_commit']+':'+name],cwd=ROOT)).hexdigest(),h)
  for entry in p['notebooks']:
   raw=(HERE/entry['file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['sha256']);self.assertLess(len(raw),1000000)
   nb=json.loads(raw)
   for c in nb['cells']:
    if c['cell_type']=='code':
     compile(c['source'],'C0 cell','exec');self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
   code='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
   self.assertIn("ARM='C0'",code);self.assertIn('objective_weights=policy_module.objective_weights(ARM)',code)
   self.assertIn('post_pilot_ablation=True',code);self.assertIn('PREVIOUS_PREPARED=None',code)
   self.assertIn('upt_hidden_caption_A/checkpoint.bin',code);self.assertIn('timeout=3720',code)
 def test_cg_released_notebooks_unchanged_and_native_model_identical(self):
  for name in ['CONCEPT_MASS_C_2GPU.ipynb','CONCEPT_MASS_G_2GPU.ipynb','concept_model.py','diagnostic_concept.py']:
   expected=subprocess.check_output(['git','show','a33a537:experiments/concept_mass_grounding/'+name],cwd=ROOT)
   self.assertEqual((HERE/name).read_text(encoding='utf-8'),expected.decode())
 def test_weighted_loss_uses_single_effective_configuration(self):
  source=(HERE/'run_concept.py').read_text();tree=ast.parse(source)
  loss=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)]
  self.assertEqual(len(loss),1)
  self.assertEqual(ast.unparse(loss[0]),"rl + weights['kl'] * kl + weights['keep'] * keep + weights['negative'] * neg")
  self.assertIn("objective_weights=weights",source)
 def test_no_neural_import(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
