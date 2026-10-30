"""Static notebooks/Git transport and scalar metric checks. No neural execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from kv_pilot_evaluate import paired_false

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = json.loads((HERE/'KV_DE_NOTEBOOK_PACKAGE.json').read_bytes())

    def test_clean_small_notebooks_compile_and_only_arm_config_differs(self):
        sources = {}
        for arm,description in self.package['notebooks'].items():
            raw = (HERE/description['file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),description['sha256'])
            self.assertLess(len(raw),1_000_000)
            notebook = json.loads(raw)
            self.assertTrue(notebook['metadata']['kaggle']['isGpuEnabled'])
            code = []
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    compile(cell['source'],description['file'],'exec')
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])
                    code.append(cell['source'])
            sources[arm] = '\n'.join(code)
            self.assertIn('/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115',sources[arm])
            self.assertIn("'--nproc_per_node=2'",sources[arm])
        self.assertEqual(sources['D'].replace("ARM='D'","ARM='E'"),sources['E'])

    def test_raw_git_assets_hashes_and_source_compilation(self):
        for path,expected in self.package['assets'].items():
            raw = subprocess.check_output(['git','show',self.package['core_commit']+':'+path],cwd=ROOT)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected,path)
            if path.endswith('.py'):compile(raw,path,'exec')

    def test_model_does_not_override_module_device_methods_or_use_gt_in_caption_reader(self):
        source = (HERE/'kv_pilot_model.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
                        self.assertNotIn(target.attr,{'to','float','half','train','eval','parameters','buffers','state_dict','forward'})
            if isinstance(node,ast.FunctionDef) and node.name in ('hidden','bank','context','logits','generate'):
                identifiers = {n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
                self.assertFalse(identifiers & {'facts','labels','ground_truth','references'})

    def test_exact_paired_false_counts_include_zero_discordance(self):
        self.assertEqual(paired_false([1,0],[1,0]),dict(repaired=0,introduced=0,exact_mcnemar_p=1.))
        result = paired_false([1,1,1,1,0],[0,0,0,0,1])
        self.assertEqual((result['repaired'],result['introduced']),(4,1))
        self.assertAlmostEqual(result['exact_mcnemar_p'],.375)


if __name__ == '__main__':unittest.main(verbosity=2)
