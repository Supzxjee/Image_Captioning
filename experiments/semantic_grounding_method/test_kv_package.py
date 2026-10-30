"""Static delivery checks; does not execute notebook, HDF5, or model code."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class PackageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = json.loads((HERE/'KV_CPU_NOTEBOOK_PACKAGE.json').read_bytes())
        cls.raw = (HERE/cls.package['notebook']).read_bytes()
        cls.notebook = json.loads(cls.raw)

    def test_clean_cpu_notebook_compiles_below_kaggle_limit(self):
        self.assertLess(len(self.raw), 1_000_000)
        self.assertFalse(self.notebook['metadata']['kaggle']['isGpuEnabled'])
        self.assertTrue(self.notebook['metadata']['kaggle']['isInternetEnabled'])
        for i, cell in enumerate(self.notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile(cell['source'], f'cell-{i}', 'exec')

    def test_all_assets_and_notebook_match_pinned_git_bytes(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.package['notebook_sha256'])
        for path, expected in self.package['assets'].items():
            raw = subprocess.check_output(['git','show',self.package['core_commit']+':'+path],cwd=ROOT)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, path)

    def test_preparation_loads_only_legacy_tensors_not_checkpoint(self):
        source = subprocess.check_output(['git','show', self.package['core_commit']+
                 ':experiments/semantic_grounding_method/prepare_kv_cpu.py'],cwd=ROOT)
        tree = ast.parse(source)
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
                 n.func.value.id == 'torch' and n.func.attr == 'load']
        self.assertEqual(len(loads), 1)
        self.assertEqual(ast.unparse(loads[0].args[0]), "cfg['legacy_prompt']")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('from_pretrained', 'backward', 'cuda'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
