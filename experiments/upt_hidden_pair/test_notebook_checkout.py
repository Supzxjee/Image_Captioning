"""Offline Git-only test of the notebook checkout helper; no notebook/model execution."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_first_clone_and_fresh_rerun_preserve_old_overlay(self):
        scratch = HERE / 'review'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            origin = root / 'origin'
            origin.mkdir()
            def git(*args, cwd=origin):
                return subprocess.check_output(['git', *args], cwd=cwd, text=True, encoding='utf-8', stderr=subprocess.STDOUT).strip()
            git('init')
            (origin / 'tracked.txt').write_text('original')
            git('add', 'tracked.txt')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
            revision = git('rev-parse', 'HEAD')
            notebooks = sorted(HERE.glob('UPT_HIDDEN_*_2GPU.ipynb'))
            self.assertEqual(len(notebooks), 3)
            for number, file in enumerate(notebooks):
                cells = json.loads(file.read_text(encoding='utf-8'))['cells']
                tree = next(ast.parse(''.join(c['source'])) for c in cells
                            if c['cell_type'] == 'code' and 'def checkout(' in ''.join(c['source']))
                repo_assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                                       and any(isinstance(t, ast.Name) and t.id == 'REPO' for t in n.targets))
                self.assertIn('mkdtemp', ast.unparse(repo_assignment.value))
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checkout')
                self.assertNotIn('--no-checkout', ast.unparse(function))
                def run(args, cwd=None):
                    result = subprocess.run(list(map(str, args)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode:
                        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
                namespace = dict(subprocess=subprocess, run=run)
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<checkout fixture>', 'exec'), namespace)
                old, fresh = root / f'old{number}', root / f'fresh{number}'
                namespace['checkout'](str(origin), old, revision)
                self.assertEqual((old / 'tracked.txt').read_text(), 'original')
                (old / 'tracked.txt').write_text('intentional embedded overlay')
                with self.assertRaises(AssertionError):
                    namespace['checkout'](str(origin), old, revision)
                namespace['checkout'](str(origin), fresh, revision)
                self.assertEqual((old / 'tracked.txt').read_text(), 'intentional embedded overlay')
                self.assertEqual((fresh / 'tracked.txt').read_text(), 'original')


if __name__ == '__main__':
    unittest.main(verbosity=2)
