"""Pure AST/argument regression; never invoke pip, venv, spaCy or a model."""
import ast
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest


class Checks(unittest.TestCase):
    def test_pipless_environment_uses_base_pip_for_every_pip_operation(self):
        notebook = json.loads((Path(__file__).parent/'CSA_PAPER_REAL_PREPARATION_CPU.ipynb').read_text(encoding='utf-8'))
        source = notebook['cells'][-1]['source']
        block = next(n for n in ast.parse(source).body if isinstance(n, ast.Try)).body
        bindings = dict(sys=SimpleNamespace(executable='/usr/bin/python3'), SOURCE=PurePosixPath('/tmp/mock'),
                        OUTPUT=PurePosixPath('/tmp/output'), CONFIG=PurePosixPath('/tmp/mock/config.json'),
                        constraint=PurePosixPath('/tmp/mock/constraints.txt'), wheel=PurePosixPath('/tmp/mock/model.whl'))
        commands = []
        for statement in block:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in ('venv', 'PYTHON', 'PIP'):
                    bindings[name] = eval(compile(ast.Expression(statement.value), '<argument>', 'eval'), bindings)
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == 'run':
                commands.append(list(map(str, eval(compile(ast.Expression(statement.value.args[0]), '<argument>', 'eval'), bindings))))
        self.assertEqual(commands[0], ['/usr/bin/python3', '-m', 'venv', '--without-pip', '--system-site-packages', '/tmp/mock/pos-env'])
        prefix = ['/usr/bin/python3', '-m', 'pip', '--python', '/tmp/mock/pos-env/bin/python']
        self.assertEqual(commands[1][:5], prefix)
        self.assertEqual(commands[2][:5], prefix)
        self.assertEqual(commands[1][-1], 'spacy==3.8.11')
        self.assertIn('--force-reinstall', commands[2])
        self.assertIn("subprocess.check_output(PIP + ['freeze', '--all']", source)
        self.assertIn("'--python' not in subprocess.check_output", source)
        self.assertNotIn("[PYTHON, '-m', 'pip'", source)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
