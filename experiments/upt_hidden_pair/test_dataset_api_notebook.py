import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from make_dataset_api_notebook import build


class Checks(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(build().read_text(encoding='utf-8'))

    def test_notebook_unexecuted_and_all_code_compiles(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                ast.parse(''.join(cell['source']))

    def test_cli_token_only_in_child_environment_and_output_redacted(self):
        code = ''.join(self.notebook['cells'][2]['source'])
        function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
        namespace = dict(os=os, subprocess=subprocess, token='synthetic-secret')
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<test>', 'exec'), namespace)
        with patch.object(subprocess, 'Popen') as popen, redirect_stdout(io.StringIO()) as log:
            popen.return_value.stdout = ['error synthetic-secret\n']
            popen.return_value.wait.return_value = 1
            with self.assertRaises(RuntimeError):
                namespace['run_cli'](['python', '-m', 'kaggle'])
            args, kwargs = popen.call_args
            self.assertNotIn('synthetic-secret', str(args))
            self.assertEqual(kwargs['env']['KAGGLE_API_TOKEN'], 'synthetic-secret')
            self.assertNotIn('synthetic-secret', log.getvalue())
            self.assertIn('[REDACTED]', log.getvalue())
            popen.return_value.wait.return_value = 0
            namespace['run_cli'](['python', '-m', 'pip'], authenticated=False)
            self.assertNotIn('KAGGLE_API_TOKEN', popen.call_args.kwargs['env'])

    def test_create_is_private_flat_and_failure_removes_only_metadata(self):
        code = next(''.join(c['source']) for c in self.notebook['cells']
                    if ''.join(c['source']).startswith('metadata_path ='))
        scratch = Path(__file__).parent / 'review'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            dest = Path(directory)
            (dest / 'features_rank0.h5').write_bytes(b'synthetic')
            def fake_cli(args):
                self.assertNotIn('--public', args)
                self.assertIn('create', args)
                self.assertEqual(args[-2:], ['--dir-mode', 'skip'])
                data = json.loads((dest / 'dataset-metadata.json').read_text())
                self.assertEqual(data['id'], 'tester/test-cache')
                raise RuntimeError('mock upload failure')
            namespace = dict(DESTINATION=dest, OWNER='tester', SLUG='test-cache', TITLE='Test cache',
                             json=json, sys=__import__('sys'), run_cli=fake_cli)
            with self.assertRaisesRegex(RuntimeError, 'mock upload failure'):
                exec(compile(code, '<test>', 'exec'), namespace)
            self.assertFalse((dest / 'dataset-metadata.json').exists())
            self.assertEqual((dest / 'features_rank0.h5').read_bytes(), b'synthetic')

    def test_download_pins_56_filters_files_and_never_falls_back(self):
        code = ''.join(self.notebook['cells'][3]['source'])
        scratch = Path(__file__).parent / 'review'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            calls = []
            def fake_cli(args):
                calls.append(args)
                self.assertIn('ducanh2403/2-vi-cai-kia-qua-nhieu/56', args)
                self.assertIn('--file-pattern', args)
                pattern = __import__('re').compile(args[-1])
                self.assertTrue(pattern.search('upt_hidden_pairs_11k/features_rank0.h5'))
                self.assertFalse(pattern.search('Image_Captioning/checkpoint.pth'))
            namespace = dict(KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/56',
                             Path=Path, sys=__import__('sys'), run_cli=fake_cli)
            with patch.object(tempfile, 'mkdtemp', return_value=directory), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, '0 cache receipt'):
                    exec(compile(code, '<test>', 'exec'), namespace)
            self.assertEqual(len(calls), 1)
            namespace['KERNEL_VERSION'] = 'ducanh2403/2-vi-cai-kia-qua-nhieu/57'
            with self.assertRaises(ValueError):
                exec(compile(code, '<test>', 'exec'), namespace)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
