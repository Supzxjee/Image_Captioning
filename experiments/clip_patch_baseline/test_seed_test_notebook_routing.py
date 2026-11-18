"""Offline regression checks for overwritten Kaggle routing in seed TEST notebooks."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def notebook_assignments(nb, name):
    return [node for cell in nb['cells'] if cell['cell_type']=='code'
            for node in ast.parse(''.join(cell['source'])).body
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)]


def routing_source(nb):
    assignments=notebook_assignments(nb,'ROUTING_PREFLIGHT')
    if len(assignments)!=1:
        raise ValueError('Expected one routing definition across all cells')
    return ast.literal_eval(assignments[0].value)


class FakeApi:
    def build_kaggle_client(self):
        raise AssertionError('A network client must be mocked')

    def kernels_output(self, source, path):
        owner,slug,version=source.split('/')
        with self.build_kaggle_client() as client:
            request=SimpleNamespace(user_name=owner,kernel_slug=slug,version_label='v'+version)
            client.kernels.kernels_api_client.list_kernel_session_output(request)


class Checks(TestCase):
    def test_effective_routing_and_duplicate_regression(self):
        package_names=('kaggle','kaggle.api','kaggle.api.kaggle_api_extended')
        modules={name:ModuleType(name) for name in package_names}
        modules[package_names[-1]].KaggleApi=FakeApi
        for seed,expected in ((43,'ducanh3/notebook7bc83c4c76/179'),
                              (44,'ducanh2403/2-vi-cai-kia-qua-nhieu/236')):
            with self.subTest(seed=seed):
                nb=json.loads((HERE/f'CSA_PER_CAPTION_SEED{seed}_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))
                for cell in nb['cells']:
                    if cell['cell_type']=='code':compile(''.join(cell['source']),'cell','exec')
                actual=ast.literal_eval(notebook_assignments(nb,'SOURCE_VERSION')[0].value)
                self.assertEqual(actual,expected)
                routing=routing_source(nb)
                self.assertIn('api.kernels_output(SOURCE_VERSION',routing)
                self.assertNotIn('kltnnnnnff',routing)
                # Execute the effective assignment as Python, then its mocked preflight.
                env={'SOURCE_VERSION':actual}
                nodes=notebook_assignments(nb,'ROUTING_PREFLIGHT')
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'assignments','exec'),env)
                with patch.dict(sys.modules,modules),contextlib.redirect_stdout(io.StringIO()):
                    exec(env['ROUTING_PREFLIGHT'],env)
                nb['cells'].append(dict(cell_type='code',source=["ROUTING_PREFLIGHT='stale'\n"]))
                with self.assertRaises(ValueError):
                    routing_source(nb)


if __name__=='__main__':
    main(verbosity=2)
