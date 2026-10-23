"""Transport/package and notebook credential checks; no network/model execution."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from package_dataset import package,sha,validate_inventory,locate
from make_dataset_notebook import build

HERE=Path(__file__).resolve().parent


def fixture(root):
    source=root/'source';source.mkdir();inventory=json.loads((HERE/'upload_inventory.json').read_text(encoding='utf-8'))
    for name in inventory['files']:
        p=source/name;p.write_bytes(('synthetic transport payload:'+name).encode())
        inventory['files'][name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return source,inventory


class Checks(unittest.TestCase):
    def test_copy_repeat_and_corruption_preservation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp,redirect_stdout(io.StringIO()):
            source,inventory=fixture(Path(tmp));dest=Path(tmp)/'dest'
            package(source,dest,inventory);package(source,dest,inventory)
            self.assertEqual(len(list(dest.iterdir())),64)
            for n in inventory['files']:self.assertEqual((source/n).read_bytes(),(dest/n).read_bytes())
            partial=dest/'train_001.h5.partial';partial.write_bytes(b'uncommitted')
            with self.assertRaises(ValueError):package(source,dest,inventory)
            self.assertEqual(partial.read_bytes(),b'uncommitted');partial.unlink()
            (dest/'train_000.h5').write_bytes(b'preserve changed destination')
            with self.assertRaises(ValueError):package(source,dest,inventory)
            self.assertEqual((dest/'train_000.h5').read_bytes(),b'preserve changed destination')

    def test_missing_source_stops_before_copy(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp,redirect_stdout(io.StringIO()):
            source,inventory=fixture(Path(tmp));(source/'val_009.h5').unlink();dest=Path(tmp)/'dest'
            with self.assertRaises(ValueError):package(source,dest,inventory)
            self.assertEqual(list(dest.iterdir()),[])

    def test_unsafe_inventory_and_nested_destination_rejected(self):
        inventory=json.loads((HERE/'upload_inventory.json').read_text(encoding='utf-8'))
        bad=copy.deepcopy(inventory);bad['files']['../private']=dict(bytes=1,sha256='x')
        with self.assertRaises(ValueError):validate_inventory(bad)
        with self.assertRaises(ValueError):package(HERE,HERE/'inside',inventory)

    def test_explicit_source_requires_exact_receipt(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            source,inventory=fixture(Path(tmp));self.assertEqual(locate(inventory,source),source.resolve())
            (source/'cache_receipt.json').write_bytes(b'wrong run')
            with self.assertRaises(ValueError):locate(inventory,source)

    def test_notebook_source_inventory_and_private_api_contract(self):
        with redirect_stdout(io.StringIO()):path=build()
        notebook=json.loads(path.read_text(encoding='utf-8'));self.assertLess(path.stat().st_size,1000000)
        self.assertEqual(notebook['metadata']['kaggle']['accelerator'],'none')
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell['source']));self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
            values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,(ast.Constant,ast.Dict))}
            if 'SOURCE_CODE' in values:self.assertEqual(values['SOURCE_CODE'],(HERE/'package_dataset.py').read_text(encoding='utf-8'))
            if 'INVENTORY' in values:self.assertEqual(values['INVENTORY'],json.loads((HERE/'upload_inventory.json').read_text(encoding='utf-8')))
            if 'CREATE_DATASET_SCRIPT' in values:
                self.assertIn('public=False',values['CREATE_DATASET_SCRIPT']);self.assertIn('convert_to_csv=False',values['CREATE_DATASET_SCRIPT'])
                self.assertIn("dir_mode='skip'",values['CREATE_DATASET_SCRIPT']);self.assertIn('creation rejected',values['CREATE_DATASET_SCRIPT'])

    def test_token_redacted_and_only_in_authenticated_child_environment(self):
        notebook=json.loads((HERE/'SAVE_NATIVE_DETAIL_HL_15K_DATASET.ipynb').read_text(encoding='utf-8'))
        code=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
        fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef))
        namespace=dict(os=os,subprocess=subprocess,token='synthetic-secret')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<test>','exec'),namespace)
        with patch.object(subprocess,'Popen') as popen,redirect_stdout(io.StringIO()) as output:
            process=popen.return_value.__enter__.return_value;process.stdout=['failure synthetic-secret\n'];process.wait.return_value=1
            with self.assertRaises(RuntimeError):namespace['run_cli'](['python','upload'])
            self.assertNotIn('synthetic-secret',output.getvalue());self.assertIn('[REDACTED]',output.getvalue())
            self.assertEqual(popen.call_args.kwargs['env']['KAGGLE_API_TOKEN'],'synthetic-secret')
            self.assertNotIn('synthetic-secret',str(popen.call_args.args))
            process.wait.return_value=0;namespace['run_cli'](['python','pip'],authenticated=False)
            self.assertNotIn('KAGGLE_API_TOKEN',popen.call_args.kwargs['env'])


if __name__=='__main__':unittest.main(verbosity=2)
