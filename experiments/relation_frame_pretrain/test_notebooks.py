"""Mock notebook subprocesses only: verify fail/success archives without any job/model."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_static_pinned_cells(self):
        package=json.loads((HERE/'NOTEBOOK_PACKAGE.json').read_text())
        self.assertFalse(package['local_neural_execution']);self.assertFalse(package['kaggle_launched'])
        for entry in package['notebooks']:
            path=HERE/entry['file'];nb=json.loads(path.read_text(encoding='utf-8'))
            self.assertLess(path.stat().st_size,1000000)
            for cell in nb['cells']:
                if cell['cell_type']=='code':
                    compile(cell['source'],path.name,'exec')
                    self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
            text=''.join(c['source'] for c in nb['cells'])
            self.assertIn('/kaggle/input/datasets/ducanh2403/relation-frames-v2-cache-v105',text)
            self.assertIn(package['source_commit'],text)

    def test_mock_worker_failure_archived_without_success(self):self.wrapper(False)
    def test_mock_worker_success_archived(self):self.wrapper(True)

    def wrapper(self,success):
        nb=json.loads((HERE/'PRETRAIN_RELPOS_2GPU.ipynb').read_text(encoding='utf-8'))
        source=[c['source'] for c in nb['cells'] if c['cell_type']=='code'][-1]
        parent=HERE.parent/'relation_frames_inventory/review'
        with tempfile.TemporaryDirectory(prefix='notebook-mock-',dir=parent) as temp:
            root=Path(temp);config=root/'config.json';config.write_text('{}')
            out=root/'output';log=root/'log.txt';review=root/'review.zip'
            if success:
                out.mkdir();(out/'result.json').write_text(json.dumps({'status':'ENDPOINT_200_COMPLETE'}))
                (out/'checkpoint.bin').write_bytes(b'mocked, never a model')
            class FakeProcess:
                def __init__(self):self.stdout=io.StringIO('mock worker, no process executed\n')
                def wait(self):return 0 if success else 1
            fake=SimpleNamespace(Popen=lambda *args,**kwargs:FakeProcess(),CalledProcessError=subprocess.CalledProcessError,
                PIPE=subprocess.PIPE,STDOUT=subprocess.STDOUT)
            env=dict(Path=Path,LOG=log,SOURCE=root,REVIEW=review,OUTPUT=out,CONFIG=config,ARM='RelPos',
                ASSETS={},SOURCE_COMMIT='mock',subprocess=fake,sys=sys,threading=threading,json=json,zipfile=zipfile)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                if success:exec(source,env)
                else:
                    with self.assertRaises(subprocess.CalledProcessError):exec(source,env)
            with zipfile.ZipFile(review) as z:
                status=json.loads(z.read('execution_status.json'))
                self.assertEqual(status['success'],success)
                self.assertIn('notebook.log',z.namelist())
                self.assertNotIn('checkpoint.bin',z.namelist())
            if not success:self.assertNotIn('KEEP CHECKPOINT:',output.getvalue())


if __name__=='__main__':unittest.main(verbosity=2)
