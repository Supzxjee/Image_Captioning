"""Author-only packaging regression: execute DATA tests in an isolated source bundle."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from make_grounding_pilot_notebooks import FILES

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_data_tests_pass_using_only_declared_notebook_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in FILES:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(HERE/name,target)
            # A clean PYTHONPATH prevents accidental access to unbundled repo files.
            import os
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,str(root/'test_grounding_pilot_data.py')],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertIn('Ran 7 tests',result.stderr)
            self.assertIn('OK',result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
