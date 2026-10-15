import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scst_preflight import run_preflight


class PreflightTests(unittest.TestCase):
    def test_exact_file_discovery_and_success_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'preflight.log'
            with patch('scst_preflight.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'SCST PASS\n')) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    output = run_preflight('.', log)
            args = run.call_args.args[0]
            self.assertIn('discover', args)
            self.assertEqual(args[args.index('-p')+1], 'test_scst.py')
            self.assertEqual(output, log.read_text())

    def test_actual_failure_is_saved_and_in_exception_without_bypassing(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'preflight.log'
            detail = 'FAIL: test_reward\nAssertionError: reward differs\nFAILED (failures=1)\n'
            with patch('scst_preflight.subprocess.run', return_value=subprocess.CompletedProcess([], 1, detail)):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'AssertionError: reward differs'):
                        run_preflight('.', log)
            self.assertEqual(log.read_text(), detail)


if __name__ == '__main__': unittest.main()
