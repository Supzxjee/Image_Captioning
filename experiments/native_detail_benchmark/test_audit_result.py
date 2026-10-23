"""Receipt-audit regressions; no model/feature extraction."""
import copy
import json
from pathlib import Path
import unittest
from audit_result import audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded=json.loads((Path(__file__).resolve().parent/'review/loaded.json').read_text(encoding='utf-8'))

    def test_real_receipts_pass(self):
        self.assertEqual(audit(self.loaded)['status'],'AUDIT_PASS')

    def test_changed_timing_rejected(self):
        loaded=copy.deepcopy(self.loaded);summary=json.loads(loaded['files']['benchmark_summary.json'])
        summary['measured_wall_seconds']=1;loaded['files']['benchmark_summary.json']=json.dumps(summary)
        with self.assertRaises(AssertionError):audit(loaded)

    def test_changed_runtime_source_rejected(self):
        loaded=copy.deepcopy(self.loaded);loaded['files']['source/run_benchmark.py']+='\n# changed\n'
        with self.assertRaises(AssertionError):audit(loaded)


if __name__=='__main__':unittest.main(verbosity=2)
