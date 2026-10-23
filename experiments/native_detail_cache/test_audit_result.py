"""Full-cache receipt audit regressions; no HDF5/model execution."""
import copy
import json
from pathlib import Path
import unittest
from audit_result import audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.loaded=json.loads((Path(__file__).resolve().parent/'review/loaded.json').read_text(encoding='utf-8'))

    def test_completed_review_passes(self):self.assertEqual(audit(self.loaded)['status'],'RECEIPT_AUDIT_PASS')

    def test_incomplete_receipt_rejected(self):
        loaded=copy.deepcopy(self.loaded);r=json.loads(loaded['files']['cache_receipt.json']);r['shards'].pop()
        loaded['files']['cache_receipt.json']=json.dumps(r)
        with self.assertRaises(AssertionError):audit(loaded)

    def test_changed_saved_source_rejected(self):
        loaded=copy.deepcopy(self.loaded)
        name=next(n for n in loaded['files'] if n.startswith('notebook_attempts/') and n.endswith('/run_full_cache.py'))
        loaded['files'][name]+='\n# changed\n'
        with self.assertRaises(AssertionError):audit(loaded)


if __name__=='__main__':unittest.main(verbosity=2)
