"""Reject mixed or incomplete saved arms. Synthetic files only; no model imports."""
import json
from pathlib import Path
import tempfile
import unittest
from compare_gain import compare


class ComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace_temp=Path(__file__).resolve().parent/'review'
        cls.workspace_temp.mkdir(exist_ok=True)

    def test_reject_mixed_preparation_before_metrics(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_temp) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(self.workspace_temp.resolve()))
            root=Path(temp);paths={}
            for arm in 'CHT':
                p=root/arm;p.mkdir();paths[arm]=str(p)
                s=dict(status='COMPLETE_WAIT_MATCHED_CHT',arm=arm,updates=500,test_used=False,
                    preparation_receipt_sha256='different' if arm=='T' else 'shared',sources={'source':'fixed'},policy={'policy':'v0.3'})
                (p/'summary.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'Unmatched preparation'):compare(paths,root/'result.json')
            self.assertFalse((root/'result.json').exists())

    def test_reject_incomplete_or_wrong_arm_before_comparison(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_temp) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(self.workspace_temp.resolve()))
            root=Path(temp);paths={}
            for arm in 'CHT':
                p=root/arm;p.mkdir();paths[arm]=str(p)
                (p/'summary.json').write_text(json.dumps(dict(status='COMPLETE_WAIT_MATCHED_CHT',arm=arm,updates=499 if arm=='H' else 500,test_used=False)))
            with self.assertRaisesRegex(ValueError,'Incomplete'):compare(paths,root/'result.json')


if __name__=='__main__':unittest.main(verbosity=2)
