"""Safe boundary/static fixtures, no pretrained model or Torch imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import prompt_text_cache as cache


class Checks(unittest.TestCase):
    def test_exact_payload_receipt_and_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);payload=b'[]\n';(root/'prompt_texts.json').write_bytes(payload)
            receipt=dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',payload_sha256={'prompt_texts.json':hashlib.sha256(payload).hexdigest()})
            b=json.dumps(receipt).encode();(root/'receipt.json').write_bytes(b)
            with patch.object(cache,'RECEIPT_SHA',hashlib.sha256(b).hexdigest()):
                self.assertEqual(cache.checked_source(root)[0],root)
                (root/'prompt_texts.json').write_bytes(b'changed')
                with self.assertRaises(ValueError):cache.checked_source(root)

    def test_duplicate_and_unsafe_receipt_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);receipt=dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',payload_sha256={'../escape':'0'*64})
            b=json.dumps(receipt).encode();(root/'receipt.json').write_bytes(b)
            with patch.object(cache,'RECEIPT_SHA',hashlib.sha256(b).hexdigest()):
                with self.assertRaises(ValueError):cache.checked_source(root)
                (root/'duplicate').mkdir();(root/'duplicate/receipt.json').write_bytes(b)
                with self.assertRaises(ValueError):cache.checked_source(root)

    def test_cli_pattern_and_sources_parse(self):
        regex=r'(^|/)(receipt\.json|summary\.json|cohort\.json|prompt_links\.jsonl|prompt_texts\.json)$'
        for name in ('prompt_links.jsonl','folder/prompt_texts.json','receipt.json'):self.assertRegex(name,regex)
        self.assertIsNone(re.search(regex,'notebook.log'))
        for name in ('prompt_text_cache.py','make_prompt_text_cache_notebook.py'):
            ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
