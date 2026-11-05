"""Safe boundary/static fixtures, no pretrained model or Torch imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import types
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
        regex=cache.source_pattern()
        for name in cache.SOURCE_FILES:self.assertRegex('folder/'+name,regex)
        self.assertIsNone(re.search(regex,'notebook.log'))
        self.assertIsNone(re.search(regex,'receiptXjson'))
        for name in ('prompt_text_cache.py','make_prompt_text_cache_notebook.py'):
            ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))

    def test_download_selector_covers_entire_receipt_contract(self):
        required={'summary.json','cohort.json','prompt_links.jsonl','prompt_texts.json','review_examples.json'}
        self.assertEqual(set(cache.SOURCE_FILES),required|{'receipt.json'})
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);hashes={}
            for n in sorted(required):
                b=json.dumps({'filename':n}).encode();(root/n).write_bytes(b);hashes[n]=hashlib.sha256(b).hexdigest()
            receipt=dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',payload_sha256=hashes)
            b=json.dumps(receipt).encode();(root/'receipt.json').write_bytes(b)
            with patch.object(cache,'RECEIPT_SHA',hashlib.sha256(b).hexdigest()):
                self.assertEqual(cache.checked_source(root)[0],root)
                (root/'review_examples.json').unlink()
                with self.assertRaisesRegex(ValueError,'review_examples.json'):cache.checked_source(root)
            tree=ast.parse(Path(__file__).with_name('make_prompt_text_cache_notebook.py').read_text(encoding='utf-8'))
            final=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='final' for t in n.targets))
            lists=[n for n in ast.walk(ast.parse(final)) if isinstance(n,ast.List) and any(isinstance(x,ast.Constant) and x.value=='--file-pattern' for x in n.elts)]
            self.assertEqual(len(lists),1)
            self.assertIsInstance(lists[0].elts[-1],ast.Call)
            self.assertEqual(lists[0].elts[-1].func.id,'source_pattern')

    def test_recovery_downloads_only_missing_review_and_preserves_failed_output(self):
        import sys
        code=Path(__file__).with_name('RECOVER_PROMPT_TEXT_167_CELL.py').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as name:
            top=Path(name);download=top/'download167';root=download/'mapping';root.mkdir(parents=True)
            failed=top/'failed';failed.mkdir();(failed/'notebook.log').write_text('preserve')
            values={n:json.dumps({'file':n}).encode() for n in cache.SOURCE_FILES if n!='receipt.json'}
            receipt=json.dumps(dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',payload_sha256={n:hashlib.sha256(b).hexdigest() for n,b in values.items()})).encode()
            (root/'receipt.json').write_bytes(receipt)
            for n,b in values.items():
                if n!='review_examples.json':(root/n).write_bytes(b)
            calls=[]
            def run(args,timeout,environment):
                pattern=args[args.index('--file-pattern')+1]
                self.assertRegex('mapping/review_examples.json',pattern)
                self.assertIsNone(re.search(pattern,'mapping/prompt_links.jsonl'))
                self.assertIn('ducanh2403/2-vi-cai-kia-qua-nhieu/167',args)
                self.assertEqual(environment['KAGGLE_API_TOKEN'],'fixture-only')
                (root/'review_examples.json').write_bytes(values['review_examples.json']);calls.append(args)
            secrets=types.ModuleType('kaggle_secrets');secrets.UserSecretsClient=lambda:types.SimpleNamespace(get_secret=lambda _: 'fixture-only')
            # Keep the recovery output path inside this fixture, leaving user code unchanged.
            code=code.replace("Path('/kaggle/working/csa_prompt_text_cache_v2')","Path("+repr(str(top/'resumed'))+")")
            namespace=dict(SOURCE_ROOT=download,SOURCE=top,OUTPUT=failed,run=run,SECRET_NAME='test')
            with patch.dict(sys.modules,{'kaggle_secrets':secrets}),patch.object(cache,'RECEIPT_SHA',hashlib.sha256(receipt).hexdigest()):
                exec(compile(code,'recovery-fixture','exec'),namespace)
            self.assertEqual(len(calls),1);self.assertEqual(namespace['LOCAL_SOURCE_ROOT'],str(root))
            self.assertEqual((failed/'notebook.log').read_text(),'preserve')
            self.assertFalse((top/'resumed').exists())


if __name__=='__main__':unittest.main(verbosity=2)
