"""Safe stdlib fixtures only; never import or run metric scorers."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from score_prompt_test_extra import read, save, sha, verify, combine, KEYS

HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        values = {
            'test_references.json': [dict(coco_id=i, references=['a dog']*5) for i in (1, 2)],
            'test_lock.json': dict(checkpoint_sha256='abc'),
            'run_status.json': dict(complete=True, arm='CONTROL', training=False, test_tuning=False),
            'result.json': dict(status='PROMPT_TEST5000_COMPLETE', arm='CONTROL', images=2,
                training=False, checkpoint_sha256='abc', metrics={p: dict(CIDEr=1.) for p in ('normal', 'fixed_rerank')}),
        }
        for p in ('normal', 'fixed_rerank'):
            values[p+'_captions.json'] = [dict(coco_id=i, caption='a dog') for i in (1, 2)]
        for n, v in values.items():
            save(self.root/n, v)
        self.manifest = dict(images=2, policies=['normal', 'fixed_rerank'], arms={'CONTROL': dict(
            version='owner/source/184', checkpoint_sha256='abc',
            metrics=values['result.json']['metrics'],
            payload_sha256={n: sha(self.root/n) for n in values})})
        self.manifest['arms']['CONTROL']['payload_sha256']['test_lock.json'] = ''
        lock = read(self.root/'test_lock.json')
        lock['references_sha256'] = sha(self.root/'test_references.json')
        save(self.root/'test_lock.json', lock)
        self.repin('test_lock.json')

    def tearDown(self):
        self.tmp.cleanup()

    def repin(self, name):
        self.manifest['arms']['CONTROL']['payload_sha256'][name] = sha(self.root/name)

    def test_nested_exact_complete_inputs(self):
        refs, caps, files = verify(self.root, 'CONTROL', self.manifest)
        self.assertEqual(len(refs), 2)
        self.assertEqual(set(caps), {'normal', 'fixed_rerank'})
        self.assertEqual(len(files), 6)

    def test_changed_or_missing_input_rejected(self):
        (self.root/'normal_captions.json').write_text('[]')
        with self.assertRaises(ValueError):
            verify(self.root, 'CONTROL', self.manifest)

    def test_sixth_reference_preserved(self):
        refs = read(self.root/'test_references.json')
        refs[0]['references'].append('a sixth original description')
        save(self.root/'test_references.json', refs)
        self.repin('test_references.json')
        lock = read(self.root/'test_lock.json')
        lock['references_sha256'] = sha(self.root/'test_references.json')
        save(self.root/'test_lock.json', lock)
        self.repin('test_lock.json')
        checked, _, _ = verify(self.root, 'CONTROL', self.manifest)
        self.assertEqual(len(checked[0]['references']), 6)

    def test_duplicate_cohort_rejected_even_if_rehashed(self):
        save(self.root/'normal_captions.json', [dict(coco_id=1, caption='a dog')]*2)
        self.repin('normal_captions.json')
        with self.assertRaises(ValueError):
            verify(self.root, 'CONTROL', self.manifest)

    def test_no_partial_completion(self):
        save(self.root/'input_receipt.json', dict(arm='CONTROL', tokenized_sha256={}))
        save(self.root/'normal_basic.json', dict(images=4999, tokenized_sha256={}, metrics={}))
        with self.assertRaises(ValueError):
            combine(self.root, 'CONTROL', self.manifest)
        self.assertFalse((self.root/'metrics_x100.csv').exists())

    def test_sources_parse_no_model_imports_and_all_keys(self):
        tree = ast.parse((HERE/'score_prompt_test_extra.py').read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                self.assertTrue(all(a.name.split('.')[0] not in ('torch', 'transformers') for a in n.names))
            if isinstance(n, ast.ImportFrom):
                self.assertNotIn((n.module or '').split('.')[0], ('torch', 'transformers'))
        self.assertEqual(len(KEYS), 8)
        self.assertIn('SPICE', KEYS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
