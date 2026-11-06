"""Safe stored-data and synthetic fixtures; no metric/model execution."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import score_plain_csa10_extra as s

HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_exact_frozen_data_and_baselines(self):
        m = s.cpu.read(HERE/'PLAIN_CSA10_EXTRA_INPUTS.json')
        rows = s.load_snapshot(HERE/'plain_csa10_test_snapshot.json.gz', m)
        self.assertEqual(len(rows),5000)
        self.assertEqual(sum(len(r['references']) for r in rows),25010)
        self.assertEqual(sum(len(r['NoCSA10'].split())<=2 for r in rows),1853)
        self.assertEqual(sum(len(r['CSA10'].split())<=2 for r in rows),388)
        self.assertEqual(m['policies'],['NoCSA10','CSA10'])

    def test_hash_and_duplicate_rejected(self):
        m = s.cpu.read(HERE/'PLAIN_CSA10_EXTRA_INPUTS.json')
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'bad.gz'
            p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):s.load_snapshot(p,m)
            rows = [dict(coco_id=1,references=['a dog']*5,NoCSA10='dog',CSA10='a dog')]*2
            raw = json.dumps(rows).encode();packed = gzip.compress(raw,mtime=0);p.write_bytes(packed)
            m={**m,'images':2,'snapshot_sha256':hashlib.sha256(packed).hexdigest(),'snapshot_json_sha256':hashlib.sha256(raw).hexdigest()}
            with self.assertRaises(ValueError):s.load_snapshot(p,m)

    def test_actual_shape_with_mock_tokenizer(self):
        import sys,types
        fake=types.ModuleType('pycocoevalcap.tokenizer.ptbtokenizer')
        class MockPTB:
            def tokenize(self,items):return {i:[r['caption'] for r in rows] for i,rows in items.items()}
        fake.PTBTokenizer=MockPTB
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'pycocoevalcap.tokenizer.ptbtokenizer':fake}):
            m=s.cpu.read(HERE/'PLAIN_CSA10_EXTRA_INPUTS.json');out=Path(temp)
            s.tokenize(out,m)
            receipt=s.cpu.read(out/'input_receipt.json')
            self.assertEqual(receipt['payload_sha256'],m['arms']['BASELINES']['payload_sha256'])
            for p in ('NoCSA10','CSA10'):
                self.assertEqual(len(s.cpu.read(out/(p+'_ptb.json'))),5000)
            self.assertEqual(receipt['images'],5000)

    def test_wrapper_uses_existing_scorers_without_model_imports(self):
        tree=ast.parse((HERE/'score_plain_csa10_extra.py').read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):self.assertTrue(all(a.name.split('.')[0] not in ('torch','transformers') for a in n.names))
        text=(HERE/'score_plain_csa10_extra.py').read_text(encoding='utf-8-sig')
        self.assertIn("cpu.metric(a.output, 'BASELINES', m, a.mode)",text)
        self.assertIn('PLAIN_CSA10_TEST_EIGHT_METRICS_COMPLETE',text)


if __name__=='__main__':unittest.main(verbosity=2)
