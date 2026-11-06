"""Package and synthetic paired-statistic checks; no caption scorer/model execution."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from score_prompt_scratch_quality import checked_payload,paired

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_sealed_full5000_and_tampered_payload_rejected(self):
        path=HERE/'prompt_scratch_valid5000_saved.json.gz';manifest=json.loads((HERE/'PROMPT_SCRATCH_QUALITY_INPUTS.json').read_text(encoding='utf-8'))
        payload,refs=checked_payload(path,manifest);self.assertEqual(len(refs),5000)
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'changed.gz';p.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Changed sealed'):checked_payload(p,manifest)
            payload['models']['PROMPT9']['rows'][0]['coco_id']=payload['models']['PROMPT9']['rows'][1]['coco_id']
            raw=gzip.compress(json.dumps(payload).encode());p.write_bytes(raw);manifest['payload_sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError,'caption IDs'):checked_payload(p,manifest)

    def test_paired_ids_direction_and_exact_constant_ci(self):
        a={1:dict(CIDEr=0.,hallucination=True),2:dict(CIDEr=1.,hallucination=False)}
        b={2:dict(CIDEr=1.5,hallucination=True),1:dict(CIDEr=.5,hallucination=False)}
        r=paired(a,b,resamples=100);self.assertEqual(r['CIDEr_delta'],.5);self.assertEqual(r['CIDEr_paired95'],[.5,.5]);self.assertEqual((r['repaired'],r['introduced'],r['CHAIRs_delta']),(1,1,0.))
        with self.assertRaises(ValueError):paired(a,{})

    def test_cpu_sources_parse(self):
        for n in ('score_prompt_scratch_quality.py','make_prompt_scratch_quality_notebook.py'):ast.parse((HERE/n).read_text(encoding='utf-8'))
        source=(HERE/'score_prompt_scratch_quality.py').read_text(encoding='utf-8');self.assertNotIn('import torch',source);self.assertNotIn('from transformers',source)
        self.assertIn('zip(refs,values)',source);self.assertIn('>1e-7',source)


if __name__=='__main__':unittest.main(verbosity=2)
