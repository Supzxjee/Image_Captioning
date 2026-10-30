"""End-to-end JSON summary recovery with fake saved outputs; no neural/scorer execution."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from kv_data import save
from kv_pilot_evaluate import evaluate

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_cached_outputs_write_summary_with_numpy_counts_without_model_or_scorer(self):
        base=HERE/'review';base.mkdir(exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='summary-',dir=base)).resolve()
        try:
            out=root/'out';out.mkdir();prepared=root/'prepared';prepared.mkdir()
            rows=[dict(image_id=i,coco_id=i,references=['a dog']) for i in range(1000)]
            save(prepared/'metadata.json',dict(rows=rows,dev_indices=list(range(1000))))
            save(prepared/'cpu_preparation.json',dict(status='FAKE_TEST_RECEIPT'))
            held=[dict(fact_id=str(i)) for i in range(99)]
            save(prepared/'auxiliary_facts.json',dict(partitions=dict(held=held),multipositive=dict(held=[])))
            worker=dict(status='ENDPOINT_500_COMPLETE',updates=500,initial_digest='init',final_digest='end',
                        initial_frozen_digest='fixed',final_frozen_digest='fixed',common_text_sha256='text')
            for rank in (0,1):
                save(out/f'worker_rank{rank}.json',worker)
                save(out/f'held_rank{rank}.json',dict(records=held[rank::2]))
                save(out/f'reader_off_rank{rank}.json',[dict(coco_id=i,caption='a dog') for i in range(rank,64,2)])
            for condition,n in (('C0',10),('end',8)):
                predictions=[dict(image_id=i,coco_id=i,caption='a dog') for i in range(1000)]
                for rank in (0,1):save(out/f'{condition}_rank{rank}.json',predictions[rank::2])
                save(out/f'{condition}_per_image_cider.json',[dict(image_id=i,coco_id=i,CIDEr=1.) for i in range(1000)])
                save(out/f'{condition}_chair_input.json',[dict(image_id=i,caption='a dog') for i in range(1000)])
                sentences=[dict(image_id=i,mscoco_hallucinated_words=['dog'] if i<n else [],
                    mscoco_generated_words=['dog'],mscoco_gt_words=['cat'] if i<n else ['dog']) for i in range(1000)]
                save(out/f'{condition}_chair_output.json',dict(sentences=sentences,
                    overall_metrics=dict(CHAIRs=n/1000,CHAIRi=n/1000,Recall=(1000-n)/1000)))
            (out/'checkpoint.bin').write_bytes(b'fake checkpoint bytes; never loaded')
            save(out/'preparation.json',dict(payload_sha256={},effective_held_prompt_content_sha256='overlay'))
            cfg=dict(output=str(out),prepared_root=str(prepared),arm='D',source_commit='old',source_assets={})
            with patch('kv_pilot_evaluate.semantic_report',return_value=dict(effective_score=dict(
                    macro_family_auc=.7,query_concordance=.7,valid_query_groups=np.int64(16)))), \
                 patch('kv_pilot_evaluate.paired_ci',return_value=[-.1,.1]),patch('builtins.print'):
                evaluate(cfg,reuse_scores=True)
            result=json.loads((out/'summary.json').read_bytes())
            self.assertEqual(result['status'],'ENDPOINT_500_EVALUATED_WAIT_MATCHED_DE')
            self.assertEqual(result['false_caption_pairing']['repaired'],2)
            self.assertEqual(result['semantic']['effective_score']['valid_query_groups'],16)
            self.assertTrue(result['cached_official_scores_reused'])
        finally:
            assert root.parent==base.resolve() and root.name.startswith('summary-')
            shutil.rmtree(root)


if __name__=='__main__':unittest.main(verbosity=2)
