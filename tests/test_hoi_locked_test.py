"""Guard against tuning lambda on test and accidentally using validation rows."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from run_hoi_caption_validation import load_test_selection, score_and_select
from captioning.hoi_caption_reranking import METHOD


class LockedTestProtocol(unittest.TestCase):
    def test_validation_decision_rejects_modified_weight(self):
        original=Path(__file__).resolve().parents[1]/'audit/hoi_validation_locked_selection.json'
        self.assertEqual(load_test_selection(original)['upt_weight'],.1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'decision.json'
            decision=json.loads(original.read_text());decision['upt_weight']=.2
            path.write_text(json.dumps(decision))
            with self.assertRaisesRegex(ValueError,'hash'):
                load_test_selection(path)

    def test_test_predictions_are_invariant_to_reference_and_metric_changes(self):
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider import Cider
        candidates=[{'caption':'a man riding a horse','logprob':-2.,'clipscore':.7,
                     'hoi_evidence':{'method':METHOD,'feature':.8}},
                    {'caption':'a man near a horse','logprob':-1.,'clipscore':.7,
                     'hoi_evidence':{'method':METHOD,'feature':-.8}}]
        bundle={'metadata':{'split':'test'},'data':[{'image_id':9001,'coco_id':101,
                'filename':'test.jpg','candidates':candidates}]}
        decisions=[]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);gt=out/'gt.json';det=out/'det.json';det.write_text('{}')
            def command(args,cwd=None):
                if '--save_path' in args:
                    source=Path(args[args.index('--cap_file')+1])
                    rows=json.loads(source.read_text())
                    Path(args[args.index('--save_path')+1]).write_text(json.dumps({
                        'overall_metrics':{'CHAIRs':0},
                        'sentences':[{**r,'metrics':{'CHAIRs':0}} for r in rows]}))
            for reference,score in [('a man riding a horse',1.),('an unrelated caption',999.)]:
                gt.write_text(json.dumps({'annotations':[{'image_id':9001,'caption':reference}]}))
                with patch('captioning.object_consistency.index_detections',return_value={'test.jpg':{'person':.9,'horse':.9}}), \
                     patch.object(PTBTokenizer,'tokenize',side_effect=lambda x:x), \
                     patch.object(Cider,'compute_score',side_effect=AssertionError('Test must not tune a grid')), \
                     patch('captioning.metrics.score_files',return_value={'CIDEr':score}), \
                     patch('run_hoi_caption_validation.command',side_effect=command):
                    result=score_and_select(bundle,gt,det,out,out,locked={'upt_weight':.1})
                self.assertEqual(result['trials'],[])
                self.assertFalse(result['retuned_on_test'])
                self.assertEqual(result['selection']['upt_weight'],.1)
                decisions.append([(out/f'{name}_predictions.json').read_text()
                                  for name in ('rank0','fixed_rerank','upt_rerank')])
            self.assertEqual(decisions[0],decisions[1])
            self.assertFalse((out/'locked_validation_selection.json').exists())


if __name__=='__main__': unittest.main()
