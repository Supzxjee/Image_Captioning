import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_scst_object_locked_test import test_images,validate_selected
from run_scst_comparison import evaluate


class LockedTestTests(unittest.TestCase):
    def test_only_test_rows_and_original_eval_ids_are_kept(self):
        rows=[{'split':s,'filename':f'COCO_val2014_{i:012d}.jpg','filepath':'val2014'}
              for i,s in enumerate(['train','val']+['test']*5000)]
        result=test_images({'images':rows})
        self.assertEqual(len(result),5000);self.assertEqual(result[0]['image_id'],2)
        self.assertEqual(result[-1]['image_id'],5001)

    def test_missing_or_duplicate_test_ids_rejected(self):
        with self.assertRaises(ValueError):test_images({'images':[]})
        rows=[{'split':'test','filename':'COCO_val2014_000000000001.jpg','filepath':'val2014'}]*5000
        with self.assertRaises(ValueError):test_images({'images':rows})

    def test_selected_checkpoint_profile_rejects_wrong_coefficient(self):
        expected={'visual_adapter':'qformer','num_visual_queries':32,'qformer_layers':2,
              'optimization':'scst_ce_object','scst_weight':.05,'ce_weight':1.,'object_reward_weight':.5,
              'object_reward_parser':'aliases_v2','object_reward_penalty':'unique_count_v2',
              'source_checkpoint_sha256':'155106910d49715f8065a444baca3321bb5e668a1f684ea3c78f860de8a39012'}
        self.assertEqual(validate_selected(expected),expected)
        with self.assertRaises(ValueError):validate_selected({**expected,'object_reward_weight':1.})

    def evaluate_fixture(self,split):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoint').write_bytes(b'fixture');(root/'detections.json').write_text('{}')
            (root/'data.json').write_text(json.dumps({'images':[{'sentences':[{'raw':'a dog'}]*5}]}))
            image={'image_id':0,'coco_id':7,'filename':'a.jpg','filepath':'val2014'}
            cfg={'output':str(root),'dataset_json_path':str(root/'data.json'),'detections':str(root/'detections.json'),
                 'chair_directory':str(root),'visual_cache':'fixture','clip_revision':'fixture',
                 'evaluation_checkpoints':{'original':str(root/'checkpoint')},'groups':{split:[image]}}
            if split=='test':cfg['evaluation_split']='test'
            record={**image,'candidates':[{'caption':'a dog','avg_logprob':-1.,'clipscore':.5}]}
            def command(args,cwd=None):
                if '--output' in args:
                    Path(args[args.index('--output')+1]).write_text(json.dumps({'data':[record]}))
                else:
                    Path(args[args.index('--save_path')+1]).write_text(json.dumps({'sentences':[{'image_id':7}],
                        'overall_metrics':{'CHAIRs':0.,'CHAIRi':0.,'Recall':1.}}))
            with patch('run_scst_comparison.run_workers',return_value=[record]) as workers, \
                 patch('run_scst_comparison.command',side_effect=command), \
                 patch('captioning.metrics.score_files',return_value={'CIDEr':1.}) as metric, \
                 patch('captioning.object_consistency.index_detections',return_value={'a.jpg':[]}), \
                 patch('captioning.multiscore_reranking.rerank_multiscore',return_value=[record['candidates'][0]]):
                result=evaluate(cfg,root)
                self.assertEqual(len(result),2)
                self.assertEqual(workers.call_args.args[0]['split'],split)
                self.assertTrue((root/f'{split}_gt.json').is_file())
                for invocation in metric.call_args_list:
                    self.assertEqual(invocation.args[1],root/f'{split}_gt.json')
                self.assertEqual(json.loads((root/'original/beams.json').read_text())['metadata']['split'],split)

    def test_test_evaluation_uses_test_dataframe_and_test_groundtruth(self):self.evaluate_fixture('test')
    def test_legacy_validation_default_unchanged(self):self.evaluate_fixture('val')


if __name__=='__main__':unittest.main()
