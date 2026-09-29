import json
import tempfile
import unittest
from pathlib import Path

from prepare_chair_comparison import prepare
from summarize_chair_comparison import summarize


class ChairComparisonTests(unittest.TestCase):
    def test_prepare_uses_coco_ids_and_rank_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = {
                'metadata': {'count': 2, 'split': 'test'},
                'data': [
                    {'image_id': 1001, 'coco_id': 41,
                     'candidates': [{'caption': 'a dog', 'logprob': -1.0}]},
                    {'image_id': 1002, 'coco_id': 42,
                     'candidates': [{'caption': 'a cat', 'logprob': -1.0}]},
                ],
            }
            occ = [
                {'image_id': 1002, 'caption': 'a small cat'},
                {'image_id': 1001, 'caption': 'a dog'},
            ]
            candidates_path, occ_path = root / 'candidates.json', root / 'occ.json'
            candidates_path.write_text(json.dumps(candidates), encoding='utf-8')
            occ_path.write_text(json.dumps(occ), encoding='utf-8')
            manifest = prepare(candidates_path, occ_path, root / 'out')
            baseline = json.loads(Path(manifest['baseline']).read_text(encoding='utf-8'))
            reranked = json.loads(Path(manifest['occ']).read_text(encoding='utf-8'))
            self.assertEqual(baseline, [
                {'image_id': 41, 'caption': 'a dog'},
                {'image_id': 42, 'caption': 'a cat'},
            ])
            self.assertEqual(reranked[1], {'image_id': 42, 'caption': 'a small cat'})
            self.assertEqual(manifest['changed_captions'], 1)

    def test_summary_reports_paired_transitions(self):
        baseline = {
            'overall_metrics': {'CHAIRs': 0.5, 'CHAIRi': 0.25, 'Recall': 0.6},
            'sentences': [
                {'image_id': 1, 'caption': 'a dog', 'metrics': {'CHAIRs': 1}},
                {'image_id': 2, 'caption': 'a cat', 'metrics': {'CHAIRs': 0}},
            ],
        }
        occ = {
            'overall_metrics': {'CHAIRs': 0.0, 'CHAIRi': 0.0, 'Recall': 0.7},
            'sentences': [
                {'image_id': 1, 'caption': 'a person', 'metrics': {'CHAIRs': 0}},
                {'image_id': 2, 'caption': 'a cat', 'metrics': {'CHAIRs': 0}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / 'before.json', root / 'after.json'
            before.write_text(json.dumps(baseline), encoding='utf-8')
            after.write_text(json.dumps(occ), encoding='utf-8')
            result = summarize(before, after)
        self.assertEqual(result['sentence_transitions']['improved_to_clean'], 1)
        self.assertEqual(result['sentence_transitions']['both_clean'], 1)
        self.assertAlmostEqual(result['delta_occ_minus_baseline']['CHAIRs'], -0.5)
        self.assertEqual(result['changed_captions'], 1)


if __name__ == '__main__':
    unittest.main()
