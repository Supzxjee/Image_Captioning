import unittest

from captioning.object_consistency import (
    extract_caption_objects, index_detections, object_consistency, rerank_record,
)


class ObjectConsistencyTests(unittest.TestCase):
    def test_aliases_and_longest_phrase(self):
        labels = extract_caption_objects('Two people eat a hot dog beside their bikes.')
        self.assertEqual(set(labels), {'person', 'hot dog', 'bicycle'})
        self.assertNotIn('dog', labels)

    def test_detection_threshold_and_consistency(self):
        indexed = index_detections({
            '/old/x.jpg': {'objects': [
                {'label': 'person', 'conf': 0.9},
                {'label': 'dog', 'conf': 0.4},
            ]},
        }, min_confidence=0.5)
        result = object_consistency('a man with a dog', indexed['x.jpg'])
        self.assertEqual(result['supported'], ['person'])
        self.assertEqual(result['hallucinated'], ['dog'])
        self.assertAlmostEqual(result['score'], -0.05)

    def test_zero_weight_preserves_beam_and_occ_can_change_it(self):
        record = {
            'image_id': 1,
            'candidates': [
                {'caption': 'a dog', 'logprob': -1.0},
                {'caption': 'a person', 'logprob': -2.0},
            ],
        }
        baseline, _ = rerank_record(record, {'person': 0.9}, weight=0.0)
        reranked, _ = rerank_record(record, {'person': 0.9}, weight=0.8)
        self.assertEqual(baseline['original_rank'], 0)
        self.assertEqual(reranked['original_rank'], 1)


if __name__ == '__main__':
    unittest.main()
