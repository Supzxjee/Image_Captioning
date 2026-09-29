import unittest

from captioning.multiscore_reranking import rerank_multiscore


class MultiscoreRerankingTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            'image_id': 1,
            'candidates': [
                {'caption': 'a dog', 'logprob': -1.0, 'avg_logprob': -0.2,
                 'clipscore': 0.4},
                # avg_logprob deliberately favors rank 1. A zero-auxiliary
                # baseline must still follow the beam's raw logprob ordering.
                {'caption': 'a person', 'logprob': -2.0, 'avg_logprob': -0.1,
                 'clipscore': 0.9},
            ],
        }

    def test_zero_auxiliary_weights_preserve_rank_zero(self):
        selected, _ = rerank_multiscore(self.record, {'person': 0.9}, 0.0, 0.0)
        self.assertEqual(selected['original_rank'], 0)

    def test_clip_and_object_evidence_can_change_selection(self):
        selected, enriched = rerank_multiscore(
            self.record, {'person': 0.9}, clip_weight=0.45, object_weight=0.45)
        self.assertEqual(selected['original_rank'], 1)
        self.assertGreater(enriched[1]['combined_score'], enriched[0]['combined_score'])

    def test_rejects_invalid_weights_and_missing_scores(self):
        with self.assertRaises(ValueError):
            rerank_multiscore(self.record, {}, 0.8, 0.3)
        del self.record['candidates'][0]['clipscore']
        with self.assertRaisesRegex(ValueError, 'lacks clipscore'):
            rerank_multiscore(self.record, {}, 0.1, 0.1)


if __name__ == '__main__':
    unittest.main()
