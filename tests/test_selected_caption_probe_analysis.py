import copy
import unittest
from analyze_selected_caption_probe import analyze


class ProbeAnalysisTests(unittest.TestCase):
    def fixture(self):
        def outcome(score, active, delta):
            return dict(candidates=[dict(caption='a person riding a bicycle', logprob=score)],
                valid_relations=active, memory_relative_change=delta,
                evidence_weights=[.02] if active else [])
        return dict(sample_count=1, off_top_changed=0, swap_top_changed=0), [dict(
            filename='x.jpg', off_top_changed=False, swap_top_changed=False,
            conditions=dict(on=outcome(-1, 1, 0), off=outcome(-1.01, 0, .001),
                            target_swap=outcome(-1.00001, 1, .000001)))]

    def test_score_change_is_not_wording_change(self):
        summary, rows = self.fixture()
        result = analyze(summary, rows)
        for condition in result['conditions'].values():
            self.assertEqual(condition['candidate_wording_or_order_changed'], 0)
            self.assertEqual(condition['images_with_candidate_logprob_changes'], 1)

    def test_inconsistent_mask_or_summary_rejected(self):
        summary, rows = self.fixture()
        broken = copy.deepcopy(rows); broken[0]['conditions']['off']['valid_relations'] = 1
        with self.assertRaises(ValueError): analyze(summary, broken)
        summary['off_top_changed'] = 1
        with self.assertRaises(ValueError): analyze(summary, rows)


if __name__ == '__main__': unittest.main()
