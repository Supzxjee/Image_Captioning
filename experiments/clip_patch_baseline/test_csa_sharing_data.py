"""Author-safe sharing checks; no neural imports or execution."""
import ast
from pathlib import Path
from unittest import TestCase, main
import numpy as np
from csa_sharing_data import image_plan, benchmark_orders, local_events

HERE = Path(__file__).resolve().parent


class Checks(TestCase):
    def test_inverse_and_gradient_accumulation_reference(self):
        mapping = np.array([2, 0, 2, 1, 0, 2])
        events = np.array([5, 1, 0, 4, 3, 2])
        first, inverse = image_plan(mapping, events, True)
        np.testing.assert_array_equal(mapping[events[first]][inverse], mapping[events])
        values = np.arange(18).reshape(3, 6)
        np.testing.assert_array_equal(values[mapping[events[first]]][inverse], values[mapping[events]])
        # Index-select backward must ADD repeated caption contributions, not average.
        contributions = np.arange(36).reshape(6, 6)
        summed = np.zeros((len(first), 6), dtype=np.int64); np.add.at(summed, inverse, contributions)
        for i in range(len(first)):
            np.testing.assert_array_equal(summed[i], contributions[inverse == i].sum(0))
        positions, no_inverse = image_plan(mapping, events, False)
        np.testing.assert_array_equal(positions, np.arange(6)); self.assertIsNone(no_inverse)

    def test_no_duplicates_identity_and_invalid_events(self):
        first, inverse = image_plan(np.arange(10), np.array([4, 2, 7]), True)
        np.testing.assert_array_equal(np.array([4, 2, 7])[first][inverse], [4, 2, 7])
        for events in ([], [-1], [10], [1.2], [[1]]):
            with self.assertRaises(ValueError): image_plan(np.arange(10), events, True)

    def test_grouped_and_random_keep_all_events_and_improve_duplicates(self):
        mapping = np.repeat(np.arange(256), 5); orders = benchmark_orders(mapping)
        for order in orders.values(): np.testing.assert_array_equal(np.sort(order), np.arange(1280))
        def unique_count(order):
            return np.mean([len(image_plan(mapping, local_events(events, rank), True)[0])
                            for events in order.reshape(40, 32) for rank in (0, 1)])
        self.assertLess(unique_count(orders['grouped']), unique_count(orders['random']))
        for k, order in orders.items(): np.testing.assert_array_equal(order, benchmark_orders(mapping)[k])

    def test_tail_token_normalization_and_rank_coverage(self):
        events = np.arange(27); parts = [local_events(events, r) for r in (0,1)]
        self.assertEqual([len(p) for p in parts], [16,11])
        np.testing.assert_array_equal(np.concatenate(parts), events)
        local_gradient_sums = np.array([19., 31.]); token_counts = np.array([140, 61])
        distributed_average = (local_gradient_sums*2/token_counts.sum()).mean()
        self.assertAlmostEqual(distributed_average, local_gradient_sums.sum()/token_counts.sum())

    def test_model_shares_before_decoder_with_autograd(self):
        source = (HERE/'csa_shared_model.py').read_text(encoding='utf-8'); tree = ast.parse(source)
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertIn('encode', calls); self.assertIn('index_select', calls); self.assertIn('decode', calls)
        self.assertNotIn('detach', calls); self.assertNotIn('no_grad', source)
        self.assertNotIn('__init__', source) # no extra parameters or reinitialization
        ast.parse((HERE/'run_csa_sharing_benchmark.py').read_text(encoding='utf-8'))


if __name__ == '__main__': main(verbosity=2)
