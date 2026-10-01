import unittest

import torch

from captioning.models import ObjectSemanticAlignment
from captioning.object_concepts import (
    align_object_concept_cache,
    build_object_concept_index,
)


class ObjectConceptCacheTests(unittest.TestCase):
    def test_selects_unique_high_confidence_objects_in_rank_order(self):
        source = {'/old/a.jpg': {'objects': [
            {'label': 'Dog', 'conf': 0.7},
            {'label': 'person', 'conf': 0.9},
            {'label': 'dog', 'conf': 0.8},
            {'label': 'chair', 'conf': 0.2},
        ]}}
        result = build_object_concept_index(
            source, [{'filename': 'a.jpg'}], min_confidence=0.5, max_objects=2)
        self.assertEqual(result['a.jpg'], [('person', 0.9), ('dog', 0.8)])

    def test_alignment_accepts_filename_keys_and_validates_shape(self):
        entry = {'tokens': torch.zeros(3, 512), 'mask': torch.tensor([1, 0, 0])}
        aligned = align_object_concept_cache({'a.jpg': entry}, ['/new/path/a.jpg'])
        self.assertIs(aligned['/new/path/a.jpg'], entry)
        with self.assertRaisesRegex(ValueError, 'O, 512'):
            align_object_concept_cache(
                {'a.jpg': {'tokens': torch.zeros(3, 8), 'mask': torch.ones(3)}},
                ['/new/path/a.jpg'])


class ObjectSemanticAlignmentTests(unittest.TestCase):
    def test_valid_objects_change_queries_and_receive_gradients(self):
        torch.manual_seed(7)
        module = ObjectSemanticAlignment(16, 4, dropout=0.0)
        queries = torch.randn(2, 5, 16, requires_grad=True)
        objects = torch.randn(2, 3, 512)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        output = module(queries, objects, mask)
        self.assertEqual(tuple(output.shape), (2, 5, 16))
        self.assertFalse(torch.equal(output, queries))
        output.sum().backward()
        self.assertGreater(module.object_projection[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.cross_attn.in_proj_weight.grad.abs().sum().item(), 0)

    def test_empty_object_row_is_exact_baseline(self):
        module = ObjectSemanticAlignment(8, 2, dropout=0.0).eval()
        queries = torch.randn(2, 4, 8)
        objects = torch.randn(2, 3, 512)
        mask = torch.tensor([[0, 0, 0], [1, 0, 0]])
        output = module(queries, objects, mask)
        self.assertTrue(torch.equal(output[0], queries[0]))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
