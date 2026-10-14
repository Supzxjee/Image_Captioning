import unittest
import torch
from captioning.selected_grounding_bridge import prepare_selected, pack_selected
from captioning.grounded_relations import align_grounded_relation_cache, PairGroundedRelationAlignment
from probe_selected_caption_grounding import alternate_target


class SelectedBridgeTests(unittest.TestCase):
    def fixture(self):
        return dict(objects=[dict(instance_id='s', label='person', bbox=[0, 0, .4, 1], confidence=.9),
                             dict(instance_id='o', label='bicycle', bbox=[.5, .2, 1, 1], confidence=.8)],
                    relations=[dict(subject_instance='s', object_instance='o', subject='person',
                                    object='bicycle', relation='riding', score=99)])

    def test_normalized_boxes_scores_and_memory_path(self):
        items = prepare_selected(self.fixture())
        embeddings = {items[0][key]: torch.randn(512) for key in ('positive', 'negative')}
        entry = pack_selected(items, embeddings)
        self.assertTrue(torch.allclose(entry['object_boxes'][0], torch.tensor([.5, .2, 1, 1])))
        self.assertAlmostEqual(entry['confidences'][0].item(), .8)
        align_grounded_relation_cache({'x.jpg': entry}, ['/images/x.jpg'])
        torch.manual_seed(42)
        module = PairGroundedRelationAlignment(8, 2, dropout=0).eval()
        queries, visual = torch.randn(1, 3, 8), torch.randn(1, 197, 8)
        def run(mask, boxes):
            return module(queries, visual, entry['tokens'].float()[None],
                entry['negative_tokens'].float()[None], entry['subject_boxes'][None],
                boxes[None], entry['confidences'][None], mask[None])[0]
        enabled = run(entry['mask'], entry['object_boxes'])
        disabled = run(torch.zeros_like(entry['mask']), entry['object_boxes'])
        swapped = entry['object_boxes'].clone(); swapped[0] = torch.tensor([0, .1, .2, .3])
        self.assertTrue(torch.equal(disabled, queries))
        self.assertFalse(torch.equal(enabled, disabled))
        self.assertFalse(torch.equal(enabled, run(entry['mask'], swapped)))

    def test_invalid_identity_and_pixel_boxes_rejected(self):
        entry = self.fixture(); entry['objects'][1]['bbox'] = [10, 20, 100, 200]
        with self.assertRaises(ValueError): prepare_selected(entry)

    def test_target_intervention_preserves_text_and_original(self):
        selected = self.fixture()
        selected['objects'].append(dict(instance_id='b', label='bicycle',
                                        bbox=[0, .2, .3, 1], confidence=.7))
        items = prepare_selected(selected)
        embeddings = {items[0][key]: torch.randn(512) for key in ('positive', 'negative')}
        original = pack_selected(items, embeddings)
        snapshot = original['object_boxes'].clone()
        swapped, metadata = alternate_target(selected, original)
        self.assertEqual(metadata['alternate_target'], 'b')
        self.assertTrue(torch.equal(original['tokens'], swapped['tokens']))
        self.assertTrue(torch.equal(original['mask'], swapped['mask']))
        self.assertTrue(torch.equal(original['object_boxes'], snapshot))
        self.assertFalse(torch.equal(original['object_boxes'], swapped['object_boxes']))
        entry = self.fixture(); entry['relations'][0]['object'] = 'horse'
        with self.assertRaises(ValueError): prepare_selected(entry)


if __name__ == '__main__': unittest.main()
