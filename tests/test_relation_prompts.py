import unittest

from captioning.relation_prompts import build_relation_entry, infer_relations, select_detections
from build_relation_prompts import select_source_filenames
from embed_relation_prompts import index_prompts


def detection(label, box, conf=0.9):
    return {'label': label, 'bbox': box, 'conf': conf}


class RelationPromptTests(unittest.TestCase):
    def test_person_object_containment_becomes_action_not_inside(self):
        entry = {'objects': [
            detection('person', [0, 0, 100, 200]),
            detection('cell phone', [55, 70, 70, 100]),
        ]}
        result = build_relation_entry(entry)
        self.assertEqual(result['triplets'][0]['relation'], 'holding')
        self.assertNotIn('contains', result['prompt'])
        self.assertNotIn('inside person', result['prompt'])

    def test_surface_containment_becomes_on(self):
        entry = {'objects': [
            detection('dining table', [0, 80, 200, 200]),
            detection('cake', [60, 55, 110, 105]),
        ]}
        result = build_relation_entry(entry)
        self.assertEqual(result['triplets'][0]['relation'], 'on')
        self.assertEqual(result['prompt'], 'a photo showing cake on dining table')

    def test_implausible_furniture_on_support_is_rejected(self):
        entry = {'objects': [
            detection('dining table', [0, 80, 200, 200]),
            detection('chair', [60, 55, 110, 105]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('chair on dining table', result['prompt'])

    def test_potted_plant_is_not_put_on_couch_from_box_overlap(self):
        entry = {'objects': [
            detection('couch', [0, 80, 200, 200]),
            detection('potted plant', [60, 55, 110, 105]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('potted plant on couch', result['prompt'])

    def test_implausible_animal_containment_is_not_inside(self):
        entry = {'objects': [
            detection('dog', [0, 0, 160, 160]),
            detection('frisbee', [20, 50, 80, 100]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotEqual(result['triplets'][0]['relation'], 'inside')

    def test_true_vehicle_container_is_allowed(self):
        entry = {'objects': [
            detection('car', [0, 0, 200, 150]),
            detection('dog', [60, 30, 120, 100]),
        ]}
        result = build_relation_entry(entry)
        self.assertEqual(result['triplets'][0]['relation'], 'inside')

    def test_tiny_passenger_box_does_not_imply_vehicle_containment(self):
        entry = {'objects': [
            detection('airplane', [0, 0, 500, 300]),
            detection('person', [240, 220, 260, 280]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('person inside airplane', result['prompt'])

    def test_carried_vertical_surfboard_is_not_ridden(self):
        entry = {'objects': [
            detection('person', [80, 20, 140, 200]),
            detection('surfboard', [120, 40, 170, 230]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('person riding surfboard', result['prompt'])

    def test_torso_scissors_do_not_imply_holding(self):
        entry = {'objects': [
            detection('person', [0, 0, 100, 200]),
            detection('scissors', [45, 35, 55, 60]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('person holding scissors', result['prompt'])

    def test_arbitrary_overlap_is_not_emitted(self):
        entry = {'objects': [
            detection('car', [0, 0, 200, 120]),
            detection('person', [80, 20, 140, 160]),
        ]}
        result = build_relation_entry(entry)
        self.assertNotIn('overlapping', result['prompt'])

    def test_multiple_same_class_instances_do_not_create_text_contradiction(self):
        entry = {'objects': [
            detection('person', [90, 20, 140, 160]),
            detection('surfboard', [0, 100, 80, 150], 0.95),
            detection('surfboard', [150, 100, 240, 150], 0.90),
        ]}
        result = build_relation_entry(entry)
        relations = [item for item in result['triplets']
                     if {item['subject'], item['object']} == {'person', 'surfboard'}]
        self.assertEqual(len(relations), 1)
        self.assertIn('#', relations[0]['subject_id'])
        self.assertIn('#', relations[0]['object_id'])

    def test_same_label_nms_preserves_distinct_instances(self):
        detections = select_detections({'objects': [
            detection('dog', [0, 0, 100, 100], 0.9),
            detection('dog', [2, 2, 101, 101], 0.8),
            detection('dog', [200, 0, 300, 100], 0.7),
        ]})
        self.assertEqual([item.instance_id for item in detections], ['dog#1', 'dog#2'])

    def test_only_one_canonical_relation_per_instance_pair(self):
        detections = select_detections({'objects': [
            detection('person', [0, 0, 100, 200]),
            detection('snowboard', [10, 170, 130, 220]),
        ]})
        relations = infer_relations(detections)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]['relation'], 'riding')

    def test_prompt_index_uses_portable_filenames_and_rejects_collisions(self):
        indexed = index_prompts({'/old/path/a.jpg': {'prompt': 'person riding bicycle'}})
        self.assertEqual(indexed['a.jpg']['prompt'], 'person riding bicycle')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            index_prompts({
                '/first/a.jpg': {'prompt': 'first'},
                '/second/a.jpg': {'prompt': 'second'},
            })

    def test_exact_filename_subset_preserves_requested_order(self):
        source = {
            '/old/b.jpg': {'objects': []},
            '/old/a.jpg': {'objects': []},
        }
        selected = select_source_filenames(source, ['a.jpg', 'b.jpg'])
        self.assertEqual(list(selected), ['/old/a.jpg', '/old/b.jpg'])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            select_source_filenames(source, ['missing.jpg'])


if __name__ == '__main__':
    unittest.main()
