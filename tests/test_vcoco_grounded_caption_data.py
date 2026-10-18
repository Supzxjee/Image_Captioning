import copy
import unittest

from prepare_vcoco_grounded_caption_data import grounded_images, prepare, relation_statistics


def fixture():
    # Three riders; two explicit target links and one missing role object.
    actions = [
        {'action_name': 'ride', 'role_name': ['agent', 'instr'],
         'ann_id': [1, 2, 3], 'image_id': [100] * 3, 'label': [1, 1, 1],
         'role_object_id': [1, 2, 3, 10, 11, 0]},
        {'action_name': 'hold', 'role_name': ['agent', 'obj'],
         'ann_id': [1, 2, 3], 'image_id': [100] * 3, 'label': [1, 0, -1],
         'role_object_id': [1, 2, 3, 10, 0, 0]},
    ]
    instances = {
        'images': [{'id': 100, 'file_name': 'COCO_train2014_000000000100.jpg', 'width': 640, 'height': 480}],
        'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'horse'}],
        'annotations': [{'id': aid, 'image_id': 100, 'category_id': 1 if aid < 10 else 2,
                         'bbox': [aid * 2, 20, 30, 50], 'iscrowd': 0} for aid in (1, 2, 3, 10, 11)],
    }
    captions = {100: {'split': 'train', 'filename': 'COCO_train2014_000000000100.jpg', 'filepath': 'train2014'}}
    return actions, instances, captions


class GroundedCaptionDataTests(unittest.TestCase):
    def test_all_links_multiactor_and_multilabel_retained(self):
        rows, stats = grounded_images(*fixture(), {100})
        self.assertEqual(len(rows[0]['relations']), 3)
        self.assertEqual(stats['relations']['riding'], {'links': 2, 'images': 1})
        self.assertEqual(stats['images_with_multiple_actors_same_predicate_object_class'], 1)
        self.assertEqual(stats['multi_actor_image_counts_by_predicate_object_class'], {'riding:horse': 1})
        self.assertEqual(len(rows[0]['instances']), 5)
        self.assertFalse(stats['maximum_relation_cap_applied'])

    def test_missing_role_keeps_action_without_inventing_link(self):
        rows, stats = grounded_images(*fixture(), {100})
        self.assertEqual(stats['positive_actions_missing_role'], {'riding': 1})
        self.assertIn({'subject_id': 3, 'predicate': 'riding', 'label': 1}, rows[0]['actor_action_labels'])
        self.assertFalse(any(r['subject_id'] == 3 for r in rows[0]['relations']))
        self.assertIn({'subject_id': 3, 'predicate': 'holding', 'label': -1}, rows[0]['actor_action_labels'])

    def test_crowd_and_invalid_target_skip_edge_keep_actor(self):
        actions, instances, captions = fixture()
        instances['annotations'][3]['iscrowd'] = 1
        instances['annotations'][4]['bbox'][0] = float('nan')
        rows, stats = grounded_images(actions, instances, captions, {100})
        self.assertEqual(rows[0]['relations'], [])
        self.assertEqual(len(rows[0]['actor_action_labels']), 6)
        self.assertEqual(stats['unusable_endpoints']['riding:object:crowd_box'], 1)
        self.assertEqual(stats['unusable_endpoints']['riding:object:invalid_box'], 1)

    def test_missing_target_annotation_is_reported(self):
        actions, instances, captions = fixture()
        instances['annotations'] = instances['annotations'][:-1]
        _, stats = grounded_images(actions, instances, captions, {100})
        self.assertEqual(stats['unusable_endpoints']['riding:object:missing_annotation'], 1)

    def test_wrong_image_link_and_nonperson_actor_rejected(self):
        actions, instances, captions = fixture()
        instances['images'].append({'id': 101, 'file_name': 'other.jpg', 'width': 640, 'height': 480})
        instances['annotations'][3]['image_id'] = 101
        with self.assertRaises(ValueError):
            grounded_images(actions, instances, captions, {100})
        actions, instances, captions = fixture()
        instances['annotations'][0]['category_id'] = 2
        with self.assertRaises(ValueError):
            grounded_images(actions, instances, captions, {100})

    def test_caption_heldout_and_source_mismatch_rejected(self):
        actions, instances, captions = fixture()
        captions[100]['split'] = 'val'
        with self.assertRaises(ValueError):
            grounded_images(actions, instances, captions, {100})
        with self.assertRaises(ValueError):
            grounded_images(*fixture(), {101})

    def test_split_overlap_rejected(self):
        actions, instances, captions = fixture()
        with self.assertRaises(ValueError):
            prepare({'train': actions, 'val': actions}, instances, captions,
                    {'relation_fit': [100], 'relation_calibration': [], 'relation_dev': [100]})

    def test_missing_role_schema_and_duplicate_ids_rejected(self):
        actions, instances, captions = fixture()
        actions[0]['role_name'][1] = 'wrong'
        with self.assertRaises(ValueError):
            grounded_images(actions, instances, captions, {100})
        actions, instances, captions = fixture()
        instances['annotations'].append(copy.deepcopy(instances['annotations'][0]))
        with self.assertRaises(ValueError):
            grounded_images(actions, instances, captions, {100})

    def test_multiple_predicates_one_actor_do_not_prove_multiple_people(self):
        rows, _ = grounded_images(*fixture(), {100})
        rows[0]['relations'] = [r for r in rows[0]['relations'] if r['subject_id'] == 1]
        stats = relation_statistics(rows)
        self.assertEqual(stats['images_with_multiple_links'], 1)
        self.assertEqual(stats['images_with_multiple_actors_same_predicate_object_class'], 0)


if __name__ == '__main__':
    unittest.main()
