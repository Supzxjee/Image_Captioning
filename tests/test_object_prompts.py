import unittest
from captioning.object_prompts import build_object_texts, object_names


class ObjectPromptTests(unittest.TestCase):
    def test_uses_only_saved_labels_not_relations_attributes_or_captions(self):
        source = {'/old/path/x.jpg': {'objects': [{'name': 'Person', 'attributes': ['red']},
                      {'name': 'bicycle'}, {'name': 'person'}],
                     'prompt': 'person riding bicycle', 'caption': 'never read this'}}
        self.assertEqual(build_object_texts(source, [{'filename': 'x.jpg'}]),
                         {'x.jpg': 'person, bicycle'})

    def test_missing_objects_cannot_be_recovered_from_embeddings_or_full_prompt(self):
        for entry in ({'tokens': [1], 'mask': [1]}, {'prompt': 'person next to bicycle'}):
            with self.assertRaisesRegex(ValueError, 'Expected an object list'):
                object_names(entry)

    def test_alternate_detection_field_and_label_key(self):
        source = {'x.jpg': {'detections': [{'category': 'traffic light', 'confidence': .8}]}}
        self.assertEqual(build_object_texts(source, [{'filename': 'x.jpg'}], 'detections', 'category'),
                         {'x.jpg': 'traffic light'})

    def test_rejects_missing_images_ambiguous_names_and_numeric_classes(self):
        for source in ({}, {'a/x.jpg': [], 'b/x.jpg': []}, {'x.jpg': [{'name': 1}]}):
            with self.assertRaises(ValueError):
                build_object_texts(source, [{'filename': 'x.jpg'}])

    def test_empty_detection_kept_empty_without_fabricated_object(self):
        self.assertEqual(object_names([]), [])
        self.assertEqual(build_object_texts({'x.jpg': []}, [{'filename': 'x.jpg'}]), {'x.jpg': ''})


if __name__ == '__main__':
    unittest.main()
