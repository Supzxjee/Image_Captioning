import unittest
from audit_prompt_target_information import serialize_case, grounded_payload


class PromptTargetInformationTest(unittest.TestCase):
    def test_same_class_swap_loses_text_identity_but_keeps_boxes(self):
        case = dict(filename='example.jpg', subject_box=[0, 0, .3, .9], relation='riding',
                    targets=[dict(label='bicycle', bbox=[0, .3, .4, 1]),
                             dict(label='bicycle', bbox=[.6, .3, 1, 1])])
        a, b = serialize_case(case, 0), serialize_case(case, 1)
        self.assertEqual(a['prompt'], b['prompt'])
        self.assertNotEqual(grounded_payload(a), grounded_payload(b))
        self.assertEqual(grounded_payload(a)[0]['subject_box'], case['subject_box'])
        self.assertEqual(grounded_payload(b)[0]['object_box'], case['targets'][1]['bbox'])

    def test_different_class_is_visible_in_text(self):
        case = dict(filename='example.jpg', subject_box=[0, 0, .3, .9], relation='holding',
                    targets=[dict(label='cup', bbox=[0, .3, .4, 1]),
                             dict(label='tie', bbox=[.6, .3, 1, 1])])
        self.assertNotEqual(serialize_case(case, 0)['prompt'], serialize_case(case, 1)['prompt'])


if __name__ == '__main__':
    unittest.main()
