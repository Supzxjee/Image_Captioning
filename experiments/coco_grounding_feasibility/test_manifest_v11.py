import unittest
import re
from manifest_policy import POLICY as OLD
from manifest_policy_v11 import POLICY,context_exclusion


def check(text,start,end,label):
    raw=text.lower().split();norm=[re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',w) for w in raw]
    return context_exclusion(norm,start,end,label,raw_tokens=raw)


class Checks(unittest.TestCase):
    def test_explicit_plural_rules_and_paired_units(self):
        self.assertIn('train',POLICY['ambiguity_exclusions']);self.assertIn('trains',POLICY['ambiguity_exclusions'])
        self.assertEqual(check('people on a street',0,1,'person'),'plural_surface_single_annotation')
        self.assertEqual(check('a person with two cats',4,5,'cat'),'plural_surface_single_annotation')
        self.assertIsNone(check('two skis',1,2,'skis'))
        self.assertEqual(check('several pairs of skis',3,4,'skis'),'explicit_multiple_units')
        self.assertEqual(check('three skis',1,2,'skis'),'explicit_multiple_units')
        self.assertEqual(check('two scissors',1,2,'scissors'),'explicit_multiple_units')

    def test_color_scope_punctuation_and_hyphens(self):
        for text in ('a blue suitcase','a blue-colored suitcase','a dark-blue suitcase','a black-and-white suitcase'):
            self.assertEqual(check(text,2,3,'suitcase'),'local_color_modifier')
        self.assertIsNone(check('a red bicycle and a suitcase',5,6,'suitcase'))
        self.assertIsNone(check('a red bicycle beside a suitcase',5,6,'suitcase'))
        self.assertIsNone(check('a blue, suitcase',2,3,'suitcase'))
        self.assertIsNone(check('a red shirt holding a suitcase',5,6,'suitcase'))

    def test_gates_and_source_scope_not_relaxed(self):
        for key in ('minimum_images','minimum_unique_spans','minimum_classes_with_50_spans',
                    'minimum_patch_area','maximum_image_fraction','person_maximum_fraction','source_scope'):
            self.assertEqual(POLICY[key],OLD[key])


if __name__=='__main__':unittest.main(verbosity=2)
