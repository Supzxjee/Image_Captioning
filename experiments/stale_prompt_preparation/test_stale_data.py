"""CPU synthetic invariants, no torch/model imports."""
import unittest
import numpy as np
from stale_data import *


class Checks(unittest.TestCase):
    def test_longest_phrase_prevents_wrong_dog_replacement(self):
        aliases=[(('dog',),'dog'),(('dogs',),'dog'),(('hot','dog'),'hot dog')]
        self.assertEqual(neutralize('A hot dog beside a dog and dogs.', 'dog',aliases),'A hot dog beside an object and object.')
        self.assertIsNone(neutralize('a hot dog','dog',aliases))

    def test_word_boundaries_and_all_mentions(self):
        aliases=[(('cat',),'cat'),(('cats',),'cat')]
        self.assertEqual(neutralize('Cat beside cats; cathedral.','cat',aliases),'object beside object; cathedral.')

    def test_ambiguous_and_multiword_aliases_not_class_probabilities(self):
        vocab={'cat':4,'cat.':5,'bat':6,'hot dog':7,'dog':8}
        aliases=[(('cat',),'cat'),(('bat',),'baseball bat'),(('bat',),'bird'),(('hot','dog'),'hot dog'),(('dog',),'dog')]
        self.assertEqual(lexical_ids(vocab,aliases,{'dog'}),{'cat':[4,5]})

    def test_union_covers_every_instance_and_rejects_any_crowd(self):
        a=np.zeros((10,10),dtype='u1');a[:2,:2]=1;b=np.zeros_like(a);b[8:,8:]=1
        annotations=[dict(segmentation='a',iscrowd=0),dict(segmentation='b',iscrowd=0)]
        decode=lambda seg,h,w:{'a':a,'b':b}[seg]
        result=union_mask(annotations,10,10,decode)
        self.assertEqual(int(result.sum()),8)
        annotations[1]['iscrowd']=1;self.assertIsNone(union_mask(annotations,10,10,decode))

    def test_invalid_mask_never_silently_skips_instance(self):
        ann=[dict(segmentation='ok'),dict(segmentation='bad')]
        self.assertIsNone(union_mask(ann,10,10,lambda seg,h,w:np.ones((h,w))*.5))
        self.assertIsNone(union_mask([dict(segmentation='ok')],10,10,lambda *args:np.ones((10,10))))

    def test_four_condition_labels_not_swapped(self):
        s=contrast_values(-1,-2,-2,-3)
        self.assertEqual(s['factual_prompt_boost'],1)
        self.assertAlmostEqual(s['stale_hinge'],.9)
        self.assertEqual(s['visual_contrast_original'],1)
        self.assertEqual(s['prompt_visual_interaction'],0)
        different=contrast_values(-1,-3,-4,-5)
        self.assertEqual(different['prompt_visual_interaction'],1)

    def test_reference_positions_follow_whitespace_vocab_not_char_count(self):
        aliases=[(('cat',),'cat'),(('dog',),'dog')];vocab={'dog.':4,'dog':5,'cat,dog':6,'cat-dog':7}
        self.assertEqual(reference_target('A small dog.','dog',aliases,vocab,[4,5]),dict(position=2,token_id=4))
        self.assertIsNone(reference_target('cat,dog','dog',aliases,vocab,[4,5]))
        self.assertIsNone(reference_target('cat-dog','dog',aliases,vocab,[4,5]))

    def test_visible_window_does_not_reveal_truncated_tail(self):
        text='a dog beside a television'
        encoded=dict(offset_mapping=[(0,0),(0,1),(2,5),(0,0)],attention_mask=[1,1,1,1])
        self.assertEqual(visible_window(text,encoded),'a dog')


if __name__=='__main__':unittest.main(verbosity=2)
