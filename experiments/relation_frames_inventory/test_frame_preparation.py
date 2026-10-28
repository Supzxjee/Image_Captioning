"""CPU array/identity checks only; never import torch or execute preparation."""
import unittest
import numpy as np
from prepare_frames_cpu import records,valid_prompt


class Checks(unittest.TestCase):
    def test_invalid_cached_values_fail(self):
        tokens=np.zeros((20,512),dtype='f4');mask=np.ones(20,dtype='u1')
        self.assertTrue(valid_prompt(tokens,mask))
        self.assertFalse(valid_prompt(tokens,np.zeros(20,dtype='u1')))
        self.assertFalse(valid_prompt(tokens,np.full(20,2,dtype='u1')))
        tokens[0,0]=float('nan');self.assertFalse(valid_prompt(tokens,mask))

    def test_deduplicate_images_not_assertions(self):
        row=dict(image_id=1,image=dict(file_name='a.jpg',width=640,height=480))
        pair=dict(positive=row,negative=dict(image_id=2,image=dict(file_name='b.jpg',width=320,height=240)))
        self.assertEqual(len(records(dict(train=[pair,pair],heldout=[]))),2)

    def test_conflicting_image_identity_fails(self):
        a=dict(image_id=1,image=dict(file_name='a.jpg',width=640,height=480))
        b=dict(image_id=1,image=dict(file_name='b.jpg',width=640,height=480))
        with self.assertRaises(ValueError):records(dict(train=[dict(positive=a,negative=b)],heldout=[]))


if __name__=='__main__':unittest.main(verbosity=2)
