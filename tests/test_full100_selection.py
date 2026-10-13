import unittest
from select_actor_full100_images import select_images


class SelectionTests(unittest.TestCase):
    def test_excluded_and_caption_heldout_never_selected(self):
        rows=[{'cocoid':i,'filename':str(i)+'.jpg','split':'train' if i<8 else 'test'} for i in range(10)]
        chosen,pool=select_images(rows,range(10),{0,1},4,42)
        self.assertEqual(pool,list(range(2,8)))
        self.assertEqual(len({r['image_id'] for r in chosen}),4)
        self.assertEqual(chosen,select_images(list(reversed(rows)),range(10),{0,1},4,42)[0])

    def test_insufficient_pool_fails_without_replacement(self):
        with self.assertRaises(ValueError):select_images([],[],[],100,42)


if __name__=='__main__':unittest.main()
