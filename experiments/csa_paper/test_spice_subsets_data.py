"""Pure data checks only; no official scorer/model imports or execution."""
import unittest
from score_spice_subsets import clean, summarize


class Checks(unittest.TestCase):
    def test_undefined_is_not_zero(self):
        rows = [dict(scores=clean({'All': {'f': .5}, 'Relations': {'f': float('nan')}})),
                dict(scores=clean({'All': {'f': .3}, 'Relations': {'f': .2}}))]
        s = summarize(rows)
        self.assertEqual(s['All']['f_x100'], 40)
        self.assertEqual(s['Relations']['f_x100'], 20)
        self.assertEqual(s['Relations']['finite_images'], 1)
        self.assertEqual(s['Objects']['f_x100'], None)
        self.assertEqual(rows[0]['scores']['Relations']['f'], None)


if __name__ == '__main__': unittest.main(verbosity=2)
