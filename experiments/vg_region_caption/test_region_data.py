import unittest
from region_data import patch_mask, select, schedule, vocabulary


def row(cid=1, phrase='a person', box=None):
    return dict(coco_id=cid, source_split='train', region_id=cid, phrase=phrase,
                region_normalized=box or [0, 0, 1, 1])


class Checks(unittest.TestCase):
    def test_patch_centers_square_resize_no_cls_no_fallback(self):
        self.assertEqual(sum(patch_mask([0, 0, 1, 1])), 196)
        self.assertEqual(sum(patch_mask([0, 0, .5, 1])), 98)
        self.assertEqual(sum(patch_mask([0, 0, .001, .001])), 0)
        for bad in ([0, 0, 2, 1], [0, 0, float('nan'), 1], [1, 0, 0, 1]):
            with self.assertRaises(ValueError): patch_mask(bad)

    def test_no_oov_no_truncation_exclusions_and_eos(self):
        vocab = dict(vocabulary(dict(images=[dict(split='train', sentences=[dict(raw='a person')]),
                                               dict(split='test', sentences=[dict(raw='leak')])])))
        self.assertNotIn('leak', vocab)
        rows = [row(), row(2), row(3, 'an unknown'), row(4, box=[0,0,.001,.001]),
                row(5, ' '.join(['a']*29)), row(6)]
        bank, report = select(rows, vocab, {1,2,3,4,5}, {2})
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0]['encoded'][-1], vocab['<eos>'])
        self.assertEqual(report['rejects']['any_oov_token'], 1)

    def test_balanced_schedule_and_cap_repeatable(self):
        rows = [dict(row(cid, box=[0,0,1-j*.01,1]), region_id=100*cid+j)
                for cid in range(1,4) for j in range(9)]
        bank, _ = select(rows, {'<bos>':1,'<eos>':2,'a':4,'person':5}, {1,2,3}, set())
        self.assertEqual(len(bank), 15)
        s = schedule(bank, 10000)
        self.assertEqual(s, schedule(bank, 10000))
        self.assertEqual(len({bank[i]['coco_id'] for i in s[:3]}), 3)
        self.assertEqual(len(s), 10000)


if __name__ == '__main__': unittest.main(verbosity=2)
