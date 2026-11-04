"""CPU metadata/sampling fixtures only, never imports Torch or executes a model."""
from collections import Counter
import unittest
import numpy as np
from grounding_labels import cohort
from grounding_pilot_data import round_robin_schedule


class Checks(unittest.TestCase):
    def test_30k_extends_same_10k_and_excludes_validation_test(self):
        rows=[dict(cocoid=i,filename=str(i),split='train' if i%2 else 'restval') for i in range(31000)]
        rows += [dict(cocoid=40000+i,filename='held'+str(i),split=s) for i,s in enumerate(('val','test'))]
        small=cohort(rows,10000);large=cohort(rows,30000)
        self.assertEqual(large[:10000],small);self.assertEqual(large,cohort(rows[::-1],30000))
        self.assertEqual({r['split'] for r in large},{'train','restval'})

    def test_64k_events_cover_all_30k_before_repeating(self):
        groups=[list(range(i*5,(i+1)*5)) for i in range(30000)]
        order=round_robin_schedule(groups);flat=order.ravel();images=flat//5
        self.assertEqual(order.shape,(2000,32));self.assertEqual(len(set(images[:30000])),30000)
        self.assertEqual(len(set(images[30000:60000])),30000);self.assertEqual(len(set(flat)),64000)
        counts=Counter(images.tolist());self.assertEqual(Counter(counts.values()),{2:26000,3:4000})
        np.testing.assert_array_equal(order,round_robin_schedule(groups))

    def test_variable_reference_counts(self):
        groups=[[0,1],[2,3,4],[5,6,7,8]];flat=round_robin_schedule(groups).ravel()
        image={event:i for i,events in enumerate(groups) for event in events}
        for begin in (0,3,6):self.assertEqual({image[int(e)] for e in flat[begin:begin+3]},{0,1,2})
        for i,events in enumerate(groups):
            first=[int(e) for e in flat if image[int(e)]==i][:len(events)]
            self.assertEqual(set(first),set(events))

    def test_invalid_schedule_rejected(self):
        for groups in ([],[[0],[]]):
            with self.assertRaises(ValueError):round_robin_schedule(groups)


if __name__=='__main__':unittest.main(verbosity=2)
