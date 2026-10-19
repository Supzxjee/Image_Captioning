import unittest
import numpy as np
from pilot_data import box, pool, geometry, region_slots, cohorts, identity, evidence_bounds, phrase_identity


class DataChecks(unittest.TestCase):
    def test_spatial_pool_and_tiny_box(self):
        patches = np.repeat(np.arange(196, dtype=np.float32)[:, None], 768, axis=1)
        np.testing.assert_allclose(pool(patches, [0, 0, 1, 1]), 97.5)
        np.testing.assert_allclose(pool(patches, [0, 0, .001, .001]), 0)
        np.testing.assert_allclose(geometry(box([0, 0, 1, 1]), box([0, 0, 1, 1])), [0,0,0,0,1,1,1,0])
        for b in ([0,0,0,1], [-1,0,1,1], [0,0,float('nan'),1]):
            with self.assertRaises(ValueError): box(b)

    def test_instance_link_fallback_and_no_objects(self):
        objects = [dict(index=0, instance_id='p0', label='person', confidence=.9, bbox=[0,0,.4,1]),
                   dict(index=1, instance_id='o1', label='horse', confidence=.8, bbox=[.4,0,1,1])]
        relation = dict(subject_index=0, object_index=1, subject_instance='p0', object_instance='o1',
                        score=.7, threshold=.5, text='person riding horse')
        entry = dict(objects=objects, relations=[relation], mode='object_relation')
        visual = np.ones((197, 768), dtype=np.float32)
        f, roles, valid, texts = region_slots(entry, visual)
        self.assertEqual(valid.sum(), 3)
        self.assertEqual(texts[:3], ['person','horse','person riding horse'])
        bounds=evidence_bounds(entry)
        np.testing.assert_array_equal(bounds[0,2],bounds[0,0])
        np.testing.assert_array_equal(bounds[1,2],bounds[1,1])
        np.testing.assert_array_equal(bounds[2,2],[0,0,1,1])
        self.assertEqual(phrase_identity('Person'),phrase_identity(' person '))
        self.assertNotEqual(phrase_identity('horse'),phrase_identity('person'))
        np.testing.assert_array_equal(roles[:3], [0,1,2])
        np.testing.assert_array_equal(f[2,:2304], 1)
        entry['relations'] = []; entry['mode'] = 'object_only'
        self.assertEqual(region_slots(entry, visual)[2].sum(), 2)
        entry['objects'] = []; entry['mode'] = 'no_objects'
        self.assertFalse(region_slots(entry, visual)[2].any())
        entry.update(objects=objects, relations=[dict(relation, subject_instance='wrong')], mode='object_relation')
        with self.assertRaises(ValueError): region_slots(entry, visual)

    def test_locked_split_is_deterministic_and_excludes_test(self):
        data = {'images': [dict(filename=f'COCO_train2014_{i:012d}.jpg', split=split)
                           for i, split in enumerate(['train']*6+['restval']*2+['val']*5+['test']*3)]}
        a = cohorts(data, 4, 3); b = cohorts(data, 4, 3)
        self.assertEqual(identity(a), identity(b))
        self.assertTrue(all(x['image_id'] < 8 for x in a['train']))
        self.assertTrue(all(8 <= x['image_id'] < 13 for x in a['val']))
        self.assertFalse({r['coco_id'] for r in a['train']} & {r['coco_id'] for r in a['val']})


if __name__ == '__main__': unittest.main(verbosity=2)
