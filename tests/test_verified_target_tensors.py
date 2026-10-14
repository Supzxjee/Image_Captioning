import unittest
from run_interaction_region_cached import verified_target_tensors


class VerifiedTests(unittest.TestCase):
    def test_only_reviewed_unknown_changes_original_stays_intact(self):
        rows=[{'image_id':1,'subject_id':2,'object_id':3,'labels':[1,-1,0,0,0,0]},
              {'image_id':1,'subject_id':2,'object_id':4,'labels':[-1,-1,0,0,0,0]}]
        filtered={'eligible_contrasts':[{'positive_row_index':0,'alternate_row_index':1,'relation_index':0}]}
        labels,edges=verified_target_tensors(rows,filtered)
        self.assertEqual(int(labels[1,0]),0);self.assertEqual(int(labels[1,1]),-1)
        self.assertEqual(rows[1]['labels'][0],-1);self.assertEqual(edges.tolist(),[[0,0,1,0]])
        rows[1]['subject_id']=5
        with self.assertRaises(ValueError):verified_target_tensors(rows,filtered)

    def test_known_labels_and_duplicate_edges_rejected(self):
        rows=[{'image_id':1,'subject_id':2,'object_id':3,'labels':[1]*6},
              {'image_id':1,'subject_id':2,'object_id':4,'labels':[0]*6}]
        e={'positive_row_index':0,'alternate_row_index':1,'relation_index':0}
        with self.assertRaises(ValueError):verified_target_tensors(rows,{'eligible_contrasts':[e]})
        rows[1]['labels']=[-1]*6
        with self.assertRaises(ValueError):verified_target_tensors(rows,{'eligible_contrasts':[e,e]})


if __name__=='__main__':unittest.main()
