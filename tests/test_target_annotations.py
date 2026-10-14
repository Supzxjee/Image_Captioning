import unittest
from import_target_annotations import filter_annotations


class AnnotationTests(unittest.TestCase):
    def test_blank_and_unknown_never_become_negative(self):
        mapping={'manifest_sha256':'test','cases':[{'case_id':'c','positive_side':'B'}]}
        review={'schema_version':1,'manifest_sha256':'test','reviewer_type':'human_self_reported','reviewer':'r','annotations':{}}
        self.assertEqual(filter_annotations(review,mapping)['counts'],{'incomplete':1})
        review['annotations']['c']={'subject_box_status':'clear','A_box_status':'clear','B_box_status':'clear','A_status':'unclear','B_status':'supported'}
        self.assertEqual(filter_annotations(review,mapping)['counts'],{'alternative_unknown':1})
        self.assertEqual(filter_annotations(review,mapping)['eligible_contrasts'],[])

    def test_confirmed_pair_requires_evidence_and_good_boxes(self):
        mapping={'manifest_sha256':'test','cases':[{'case_id':'c','positive_side':'B','positive_row_index':5,'alternate_row_index':9,'relation_index':0}]}
        a={'subject_box_status':'clear','A_box_status':'clear','B_box_status':'clear','A_status':'contradicted','B_status':'supported','notes':'Hands visibly away from A.'}
        review={'schema_version':1,'manifest_sha256':'test','reviewer_type':'human_self_reported','reviewer':'r','annotations':{'c':a}}
        result=filter_annotations(review,mapping);self.assertEqual(result['counts'],{'eligible':1})
        self.assertEqual(result['eligible_contrasts'][0]['alternate_row_index'],9)
        a['A_box_status']='unclear';self.assertEqual(filter_annotations(review,mapping)['eligible_contrasts'],[])
        a['A_box_status']='clear';a['B_status']='contradicted'
        self.assertEqual(filter_annotations(review,mapping)['counts'],{'source_positive_conflict':1})

    def test_assistant_or_wrong_manifest_rejected(self):
        mapping={'manifest_sha256':'test','cases':[]}
        review={'schema_version':1,'manifest_sha256':'test','reviewer_type':'assistant','reviewer':'a'}
        with self.assertRaises(ValueError):filter_annotations(review,mapping)
        review['reviewer_type']='human_self_reported';review['manifest_sha256']='other'
        with self.assertRaises(ValueError):filter_annotations(review,mapping)


if __name__=='__main__':unittest.main()
