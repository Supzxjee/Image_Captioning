import copy
import json
from pathlib import Path
import unittest
from data import choose,patch_support,proposal_match,identity
from review_result import summarize


class Checks(unittest.TestCase):
    def test_patch_counts_and_invalid_box(self):
        self.assertEqual(patch_support([0,0,1,1]),196)
        self.assertEqual(patch_support([0,0,.01,.01]),0)
        with self.assertRaises(AssertionError):patch_support([.5,0,.1,1])

    def test_selection_reproducible_no_image_reuse(self):
        def event(i):return dict(event_id=str(i),image_id=i,subject_id=i*2,object_id=i*2+1)
        pools={'x':dict(descriptions=['first','second'],sides=[[event(i) for i in range(10)],[event(i) for i in range(10,20)]])}
        groups,_=choose(pools,5);self.assertEqual(groups,choose(pools,5)[0])
        self.assertEqual(len({e['image_id'] for g in groups for e in g['events']}),10)
        self.assertTrue(all(not g['training_eligible'] and g['human_truth_matrix']==[['unreviewed']*2]*2 for g in groups))

    def test_upt_needs_two_distinct_matching_instances(self):
        e=dict(subject_box=[0,0,.5,1],object_box=[.5,0,1,1],classes=['person','bicycle'])
        entry=dict(objects=[dict(label='person',instance_id='s',bbox=e['subject_box']),dict(label='bicycle',instance_id='o',bbox=e['object_box'])])
        self.assertTrue(proposal_match(e,entry)['pair_found_at_iou05']);entry['objects'][1]['instance_id']='s'
        self.assertFalse(proposal_match(e,entry)['pair_found_at_iou05'])

    def test_review_unknown_not_false_and_ambiguous_not_contrast(self):
        m=dict(groups_sha256='fixture',groups=[dict(group_id='g',same_image=True)])
        a=dict(truth=[['unreviewed']*2]*2,boxes=[['clear']*2]*2,detail_224=['yes']*2,notes='')
        report=dict(schema_version=1,groups_sha256='fixture',reviewer='Synthetic test fixture',reviewer_type='human_self_reported',annotations={'g':a})
        self.assertEqual(summarize(m,report)['summary']['clear_semantic_contrasts'],0)
        a['truth']=[['supported']*2]*2;self.assertEqual(summarize(m,report)['summary']['clear_semantic_contrasts'],0)
        a['truth']=[['contradicted','supported'],['supported','contradicted']]
        self.assertEqual(summarize(m,report)['summary']['contrasts_with_clear_boxes_and_visible_224'],1)
        a['boxes'][0][0]='incorrect';self.assertEqual(summarize(m,report)['summary']['contrasts_with_clear_boxes'],0)
        report['groups_sha256']='wrong'
        with self.assertRaises(AssertionError):summarize(m,report)

    def test_real_manifest_and_assets_if_prepared(self):
        root=Path(__file__).parent/'review'
        if not (root/'manifest.json').exists():self.skipTest('No local prepared artifact')
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'));groups=m['groups'];self.assertEqual(len(groups),30)
        self.assertEqual(identity(groups),m['groups_sha256']);used=set()
        for g in groups:
            ids={e['image_id'] for e in g['events']};self.assertFalse(ids&used);used|=ids
            self.assertFalse(ids&set(m['excluded_known_review_image_ids']))
            self.assertEqual(g['human_truth_matrix'],[['unreviewed']*2]*2)
            for e in g['events']:
                self.assertIn(e['caption_split'],('train','restval'));self.assertTrue((root/'images'/e['filename']).is_file())
        self.assertEqual(len(used),53)


if __name__=='__main__':unittest.main(verbosity=2)
