"""Metadata tests for the frozen panel; never download images or run models."""
import unittest
from collections import Counter
from make_panel import assemble

class Checks(unittest.TestCase):
    def test_locked_membership_and_blinding(self):
        public,key,downloads=assemble()
        self.assertEqual(len(public['cards']),64)
        self.assertEqual(len(downloads),64)
        self.assertEqual(len(key),175)
        self.assertEqual(sum(len(k['arms']) for k in key),192)
        counts=Counter((k['pair_id'],k['role'],a) for k in key for a in k['arms'])
        self.assertEqual(len(counts),192)
        self.assertTrue(all(n==1 for n in counts.values()))
        self.assertEqual(Counter({pid:next(k['stratum'] for k in key if k['pair_id']==pid) for pid in {k['pair_id'] for k in key}}.values()),{'projective_containment':16,'contact_proximity':16})
        for card in public['cards']:
            self.assertFalse({'arms','role','pair_id','stratum','coco_id'}&card.keys())
            self.assertEqual(len({u['caption'] for u in card['captions']}),len(card['captions']))
            for box in card['boxes'].values():
                self.assertTrue(0<=box[0]<box[2]<=card['width'])
                self.assertTrue(0<=box[1]<box[3]<=card['height'])
        self.assertEqual(public,assemble()[0])

    def test_assistant_proposals_keep_human_provenance_separate(self):
        import json
        from make_panel import HERE
        public,_,_=assemble()
        draft=json.loads((HERE/'ASSISTANT_PANEL_REVIEW.json').read_text(encoding='utf8'))
        ids={u['id'] for c in public['cards'] for u in c['captions']}
        self.assertEqual(set(draft['annotations']),ids)
        self.assertEqual(draft['panel_sha256'],public['panel_sha256'])
        self.assertEqual(draft['reviewer_type'],'assistant_proposed_not_human')
        self.assertTrue(all(a['confirmed_by_human'] is False for a in draft['annotations'].values()))
        self.assertTrue(all(a['notes'] for a in draft['annotations'].values()))

if __name__=='__main__':unittest.main()
