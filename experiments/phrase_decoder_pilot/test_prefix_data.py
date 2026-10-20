"""Pure checks only. Importing the diagnostic never starts inference."""
import unittest
from run_prefix_diagnostic import common_branch


class Checks(unittest.TestCase):
    def test_shared_prefix_has_no_answer_leak(self):
        prefix,branches=common_branch([1,4,5,6,7],[1,4,5,8,9])
        self.assertEqual(prefix,[1,4,5]);self.assertEqual(branches,[6,8])

    def test_no_lexical_contrast_rejected(self):
        for left,right in [([1,4],[1,4]),([1,4],[1,4,5]),([1],[2])]:
            with self.assertRaises(AssertionError):common_branch(left,right)

    def test_locked_same_image_controls(self):
        import json
        from pathlib import Path
        m=json.loads((Path(__file__).parent/'locked_manifest.json').read_text(encoding='utf-8'))
        selected=[g for g in m['groups'] if g['same_image']]
        self.assertEqual(len(selected),4)
        for split,ids in m['partitions'].items():self.assertEqual(sum(g['group_id'] in ids for g in selected),2)
        for g in selected:
            self.assertEqual(g['events'][0]['image_id'],g['events'][1]['image_id'])
            prefix,branches=common_branch(*[s.split() for s in g['descriptions']])
            self.assertEqual(prefix,['a','person'])
            self.assertIn(set(branches),[{'standing','carrying'},{'holding','looking'}])


if __name__=='__main__':unittest.main(verbosity=2)
