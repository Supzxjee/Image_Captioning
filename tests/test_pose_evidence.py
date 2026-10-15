import unittest
import torch
from captioning.pose_evidence import match_subject, pair_features, subject_key
from run_pose_target_cv import PoseCorrection


def pose():
    points = [dict(x=0., y=0., visibility=0., presence=0.) for _ in range(33)]
    for k, x, y in [(11,.3,.15),(12,.6,.15),(23,.35,.5),(24,.55,.5),(25,.3,.7),(26,.6,.7),(27,.3,.9),(28,.6,.9)]:
        points[k] = dict(x=x, y=y, visibility=.9, presence=.9)
    return points


class PoseTests(unittest.TestCase):
    def test_wrong_subject_and_ambiguous_poses_are_not_assigned(self):
        p = pose(); box = [.2,.05,.7,.95]
        self.assertEqual(match_subject([p], box)['status'], 'matched')
        self.assertEqual(match_subject([p,p], box)['status'], 'ambiguous')
        self.assertNotEqual(match_subject([p], [.75,0,1,1])['status'], 'matched')

    def test_hidden_hips_and_nan_points_do_not_become_contact(self):
        p = pose(); p[23]['visibility'] = 0; p[24]['presence'] = None
        result = match_subject([p], [.2,.05,.7,.95])
        self.assertNotEqual(result['status'], 'matched')
        features, valid = pair_features(result, [.2,.05,.7,.95], [0,0,1,1])
        self.assertFalse(valid); self.assertEqual(features, [0.] * 12)

    def test_proxy_has_identical_visibility_and_only_changes_joint_positions(self):
        p = pose(); p[27]['visibility'] = p[28]['visibility'] = 0
        entry = {**match_subject([p], [.2,.05,.7,.95]), 'image_size':[640,400]}
        a, valid = pair_features(entry, [.2,.05,.7,.95], [.4,.3,.9,1])
        b, other = pair_features(entry, [.2,.05,.7,.95], [.4,.3,.9,1], proxy=True)
        self.assertTrue(valid and other); self.assertEqual(len(a), 12)
        self.assertEqual(a[7:], b[7:]); self.assertEqual(a[9], 0)
        self.assertNotEqual(a[:7], b[:7])

    def test_missing_pose_preserves_prior_and_only_two_predicates_can_change(self):
        head = PoseCorrection(); features = torch.randn(3, 12); available = torch.tensor([1.,0.,1.])
        self.assertEqual(head(features, available).abs().sum().item(), 0)
        head.linear.weight.data.fill_(1)
        correction = head(features, available)
        self.assertEqual(correction[1].abs().sum().item(), 0)
        self.assertEqual(correction[:,[0,3,4,5]].abs().sum().item(), 0)
        self.assertTrue((correction.abs() <= .5).all())

    def test_subject_key_does_not_depend_on_annotation_id(self):
        self.assertEqual(subject_key('a.jpg',[0,0,1,1]), subject_key('a.jpg',[0,0,1,1]))
        self.assertNotEqual(subject_key('a.jpg',[0,0,1,1]), subject_key('b.jpg',[0,0,1,1]))


if __name__ == '__main__': unittest.main()
