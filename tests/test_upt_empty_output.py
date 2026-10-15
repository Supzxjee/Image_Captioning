import unittest
import torch
from captioning.upt_reference import infer
from captioning.hoi_caption_reranking import evidence_inventory,candidate_evidence,VERBS


class FakeHead(torch.nn.Module):
    def __init__(self, pairs):
        super().__init__(); self.pairs=pairs

    def forward(self, features, sizes, props):
        n=self.pairs
        return (torch.zeros((n,6)),[torch.zeros((2,n,6))],
                [torch.zeros(n,dtype=torch.long)],[torch.ones(n,dtype=torch.long)],
                [torch.ones(n,dtype=torch.long)],[])


class FakeUPT(torch.nn.Module):
    def __init__(self, pairs=0, fail=False, normal=False):
        super().__init__();self.interaction_head=FakeHead(pairs);self.fail=fail;self.normal=normal

    def forward(self, images):
        props={'boxes':torch.tensor([[0.,0.,20.,20.],[30.,30.,50.,50.]]),
               'scores':torch.tensor([.9,.8]),'labels':torch.tensor([16,17])}
        self.interaction_head(None,None,[props])
        if self.fail:raise RuntimeError('real model failure')
        if not self.normal:return []
        return [{'boxes':props['boxes'],'pairing':torch.tensor([[0],[1]]),
                 'scores':torch.tensor([.123]),'labels':torch.tensor([1]),
                 'objects':torch.tensor([17]),'size':torch.tensor(images[0].shape[-2:])}]


class EmptyUPTTests(unittest.TestCase):
    def test_empty_output_keeps_image_and_returns_unknown_neutral_evidence(self):
        model=FakeUPT(); prediction=infer(model,torch.zeros((3,64,96)))
        self.assertEqual(prediction['empty_reason'],'no_human_object_pairs')
        self.assertEqual(tuple(prediction['official_output']['pairing'].shape),(2,0))
        self.assertEqual(prediction['size'].tolist(),[64,96])
        self.assertEqual(len(prediction['boxes']),2)
        inventory=evidence_inventory(prediction,['person']*80,list(VERBS),[[] for _ in range(80)])
        self.assertEqual(inventory,{})
        evidence=candidate_evidence('A man riding a bicycle.',inventory)
        self.assertEqual(evidence['feature'],0.)
        self.assertEqual(evidence['claims'][0]['status'],'unknown')
        self.assertFalse(model.interaction_head._forward_hooks)

    def test_nonempty_pairs_with_missing_output_is_not_silently_ignored(self):
        model=FakeUPT(pairs=1)
        with self.assertRaisesRegex(RuntimeError,'nonempty'):infer(model,torch.zeros((3,64,96)))
        self.assertFalse(model.interaction_head._forward_hooks)

    def test_real_model_errors_propagate_and_hook_is_removed(self):
        model=FakeUPT(fail=True)
        with self.assertRaisesRegex(RuntimeError,'real model failure'):infer(model,torch.zeros((3,64,96)))
        self.assertFalse(model.interaction_head._forward_hooks)

    def test_normal_upstream_scores_preserved(self):
        prediction=infer(FakeUPT(pairs=1,normal=True),torch.zeros((3,64,96)))
        self.assertTrue(torch.equal(prediction['official_output']['scores'],torch.tensor([.123])))
        self.assertNotIn('empty_reason',prediction)


if __name__=='__main__':unittest.main()
