import unittest
import torch
from captioning.actor_compatibility import actor_supervision, ActorCompatibility, CompactActorCompatibility, actor_loss


class ActorTests(unittest.TestCase):
    def test_unknown_actor_is_not_negative(self):
        rows=[dict(image_id=1,object_id=8,subject_id=i,labels=[v,-1,-1,-1,-1,-1])
              for i,v in enumerate((1,0,-1))]
        labels,edges=actor_supervision(rows)
        self.assertEqual(labels.tolist(),[[1,-1,-1,-1,-1,-1],[0,-1,-1,-1,-1,-1],[-1]*6])
        self.assertEqual(edges.shape,(1,4))

    def test_exhaustive_keeps_all_confirmed_actor_counterparts(self):
        rows=[dict(image_id=1,object_id=8,subject_id=i,labels=[1 if i==0 else 0]*6)
              for i in range(7)]
        _, limited=actor_supervision(rows)
        _, full=actor_supervision(rows,exhaustive=True)
        self.assertGreater(len(full),len(limited))
        self.assertEqual(len(full),36)

    def test_union_has_no_effect_and_subject_receives_signal(self):
        torch.manual_seed(42)
        model=ActorCompatibility(dim=8)
        torch.nn.init.normal_(model.network[-1].weight)
        views=torch.randn(3,3,8);geometry=torch.randn(3,10)
        old=model(views,geometry)
        union=views.clone();union[:,2]=100
        self.assertTrue(torch.equal(old,model(union,geometry)))
        subject=views.clone();subject[:,0]=100
        self.assertFalse(torch.equal(old,model(subject,geometry)))

    def test_loss_rejects_unknown_edge(self):
        with self.assertRaises(ValueError):
            actor_loss(torch.zeros(2,6),torch.tensor([[1]*6,[-1]*6]),torch.tensor([[0,0,1,0]]))

    def test_compact_ignores_union_and_has_150_parameters(self):
        model=CompactActorCompatibility()
        self.assertEqual(sum(p.numel() for p in model.parameters()),150)
        torch.nn.init.normal_(model.network.weight)
        views=torch.randn(3,3,512);text=torch.randn(3,6,512);geometry=torch.randn(3,10)
        original=model(views,geometry,text)
        changed=views.clone();changed[:,2]=100
        self.assertTrue(torch.equal(original,model(changed,geometry,text)))


if __name__=='__main__':
    unittest.main()
