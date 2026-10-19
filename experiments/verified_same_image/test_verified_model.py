"""Synthetic CPU preflight, mandatory in the manual Kaggle notebook."""
import unittest
import torch
from pilot_model import PilotModel,pair_loss
from synthetic_fixtures import FakeBase,batch as caption_batch


def batch(device='cpu'):
    d=caption_batch(device);n=len(d['visual'])
    d.update(aux_visual=torch.randn(n,197,768,device=device),aux_text=torch.randn(n,2,512,device=device),
             aux_bounds=d['bounds'][:,:2].clone(),aux_positive=torch.tensor([0,1],device=device),
             aux_valid=torch.ones(n,dtype=torch.bool,device=device))
    return d


class VerifiedChecks(unittest.TestCase):
    def test_positive_orientation_and_empty_loss(self):
        scores=torch.tensor([[.4,.1],[.5,.6]],requires_grad=True)
        value,info=pair_loss(scores,torch.tensor([0,1]),torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(value,torch.tensor(.05));self.assertEqual(info['accuracy'],1.)
        flipped,_=pair_loss(scores.flip(1),torch.tensor([1,0]),torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(value,flipped)
        zero,info=pair_loss(scores,torch.tensor([0,1]),torch.zeros(2,dtype=torch.bool))
        zero.backward();self.assertEqual(info['pairs'],0);torch.testing.assert_close(scores.grad,torch.zeros_like(scores))

    def test_forward_backward_ce_scst_and_replaced_loss(self):
        for weight in (0.,.05):
            torch.manual_seed(42);m=PilotModel(FakeBase(),'F').eval();d=batch()
            # The old cross-image pseudo loss must never run.
            def forbidden(*args):raise AssertionError('Old pseudo loss used')
            m.extra.alignment=forbidden
            loss,ce,rl=m(d,d['captions'][:,0,:4],torch.ones(2,3,dtype=torch.bool),torch.tensor([1.,-.3]),weight,0)
            torch.testing.assert_close(loss.detach(),ce+weight*rl+.1*m.last_alignment['loss'])
            self.assertEqual(m.last_alignment['pairs'],2);loss.backward()
            for part in (m.extra.query,m.extra.key,m.extra.value):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            d['aux_valid'].zero_();m.zero_grad();loss,_,_=m(d,None,None,None,0.,0)
            self.assertEqual(m.last_alignment['pairs'],0);loss.backward();self.assertTrue(torch.isfinite(loss))

    def test_off_is_original_decoder(self):
        torch.manual_seed(42);m=PilotModel(FakeBase(),'F').eval();d=batch()
        old,pad=m.base.build_memory(d['visual'],d['legacy'],d['legacy_mask']);mem,mask=m.memory(d,'off')
        t=d['captions'][:,0,:-1]
        torch.testing.assert_close(m.decoder(t,old,memory_key_padding_mask=pad),m.decoder(t,mem,memory_key_padding_mask=mask))


if __name__=='__main__':unittest.main(verbosity=2)
