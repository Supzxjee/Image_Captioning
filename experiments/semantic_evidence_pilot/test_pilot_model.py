"""Kaggle CPU preflight; no pretrained downloads or notebook launch."""
import unittest
import torch
from torch import nn
from pilot_model import EvidenceMemory, PilotModel


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(9,512);self.out=nn.Linear(512,9)
    def forward(self,tgt,memory,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        keep=(~memory_key_padding_mask)[...,None]
        value=(memory*keep).sum(1)/keep.sum(1).clamp_min(1)
        return self.out(self.embedding(tgt)+value[:,None])


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(768,512);self.decoder=FakeDecoder()
    def build_memory(self,visual,legacy,mask):
        memory=torch.cat((legacy,self.projection(visual[:,:2])),1)
        padding=torch.cat((mask.eq(0),torch.zeros(len(visual),2,dtype=torch.bool,device=visual.device)),1)
        return memory,padding


def batch(device='cpu'):
    b=2
    bounds=torch.tensor([0.,0.,1.,1.],device=device).expand(b,9,3,4).clone()
    d=dict(visual=torch.randn(b,197,768,device=device),legacy=torch.randn(b,2,512,device=device),
           legacy_mask=torch.ones(b,2,dtype=torch.long,device=device),text=torch.randn(b,9,512,device=device),
           roles=torch.tensor([[0,1,2]*3]*b,device=device),valid=torch.ones(b,9,dtype=torch.bool,device=device),bounds=bounds,
           phrase_ids=torch.arange(1,19,device=device).reshape(b,9),image_id=torch.arange(b,device=device),
           captions=torch.tensor([[[1,4,5,2,0]]*5]*b,device=device),
           caption_mask=torch.tensor([[[False,False,False,False,True]]*5]*b,device=device))
    d.update(swap_text=d['text'].flip(0),swap_visual=d['visual'].flip(0),swap_bounds=bounds.flip(0),swap_valid=d['valid'].clone())
    return d


class EvidenceChecks(unittest.TestCase):
    def test_attention_mask_and_nearest_patch(self):
        torch.manual_seed(42);m=EvidenceMemory();d=batch()
        bounds=d['bounds'][0,:2].clone();bounds[0,2]=torch.tensor([0.,0.,.5,.5]);bounds[1,2]=torch.tensor([0.,0.,.001,.001])
        _,_,a=m.attend(d['visual'][0,1:].expand(2,-1,-1),d['text'][0,:2],d['roles'][0,:2],bounds)
        outside=(m.xy[:,0]>.5)|(m.xy[:,1]>.5)
        self.assertEqual(int(torch.count_nonzero(a[0,outside])),0)
        self.assertEqual(int(torch.count_nonzero(a[1])),1)
        torch.testing.assert_close(a.sum(-1),torch.ones(2))

    def test_text_changes_attention_but_cannot_leak_into_constant_image_values(self):
        torch.manual_seed(42);m=EvidenceMemory();d=batch();p=d['visual'][0,1:].expand(2,-1,-1)
        e,s,a=m.attend(p,d['text'][0,:2],torch.ones(2,dtype=torch.long),d['bounds'][0,:2])
        self.assertGreater(float((a[0]-a[1]).abs().sum().detach()),1e-5)
        constant=p[:,:1].expand_as(p)
        e,_,_=m.attend(constant,d['text'][0,:2],torch.ones(2,dtype=torch.long),d['bounds'][0,:2])
        torch.testing.assert_close(e[0],e[1],atol=2e-5,rtol=2e-5)

    def test_alignment_eligibility_and_gradients(self):
        torch.manual_seed(42);m=PilotModel(FakeBase(),'E');d=batch()
        loss,_,_=m(d,None,None,None,0.,0);loss.backward()
        self.assertEqual(m.last_alignment['pairs'],18)
        for part in (m.extra.query,m.extra.key,m.extra.value):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        d['phrase_ids'].fill_(1);m.zero_grad();loss,_,_=m(d,None,None,None,0.,0)
        self.assertEqual(m.last_alignment['pairs'],0)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        d['valid'].zero_();loss,_,_=m(d,None,None,None,0.,0)
        self.assertEqual(m.last_alignment['pairs'],0);self.assertTrue(torch.isfinite(loss))

    def test_off_preserves_base_and_ce_scst_backward(self):
        for arm in ('A','E'):
            torch.manual_seed(42);d=batch();m=PilotModel(FakeBase(),arm).eval()
            old,pad=m.base.build_memory(d['visual'],d['legacy'],d['legacy_mask'])
            mem,mask=m.memory(d,'off');t=d['captions'][:,0,:-1]
            torch.testing.assert_close(m.decoder(t,old,memory_key_padding_mask=pad),m.decoder(t,mem,memory_key_padding_mask=mask))
            loss,_,_=m(d,d['captions'][:,0,:4],torch.ones(2,3,dtype=torch.bool),torch.tensor([1.,-.3]),.05,0)
            loss.backward();self.assertTrue(torch.isfinite(loss))


if __name__=='__main__': unittest.main(verbosity=2)
