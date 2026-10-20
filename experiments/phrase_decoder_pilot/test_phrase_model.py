"""Synthetic Kaggle CPU preflight. No real checkpoint/cache, no pretrained download."""
import unittest
import torch
from torch import nn
from captioning.models import CaptionDecoder
from phrase_model import PhraseModel

torch.set_num_threads(2)


def synthetic(device='cpu'):
    torch.manual_seed(42)
    model=PhraseModel(CaptionDecoder(16),nn.Linear(768,512)).to(device).eval()
    patches=torch.randn(4,196,768,device=device);global_memory=torch.randn(4,32,512,device=device)
    boxes=torch.tensor([[[0,0,.5,1],[.5,0,1,1]],[[0,0,.001,.001],[0,0,.001,.001]]]*2,dtype=torch.float32,device=device)
    geo=torch.randn(4,8,device=device)
    targets=torch.randint(4,16,(8,8),device=device);targets[:,0]=1;targets[:,-2]=2;targets[:,-1]=0
    positive=torch.tensor([[0,1],[1,0]],device=device)
    return model,(patches,global_memory,boxes,geo,targets,positive)


class Checks(unittest.TestCase):
    def test_only_loss_weight_differs_and_frozen_original_stays_fixed(self):
        model,batch=synthetic();frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        zero,ce,cross=model(*batch,0.);weighted,ce2,cross2=model(*batch,.25)
        torch.testing.assert_close(zero,ce);torch.testing.assert_close(weighted,ce+.25*cross)
        torch.testing.assert_close(ce,ce2);torch.testing.assert_close(cross,cross2)
        params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad();loss,_,_=model(*batch,.25);loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params));optimizer.step()
        for n,p in model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0);self.assertIsNone(p.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.evidence.parameters()),0)

    def test_memory_roles_mask_fallback_and_off(self):
        model,(p,g,b,geo,targets,positive)=synthetic()
        memory,pad,attn=model.memory(p,g,b,geo)
        self.assertEqual(tuple(memory.shape),(4,38,512));self.assertEqual(int(pad.sum()),0)
        torch.testing.assert_close(attn.sum(-1),torch.ones(4,6))
        self.assertEqual(int(torch.count_nonzero(attn[1,0])),1)
        off,mask,_=model.memory(p,g,b,geo,off=True)
        torch.testing.assert_close(off[:,:32],g,rtol=0,atol=0);self.assertTrue(mask[:,-6:].all())
        self.assertEqual(int(torch.count_nonzero(off[:,-6:])),0)

    def test_zero_lora_preserves_decoder_and_word_scores_ignore_eos(self):
        model,batch=synthetic();mha=model.decoder.transformer_decoder.layers[0].multihead_attn
        torch.testing.assert_close(mha.in_proj_weight,mha.parametrizations.in_proj_weight.original,rtol=0,atol=0)
        p,g,b,geo,targets,positive=batch;memory,pad,_=model.memory(p,g,b,geo)
        scores,nll=model.scores(memory,pad,targets)
        with torch.no_grad():
            logits=model.decoder(targets[:,:-1],memory.repeat_interleave(2,0),tgt_key_padding_mask=targets[:,:-1].eq(0),memory_key_padding_mask=pad.repeat_interleave(2,0))
            lp=logits.log_softmax(-1).gather(-1,targets[:,1:,None]).squeeze(-1)
            keep=targets[:,1:].ne(0);words=keep&targets[:,1:].ne(2)
            torch.testing.assert_close(scores,(lp*words).sum(-1)/words.sum(-1))
            torch.testing.assert_close(nll,-(lp*keep).sum(-1)/keep.sum(-1))
        seq=model.greedy(memory,pad,8);self.assertTrue(seq[:,0].eq(1).all());self.assertLessEqual(seq.shape[1],8)


if __name__=='__main__':unittest.main(verbosity=2)
