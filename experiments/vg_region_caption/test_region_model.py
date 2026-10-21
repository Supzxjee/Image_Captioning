"""Synthetic only; user-run on Kaggle, no pretrained downloads."""
import unittest
import torch
from torch import nn
from captioning.models import LightweightQFormer
from region_model import RegionCaptionModel, masked_queries


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_projection = nn.Linear(768,32)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(.1)
        self.qformer = LightweightQFormer(32,4,num_queries=3,num_layers=2)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(9,32); self.out=nn.Linear(32,9)
    def forward(self,tgt,memory,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        valid=(~memory_key_padding_mask).unsqueeze(-1)
        pooled=(memory*valid).sum(1)/valid.sum(1).clamp_min(1)
        return self.out(self.embedding(tgt)+pooled[:,None])


class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.decoder=Decoder()
    def build_memory(self,visual,legacy,mask):
        e=self.encoder
        memory=e.qformer(e.dropout(e.relu(e.vis_projection(visual))))
        return memory,torch.zeros(memory.shape[:2],dtype=torch.bool,device=memory.device)


def batch(device='cpu'):
    visual=torch.randn(2,197,768,device=device)
    keep=torch.zeros(2,196,dtype=torch.bool,device=device);keep[:,:8]=True
    caption=torch.tensor([[1,4,5,2,0],[1,5,2,0,0]],device=device)
    return dict(visual=visual,legacy=torch.zeros(2,1,32,device=device),legacy_mask=torch.ones(2,1,device=device),
        captions=caption[:,None].repeat(1,5,1),caption_mask=caption.eq(0)[:,None].repeat(1,5,1),
        aux_visual=visual.clone(),aux_patch_mask=keep,aux_caption=caption)


class Checks(unittest.TestCase):
    def test_masked_patch_and_cls_do_not_affect_retrieval_and_gradient(self):
        torch.manual_seed(42);model=RegionCaptionModel(Base(),'G').eval();data=batch()
        source=data['aux_visual'].requires_grad_()
        output=masked_queries(model.base.encoder,source,data['aux_patch_mask'])
        (output*torch.randn_like(output)).sum().backward()
        self.assertEqual(float(source.grad[:,0].abs().sum()),0.)
        self.assertEqual(float(source.grad[:,9:].abs().sum()),0.)
        self.assertGreater(float(source.grad[:,1:9].abs().sum()),0.)
        changed=source.detach().clone();changed[:,0]=1000;changed[:,9:]=1000
        actual=masked_queries(model.base.encoder,changed,data['aux_patch_mask'])
        torch.testing.assert_close(actual,output,atol=1e-5,rtol=2e-5)

    def test_all_patch_mask_matches_original_qformer_and_rejects_tiny_region(self):
        torch.manual_seed(42);encoder=Encoder().eval();data=batch()
        projected=encoder.dropout(encoder.relu(encoder.vis_projection(data['visual'][:,1:])))
        expected=encoder.qformer(projected)
        actual=masked_queries(encoder,data['visual'],torch.ones(2,196,dtype=torch.bool))
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=2e-5)
        with self.assertRaises(ValueError): masked_queries(encoder,data['visual'],torch.zeros(2,196,dtype=torch.bool))

    def test_shared_caption_path_no_extra_params_and_aux_ce_gradients(self):
        models=[]
        for arm in ('G','T'):
            torch.manual_seed(42);models.append(RegionCaptionModel(Base(),arm).eval())
        data=batch()
        for a,b in zip(models[0].memory(data),models[1].memory(data)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        for model in models:
            self.assertIsNone(model.extra)
            self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.base.parameters()))
            rng=torch.get_rng_state().clone()
            loss=model.phrase_loss(data,0);loss.backward()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for component in (model.base.encoder.vis_projection,model.base.encoder.qformer,model.decoder):
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in component.parameters()))
            self.assertTrue(bool(torch.isfinite(loss)))
        self.assertNotEqual(models[0].last_aux['mean_aux_patches'],models[1].last_aux['mean_aux_patches'])


if __name__=='__main__': unittest.main(verbosity=2)
