"""User-run synthetic gradient-route tests; no pretrained weights or image inference."""
import copy
import unittest
import torch
from captioning.models import CaptionDecoder
from region_model import RegionCaptionModel
from visual_only_model import VisualOnlyModel
from test_region_model import Base,batch


def models(actual_decoder=False,device='cpu'):
    torch.manual_seed(42);base=Base()
    if actual_decoder:base.decoder=CaptionDecoder(9,d_model=32,nhead=4,num_layers=1)
    return (RegionCaptionModel(copy.deepcopy(base),'G').to(device).eval(),
            VisualOnlyModel(copy.deepcopy(base)).to(device).eval())


def assert_routes(check,model,data):
    model.zero_grad(set_to_none=True)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    references=dict(model.decoder.named_parameters())
    flags={k:v.requires_grad for k,v in references.items()}
    rng=torch.get_rng_state().clone()
    loss=model.phrase_loss(data,0);loss.backward()
    check.assertTrue(torch.equal(rng,torch.get_rng_state()))
    for component in (model.base.encoder.vis_projection,model.base.encoder.qformer):
        check.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in component.parameters()))
    check.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
    for name,p in model.decoder.named_parameters():
        check.assertIs(p,references[name]);check.assertEqual(p.requires_grad,flags[name])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


class Checks(unittest.TestCase):
    def test_aux_value_and_visual_gradient_match_G_but_decoder_gradient_stops(self):
        for actual in (False,True):
            g,gv=models(actual);data=batch()
            g_loss=g.phrase_loss(data,0);gv_loss=gv.phrase_loss(data,0)
            torch.testing.assert_close(gv_loss,g_loss,atol=1e-5,rtol=2e-5)
            g_loss.backward();gv_loss.backward()
            for (name,p),(other,q) in zip(g.base.encoder.named_parameters(),gv.base.encoder.named_parameters()):
                self.assertEqual(name,other)
                if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,atol=2e-5,rtol=5e-4)
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in g.decoder.parameters()))
            assert_routes(self,gv,data)

    def test_combined_decoder_gradient_equals_caption_only_CE_and_SCST(self):
        for weight in (0.,.05):
            caption,gv=models(True);data=batch()
            sample=data['captions'][:,0,:4];active=torch.ones(2,3,dtype=torch.bool)
            advantage=torch.tensor([1.,-.2])
            # Call the caption parent directly to omit G's auxiliary task.
            from pilot_model import PilotModel
            caption_loss,_,_=PilotModel.forward(caption,data,sample,active,advantage,weight,0)
            gv_loss,_,_=gv(data,sample,active,advantage,weight,0)
            caption_loss.backward();gv_loss.backward()
            for (name,p),(other,q) in zip(caption.decoder.named_parameters(),gv.decoder.named_parameters()):
                self.assertEqual(name,other)
                if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,atol=2e-5,rtol=5e-4)
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in gv.decoder.parameters()))

    def test_aux_uses_current_decoder_not_an_initial_snapshot(self):
        g,gv=models();data=batch()
        with torch.no_grad():
            g.decoder.out.bias[4]+=2.;gv.decoder.out.bias[4]+=2.
        torch.testing.assert_close(gv.phrase_loss(data,0),g.phrase_loss(data,0),atol=1e-5,rtol=2e-5)


if __name__=='__main__':unittest.main(verbosity=2)
