import unittest
import torch
from captioning.interaction_regions import region_weights,boxes_in_crop,pool_regions,InteractionRegionVerifier


class RegionTests(unittest.TestCase):
    def test_disjoint_tiny_boxes_preserve_fractional_support(self):
        boxes=torch.tensor([[0.,0.,.01,.01],[.8,.8,.9,.9]])
        weights=region_weights(boxes)
        self.assertAlmostEqual(float(weights[0].sum()/196),.0001,places=6)
        self.assertEqual(float(weights[2].sum()),0.)
        self.assertGreater(float(weights[3].sum()),0.)
        self.assertTrue(torch.allclose(weights[0]+weights[1]-weights[2]+weights[4],torch.ones(196),atol=1e-6))

    def test_crop_rounding_and_overlap_empty_pool(self):
        boxes=boxes_in_crop([.1,.2,.3,.4],[.5,.2,.7,.4],(5,10,75,50),100,100)
        self.assertAlmostEqual(float(boxes[0,0]),5/70,places=6)
        patches=torch.ones(196,8);features,valid,mass=pool_regions(patches,torch.ones(8),boxes)
        self.assertFalse(bool(valid[2]));self.assertEqual(float(features[2].abs().sum()),0.)
        self.assertTrue(torch.isfinite(features).all());self.assertEqual(float(mass[2]),0.)

    def test_same_box_bridge_and_background_empty(self):
        weights=region_weights(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]))
        self.assertTrue(torch.equal(weights[0],weights[2]))
        self.assertEqual(float(weights[3].sum()),0.);self.assertEqual(float(weights[4].sum()),0.)

    def test_ordered_content_and_matched_ablation(self):
        torch.manual_seed(42);model=InteractionRegionVerifier(vision_dim=8,text_dim=8).eval()
        regions=torch.randn(2,6,8);valid=torch.ones(2,6,dtype=torch.bool);mass=torch.ones(2,6);g=torch.randn(2,10);t=torch.randn(2,6,8)
        changed=regions.clone();changed[:,:5]=100.
        self.assertFalse(torch.equal(model(regions,valid,mass,g,t),model(changed,valid,mass,g,t)))
        self.assertTrue(torch.equal(model(regions,valid,mass,g,t,True),model(changed,valid,mass,g,t,True)))
        swapped=regions.clone();swapped[:,[0,1]]=regions[:,[1,0]]
        self.assertFalse(torch.equal(model(regions,valid,mass,g,t),model(swapped,valid,mass,g,t)))
        self.assertFalse(torch.equal(model(regions,valid,mass,g,t),model(regions,valid,mass,g,t.roll(1,1))))


if __name__=='__main__':unittest.main()
