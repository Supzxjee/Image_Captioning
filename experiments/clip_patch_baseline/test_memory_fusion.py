"""Exercise actual fusion helper with NumPy; no Torch or models."""
import ast
from pathlib import Path
import unittest
import numpy as np
from memory_fusion import fuse_memory,update_gate_stats,merge_gate_stats


class Checks(unittest.TestCase):
    def test_old_branch_gate_cannot_remove_shared_bank_counterexample(self):
        visual=np.zeros((2,196,4),np.float32);bank=np.full_like(visual,2)
        objects=visual+bank;relations=visual+bank
        for branch in (0.,.3,1.):
            np.testing.assert_allclose(branch*objects+(1-branch)*relations,visual+bank)
            np.testing.assert_array_equal(fuse_memory(visual,objects,relations,branch,0.),visual)

    def test_strength_endpoints_broadcast_and_initial_identity(self):
        rng=np.random.default_rng(42);visual=rng.normal(size=(3,196,512)).astype(np.float32)
        objects=visual+rng.normal(size=visual.shape).astype(np.float32);relations=visual+rng.normal(size=visual.shape).astype(np.float32)
        branch=rng.uniform(size=visual.shape).astype(np.float32)
        old=branch*objects+(1-branch)*relations
        np.testing.assert_array_equal(fuse_memory(visual,objects,relations,branch,0.),visual)
        np.testing.assert_allclose(fuse_memory(visual,objects,relations,branch,1.),old,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(fuse_memory(visual,objects,relations,branch,np.full((3,196,1),.5,np.float32)),.5*(visual+old),rtol=2e-5,atol=2e-6)
        np.testing.assert_array_equal(fuse_memory(visual,visual,visual,branch,.5),visual)

    def test_patch_gate_controls_total_residual_without_changing_other_patch(self):
        visual=np.zeros((1,2,4),np.float32);objects=np.ones_like(visual);relations=objects*3
        strength=np.array([[[0],[1]]],np.float32)
        np.testing.assert_array_equal(fuse_memory(visual,objects,relations,.5,strength),[[[0]*4,[2]*4]])

    def test_accumulate_and_merge_matches_all_patch_statistics(self):
        a=np.array([0,.05,.5,1]);b=np.array([.95,.3]);one=update_gate_stats(None,a);two=update_gate_stats(None,b)
        merged=merge_gate_stats([one,two]);both=merge_gate_stats([update_gate_stats(one,b)])
        self.assertEqual(merged,both);self.assertEqual(merged['patches'],6);self.assertAlmostEqual(merged['mean'],np.r_[a,b].mean())
        self.assertAlmostEqual(merged['fraction_below_point1'],2/6);self.assertAlmostEqual(merged['fraction_above_point9'],2/6)
        for bad in ([float('nan')],[-.1],[1.1],[]):
            with self.assertRaises(ValueError):update_gate_stats(None,bad)
        with self.assertRaises(ValueError):merge_gate_stats([])

    def test_sources_parse_and_scalar_gate_recipe(self):
        here=Path(__file__).parent
        for name in ('memory_fusion.py','semantic_patch_model.py','semantic_training.py','run_full.py','make_full_notebook.py'):ast.parse((here/name).read_text(encoding='utf-8'))
        model=(here/'semantic_patch_model.py').read_text(encoding='utf-8')
        self.assertIn('nn.Linear(1024,1) if strength_gate else None',model)
        self.assertIn('return fuse_memory(visual,vo,vr,branch,strength),strength',model)
        worker=(here/'run_full.py').read_text(encoding='utf-8')
        for part in ("2500*196","merge_gate_stats","bank_prompt_strength_v1","checkpoint(promote=better)"):self.assertIn(part,worker)
        self.assertEqual(1024+1,1025)

if __name__=='__main__':unittest.main(verbosity=2)
