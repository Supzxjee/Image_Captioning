"""NumPy checks of the proposed paired-set formula; not model feasibility."""
import json
from pathlib import Path
import numpy as np

def joint_loss(masses,cosines,tau=.1):
    m=np.asarray(masses,dtype=float);c=np.asarray(cosines,dtype=float)
    if m.shape!=c.shape or m.ndim!=1 or tau<=0:raise ValueError('Invalid paired arrays.')
    if not len(m):return None
    if not np.isfinite(m).all() or not np.isfinite(c).all() or (m<0).any() or (m>1).any() or (abs(c)>1).any():raise ValueError('Invalid mass/cosine.')
    z=np.log(np.maximum(m,1e-300))-(1-c)/tau
    maximum=z.max()
    return float(-maximum-np.log(np.exp(z-maximum).mean()))

def main():
    mass=[.99,.01]
    paired=joint_loss(mass,[1.,0.]);crossed=joint_loss(mass,[0.,1.])
    assert paired<crossed
    # Independent marginal losses cannot tell the two assignments apart.
    unpaired=lambda c:float(-np.log(np.mean(mass))-np.log(np.mean(np.exp(-(1-np.asarray(c))/.1))))
    assert abs(unpaired([1.,0.])-unpaired([0.,1.]))<1e-12
    assert abs(paired-joint_loss(mass[::-1],[0.,1.]))<1e-12
    assert joint_loss([],[]) is None
    tiny=joint_loss([1e-200],[-1.]);assert np.isfinite(tiny)
    small_box=joint_loss([.2],[1.]);assert abs(small_box+np.log(.2))<1e-12
    result=dict(scope='Formula sanity only; no neural module/parser/inference/training.',
        paired_loss=paired,crossed_pair_loss=crossed,independent_marginal_loss_both=unpaired([1.,0.]),
        paired_order_invariance=True,empty_target_group='skip, not negative',stable_tiny_mass_loss=tiny,
        perfect_semantic_but_max_mass_point2_loss_floor=small_box,
        limitations='Finite positive sets are incomplete; these checks do not establish grounding, caption gain or novelty.')
    Path(__file__).with_name('MATH_SANITY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
