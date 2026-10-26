"""User-run segmentation API and one-sided gradient tests, no pretrained model."""
import numpy as np
import torch
from pycocotools import mask as api
from run_prepare import decode_segmentation
from profile_stale import stale_loss,class_score

if __name__=='__main__':
    polygon=[[0,0,4,0,4,4,0,4]]
    decoded=decode_segmentation(polygon,10,10)
    assert decoded.shape==(10,10) and decoded.sum()==16
    rle=api.encode(np.asfortranarray(decoded.astype('u1')))
    np.testing.assert_array_equal(decoded,decode_segmentation(rle,10,10))
    stale=torch.tensor([-.5,-2.],requires_grad=True);neutral=torch.tensor([-1.,-1.],requires_grad=True)
    loss=stale_loss(stale,neutral);loss.backward()
    assert neutral.grad is None and stale.grad.tolist()==[.5,0.]
    logits=torch.zeros(1,2,8,requires_grad=True)
    score=class_score(logits,dict(rows=[dict(reference=dict(position=1),alias_ids=[4,5])]))
    torch.testing.assert_close(score.exp(),torch.tensor([.25]))
    print('Segmentation polygon/RLE equivalence, detached control gradient, alias probability sum PASS',flush=True)
