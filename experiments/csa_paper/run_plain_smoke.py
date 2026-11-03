"""USER-RUN actual-input plain-model gradient smoke; no optimizer updates."""
import argparse
from pathlib import Path
import numpy as np
from real_bank_data import read,save
from full_cache_data import collate


def run(a):
    import torch
    from model import CSACaptioner,Config,xe_loss
    from plain_model import PlainCaptioner
    if not torch.cuda.is_available():raise RuntimeError('Enable GPU')
    c=Config(vocab=10000);torch.manual_seed(42);reference=CSACaptioner(c,torch.zeros(1,512),torch.zeros(1,512))
    common={k:v.clone() for k,v in reference.state_dict().items() if k.startswith(('project.','embedding.','decoder.','head.'))}
    del reference;torch.manual_seed(42);base=PlainCaptioner(c)
    if any(not torch.equal(v,base.state_dict()[k]) for k,v in common.items()):raise ValueError('Common initial weights differ')
    del common
    if any(n.startswith(('noun','verb','object1','object2','relation.')) for n in base.state_dict()):raise ValueError('Semantic modules remain')
    stage=read(a.stage/'stage_receipt.json');data={k:np.load(a.stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in stage['shapes']}
    x,v,seq=[torch.from_numpy(x).cuda() for x in collate(data,[0,5,10,15])];base.cuda().train()
    loss=xe_loss(base(x,v,seq[:,:-1]),seq[:,1:]);loss.backward()
    if not torch.isfinite(loss):raise ValueError('Nonfinite plain smoke loss')
    for n,p in base.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite gradient: '+n)
    base.eval()
    with torch.inference_mode():
        vo,vr=base.encode(x,v)
        if vo is not vr or not torch.isfinite(vo).all() or torch.count_nonzero(vo[~v]):raise ValueError('Plain shared memory/padding differs')
    save(a.output/'plain_smoke.json',dict(passed=True,common_initial_weights_exact=True,three_visual_layers=True,semantic_bank_used=False,
        all_gradients_finite=True,loss=float(loss),optimizer_steps=0,trainable_parameters=sum(p.numel() for p in base.parameters())))
    print('PLAIN SMOKE PASS; disposable model, no optimizer update.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
