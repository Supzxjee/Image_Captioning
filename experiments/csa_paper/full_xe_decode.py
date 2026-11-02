"""User-run batched beam search. Same scoring/EOS policy as model.beam_search."""
def batched_beam(model,vo,vr,valid,beams=5,maximum=20):
    import torch
    if model.training or maximum<1 or not 1<=beams<=model.c.vocab-2:raise ValueError('Invalid evaluation beam configuration')
    batch=len(vo);vocab=model.c.vocab
    seq=torch.ones((batch,1,1),dtype=torch.long,device=vo.device)
    scores=torch.zeros((batch,1),device=vo.device)
    finished=torch.zeros((batch,1),dtype=torch.bool,device=vo.device)
    for _ in range(maximum):
        width=seq.shape[1]
        logits=model.decode(seq.flatten(0,1),vo.repeat_interleave(width,0),vr.repeat_interleave(width,0),
                            valid.repeat_interleave(width,0))[:,-1]
        logp=logits.float().log_softmax(-1).reshape(batch,width,vocab)
        logp[:,:,:2]=-torch.inf
        logp=logp.masked_fill(finished[:,:,None],-torch.inf)
        logp[:,:,0]=torch.where(finished,torch.zeros_like(scores),logp[:,:,0])
        scores,flat=(scores[:,:,None]+logp).flatten(1).topk(beams,dim=1)
        parent,token=flat//vocab,flat%vocab
        seq=torch.cat((seq.gather(1,parent[:,:,None].expand(-1,-1,seq.shape[-1])),token[:,:,None]),dim=2)
        finished=finished.gather(1,parent)|(token==2)
        if finished.all():break
    return seq,scores

def beam_parity(model,features,valid):
    import torch
    from model import beam_search
    model.eval()
    with torch.inference_mode():
        vo,vr=model.encode(features,valid)
        actual,scores=batched_beam(model,vo,vr,valid)
        reference=beam_search(model,vo,vr,valid)
        for i,(seq,expected) in enumerate(reference):
            # Other images may need more iterations; finished beams only append PAD.
            for j in range(5):
                def content(x):
                    ids=x.tolist();return ids[:ids.index(2)+1] if 2 in ids else ids
                if content(actual[i,j])!=content(seq[j]):raise ValueError('Batched/per-image beam IDs differ')
            torch.testing.assert_close(scores[i],expected,rtol=1e-5,atol=5e-5)
    return dict(images=len(features),beams=5,maximum=20,passed=True)
