"""User-run checks for the existing per-image CSA beam decoder."""
import math


def check_beam_rows(rows,batch,vocab,maximum=20,beams=5):
    """Pure-data check; tensors converted by the user-run caller."""
    if len(rows)!=batch:raise ValueError('Wrong beam image count')
    for sequences,scores in rows:
        if len(sequences)!=beams or len(scores)!=beams:raise ValueError('Wrong beam width')
        if any(not math.isfinite(x) for x in scores) or any(a<b for a,b in zip(scores,scores[1:])):
            raise ValueError('Invalid/unsorted beam scores')
        for seq in sequences:
            if not 2<=len(seq)<=maximum+1 or seq[0]!=1:raise ValueError('Invalid beam prefix/length')
            content=seq[1:];end=content.index(2) if 2 in content else len(content)
            if any(type(x)!=int or not 2<=x<vocab for x in content[:end]):raise ValueError('BOS/PAD/invalid content token')
            if end<len(content) and any(x!=0 for x in content[end+1:]):raise ValueError('Non-PAD token after EOS')
            if end==len(content) and len(content)!=maximum:raise ValueError('Truncated unfinished beam')


def beam_preflight(model,features,valid):
    import torch
    from model import beam_search
    model.eval()
    with torch.inference_mode():
        vo,vr=model.encode(features,valid)
        rows=beam_search(model,vo,vr,valid,beams=5,maximum=20)
        check_beam_rows([(seq.cpu().tolist(),scores.cpu().tolist()) for seq,scores in rows],len(features),model.c.vocab)
    return dict(images=len(features),beams=5,maximum=20,passed=True,decoder='model.beam_search per-image',
                batched_decoder=False,checks='width/finite sorted scores/BOS/EOS/PAD/maximum')
