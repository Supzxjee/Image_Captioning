"""User-run frozen endpoint inspection; no training or decoding-policy changes."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
from real_bank_data import load_prepared,read,save,sha,HERE
from pilot_data import load_banks,event_batch

CHECKPOINT_SHA='99b1f9936b864e549182f2307c369a221e4ca4f17b0f0cad9c60f3fb1cb7017e'

def run(args):
    import torch
    from model import Config,CSACaptioner,beam_search
    started=time.monotonic()
    if sha(args.checkpoint)!=CHECKPOINT_SHA:raise ValueError('Endpoint checkpoint SHA differs')
    _,inputs,arrays,images,vocab,bank=load_prepared(args.prepared_root);banks,package=load_banks(args.bank_root,bank,inputs)
    expected=read(HERE/'xe_pilot_review_text/input_receipt.json')
    # Full SHA verified above; this is the user's own optimizer/RNG checkpoint from Output131.
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload['identity']!=expected or payload['epoch']!=15 or payload['updates']!=600 or payload['config']!=vars(Config(vocab=len(vocab))):
        raise ValueError('Checkpoint model/input/source identity differs')
    if expected['prepared_inputs']!=inputs or expected['bank_package']!=package:raise ValueError('Current cached inputs differ')
    for kind,v in zip(('nouns','verbs'),banks):
        if not torch.equal(payload['model'][kind],torch.from_numpy(v)):raise ValueError('Checkpoint bank buffer differs')
    model=CSACaptioner(Config(vocab=len(vocab)),*[torch.from_numpy(v) for v in banks]);model.load_state_dict(payload['model'],strict=True)
    del payload
    if not torch.cuda.is_available():raise RuntimeError('Enable GPU')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model.to('cuda:0').eval().requires_grad_(False)
    reference=read(HERE/'xe_pilot_review_text/captions_final.json');val=[i for i,m in enumerate(images) if m['group']=='val']
    if [images[i]['coco_id'] for i in val]!=[r['coco_id'] for r in reference]:raise ValueError('Val ID order differs')
    rows=[];score_errors=[]
    with torch.inference_mode():
        for number,(i,expected_row) in enumerate(zip(val,reference),1):
            x,valid,_=event_batch(arrays,images,[(i,0)]);x=torch.from_numpy(x).cuda();valid=torch.from_numpy(valid).cuda()
            vo,vr=model.encode(x,valid)
            greedy_ids,greedy_scores=beam_search(model,vo,vr,valid,beams=1,maximum=20)[0]
            beam_ids,beam_scores=beam_search(model,vo,vr,valid,beams=5,maximum=20)[0]
            bid=beam_ids[0].cpu().tolist()[1:];gid=greedy_ids[0].cpu().tolist()[1:]
            if bid!=expected_row['ids']:raise ValueError('Beam5 replay IDs differ for '+str(images[i]['coco_id']))
            error=abs(float(beam_scores[0])-expected_row['score']);score_errors.append(error)
            if not math.isclose(float(beam_scores[0]),expected_row['score'],rel_tol=1e-5,abs_tol=1e-4):raise ValueError('Beam5 replay score differs')
            prefix=torch.ones((1,1),dtype=torch.long,device='cuda:0');trace=[]
            for step in range(1,21):
                logp=model.decode(prefix,vo,vr,valid)[0,-1].float().log_softmax(-1)
                if not torch.isfinite(logp).all():raise ValueError('Nonfinite decode logprobs')
                legal=logp.clone();legal[:2]=-torch.inf;chosen=int(legal.argmax())
                alternative=legal.clone();alternative[2]=-torch.inf;other=int(alternative.argmax())
                trace.append(dict(step=step,chosen_id=chosen,chosen_word=vocab[chosen],chosen_probability=float(logp[chosen].exp()),
                    eos_probability=float(logp[2].exp()),eos_rank=1+int((legal>legal[2]).sum()),best_non_eos_word=vocab[other],
                    best_non_eos_probability=float(logp[other].exp())))
                prefix=torch.cat([prefix,torch.tensor([[chosen]],device='cuda:0')],dim=1)
                if chosen==2:break
            if [t['chosen_id'] for t in trace]!=gid:raise ValueError('Manualgreedy/beam1 IDs differ')
            def decoded(ids):
                content=ids[:ids.index(2)] if 2 in ids else ids
                return dict(ids=ids,caption=' '.join(vocab[t] for t in content if t!=0),words=len(content),eos=2 in ids)
            rows.append(dict(coco_id=images[i]['coco_id'],filename=images[i]['filename'],greedy=decoded(gid),beam5=decoded(bid),
                greedy_score=float(greedy_scores[0]),beam_score=float(beam_scores[0]),greedy_trace=trace,references=images[i]['captions']))
            if number%8==0:
                save(args.output/'pairs_partial.json',rows);print('Frozen decode inspection',number,'/64',flush=True)
    save(args.output/'pairs.json',rows)
    def summary(kind):
        lengths=[r[kind]['words'] for r in rows]
        return dict(images=len(rows),mean_words=float(np.mean(lengths)),min_words=min(lengths),max_words=max(lengths),
            eos=sum(r[kind]['eos'] for r in rows),unique_captions=len(set(r[kind]['caption'] for r in rows)))
    save(args.output/'result.json',dict(status='FROZEN_DECODE_CHECK_COMPLETE',checkpoint_sha256=CHECKPOINT_SHA,checkpoint_epoch=15,
        images=64,greedy=summary('greedy'),beam5=summary('beam5'),same_caption=sum(r['greedy']['ids']==r['beam5']['ids'] for r in rows),
        beam_replay_exact_ids=True,max_beam_replay_score_error=max(score_errors),manual_greedy_matches_beam1=True,
        model_eval=True,optimizer_steps=0,model_weights_frozen=True,inputs=inputs,bank_package=package,
        device=torch.cuda.get_device_name(0),worker_seconds=time.monotonic()-started,quality_metrics=False,grounding_accuracy=False))
    print('FROZEN DECODE CHECK COMPLETE; no training or decoder policy change.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prepared-root',required=True)
    p.add_argument('--bank-root',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args)
