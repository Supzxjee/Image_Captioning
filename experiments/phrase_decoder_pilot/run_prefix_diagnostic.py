"""Evaluation only; replay saved adapters before interpreting target-pair interventions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
from phrase_data import digest,identity
from run_phrase import load,save,corpus,build,Batches,decode,BASE_SHA


def common_branch(left,right):
    n=0
    while n<min(len(left),len(right)) and left[n]==right[n]:n+=1
    assert n>0 and n<min(len(left),len(right)),'Need a genuine lexical contrast'
    return left[:n], [left[n],right[n]]


def restore(model,adapter,cfg,expected,tokenizer,torch):
    packet=torch.load(adapter,map_location='cpu',weights_only=True)
    for k,v in dict(format='phrase_decoder_adaptation_v1',arm=expected['arm'],
                    source_checkpoint_sha256=BASE_SHA,protocol_sha256=expected['protocol_sha256'],
                    manifest_sha256=expected['manifest_sha256'],vocabulary_sha256=cfg['vocabulary_sha256'],completed_updates=300).items():
        assert packet[k]==v, f'Wrong adaptation: {k}'
    state=packet['trainable_state'];params=dict(model.named_parameters())
    assert set(state)=={n for n,p in params.items() if p.requires_grad},'Incomplete/unexpected adapter keys'
    with torch.no_grad():
        for name,tensor in state.items():
            assert tensor.shape==params[name].shape and tensor.dtype==params[name].dtype and torch.isfinite(tensor).all(),name
            params[name].copy_(tensor)


def worker(config):
    import torch
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);configure_strict_runtime(True);torch.manual_seed(42)
    cfg,m=load(config);arm=('CE','CROSS')[rank];expected=cfg['expected'][arm]
    assert digest(cfg['stage'])==cfg['stage_sha256'] and digest(cfg['checkpoint'])==BASE_SHA
    assert cfg['stage_tensor_sha256']==expected['stage_tensor_sha256'],'Original staged features do not replay'
    _,tok=corpus(cfg);assert identity(tok.word2idx)==cfg['vocabulary_sha256']
    model=build(cfg,tok);assert model_digest(model)==expected['initial_sha256'],'Initial architecture/weights changed'
    adapter=cfg['adapters'][arm];restore(model,adapter,cfg,expected,tok,torch)
    start=model_digest(model);assert start==expected['final_sha256'],'Adapter fails complete model replay'
    model.requires_grad_(False);batches=Batches(cfg,m,tok);rows=[];maxlen=m['max_length']
    shared=[tok.bos_idx]+[tok.word2idx[w] for w in ('a','person')]

    @torch.no_grad()
    def continuation(memory,padding,prefix):
        seq=torch.tensor([prefix]*len(memory),device='cuda');done=torch.zeros(len(memory),dtype=torch.bool,device='cuda')
        for _ in range(maxlen-len(prefix)):
            logits=model.decoder(seq,memory,tgt_key_padding_mask=seq.eq(0),memory_key_padding_mask=padding)[:,-1]
            nxt=torch.where(done,torch.zeros(len(seq),dtype=torch.long,device='cuda'),logits.argmax(-1))
            seq=torch.cat((seq,nxt[:,None]),1);done|=nxt.eq(2)
            if bool(done.all()):break
        return dict(captions=[decode(s,tok) for s in seq],truncated=[not bool(s.eq(2).any()) for s in seq])

    @torch.no_grad()
    def prefix_scores(memory,padding,prefix,branches):
        seq=torch.tensor([prefix]*len(memory),device='cuda')
        lp=model.decoder(seq,memory,tgt_key_padding_mask=seq.eq(0),memory_key_padding_mask=padding)[:,-1].float().log_softmax(-1)
        results=[]
        for l in lp:
            ids=l.topk(10).indices.tolist()
            results.append(dict(top10=[dict(token=tok.idx2word[i],probability=float(l[i].exp())) for i in ids],
                branch_tokens=[dict(token=tok.idx2word[i],log_probability=float(l[i]),rank=1+int((l>l[i]).sum())) for i in branches],
                description0_minus_description1_logp=float(l[branches[0]]-l[branches[1]])))
        return results

    with torch.no_grad():
        for group in m['groups']:
            if not group['same_image']:continue
            key=group['group_id'];p,g,b,geo,targets,positive=batches([key]);assert torch.equal(p[0],p[1]) and torch.equal(g[0],g[1])
            desc=[[tok.bos_idx]+[tok.word2idx[w] for w in text.split()] for text in group['descriptions']]
            branch_prefix,branches=common_branch(*desc);conditions={}
            for name in ('natural','evidence_off','explicit_geometry_swap'):
                used_geo=geo.flip(0) if name=='explicit_geometry_swap' else geo
                memory,padding,attention=model.memory(p,g,b,used_geo,off=name=='evidence_off')
                scores,nll=model.scores(memory,padding,targets)
                lp=model.decoder(targets[:,:-1],memory.repeat_interleave(2,0),tgt_key_padding_mask=targets[:,:-1].eq(0),memory_key_padding_mask=padding.repeat_interleave(2,0)).float().log_softmax(-1)
                selected=lp.gather(-1,targets[:,1:,None]).squeeze(-1)
                details=[]
                for target,values in zip(targets,selected):
                    details.append([dict(token=tok.idx2word[int(t)],log_probability=float(v)) for t,v in zip(target[1:],values) if int(t)!=tok.pad_idx])
                seq=model.greedy(memory,padding,maxlen)
                conditions[name]=dict(scores=scores.reshape(2,2).cpu().tolist(),
                    from_bos=dict(captions=[decode(s,tok) for s in seq],truncated=[not bool(s.eq(2).any()) for s in seq]),
                    from_a_person=continuation(memory,padding,shared),
                    after_a_person=prefix_scores(memory,padding,shared,branches),
                    after_common_description_prefix=prefix_scores(memory,padding,branch_prefix,branches),
                    from_common_description_prefix=continuation(memory,padding,branch_prefix),
                    teacher_forced_token_logp_event_major=details,
                    attention=attention.cpu().tolist())
            old=expected['records'][key]['conditions']
            for name,oldname in [('natural','on'),('evidence_off','off')]:
                now=conditions[name];prior=old[oldname]
                torch.testing.assert_close(torch.tensor(now['scores']),torch.tensor(prior['scores']),rtol=1e-5,atol=1e-5)
                assert now['from_bos']['captions']==prior['captions'],'Greedy replay changed: do not interpret interventions'
            rows.append(dict(group_id=key,partition=next(s for s,ids in m['partitions'].items() if key in ids),
                image_id=group['events'][0]['image_id'],descriptions=group['descriptions'],positive_indices=group['positive_indices'],
                common_prefix_tokens=[tok.idx2word[i] for i in branch_prefix],conditions=conditions))
            print(arm,key,'REPLAY_PASS',flush=True)
    assert len(rows)==4 and model_digest(model)==start
    save(Path(cfg['output'])/f'prefix_{arm}.json',dict(arm=arm,status='EVALUATION_ONLY_REPLAY_PASS',
        adaptation_sha256=digest(adapter),model_sha256=start,weights_unchanged=True,runtime=runtime_manifest(),groups=rows,
        limits=['Two previously inspected same-image holdout groups plus two fit controls; not population estimates',
                'Shared prefix is a language intervention; it is not an inference-time solution or proof of exposure bias',
                'Explicit geometry swap is inconsistent/OOD; masks and membership retain geometry; it does not isolate all geometry',
                'No automatic semantic labels; read phrases against selected person-object boxes']))


def finish(config):
    cfg,m=load(config);out=Path(cfg['output']);images=out/'images';images.mkdir(exist_ok=True)
    from run_phrase import image_path
    from html import escape as h
    blocks=['<meta charset="utf-8"><title>Pair-prefix diagnostic</title><style>body{font:16px Arial;max-width:1000px;margin:auto}img{max-width:95%}pre{white-space:pre-wrap}.image{position:relative;display:inline-block}.box{position:absolute;border:3px solid;box-sizing:border-box}</style><h1>Evaluation only: target pair and prefix</h1><p>Green: selected person; orange: selected object. Two holdout groups and two fit controls. No automated semantic correctness.</p>']
    results={arm:json.loads((out/f'prefix_{arm}.json').read_text()) for arm in ('CE','CROSS')}
    for group in m['groups']:
        if not group['same_image']:continue
        blocks.append('<h2>'+h(group['group_id'])+'</h2>')
        for i,event in enumerate(group['events']):
            shutil.copy2(image_path(cfg,event),images/event['filename'])
            blocks.append(f'<h3>Event {i}: human target {h(group["descriptions"][group["positive_indices"][i]])}</h3><div class="image"><img src="images/{h(event["filename"])}">')
            for box,color in [(event['subject_box'],'green'),(event['object_box'],'orange')]:
                x,y,r,b=box;blocks.append(f'<span class="box" style="left:{100*x}%;top:{100*y}%;width:{100*(r-x)}%;height:{100*(b-y)}%;border-color:{color}"></span>')
            blocks.append('</div>')
            for arm,result in results.items():
                row=next(r for r in result['groups'] if r['group_id']==group['group_id'])
                for condition,c in row['conditions'].items():
                    show={k:v[i] if isinstance(v,list) else v for k,v in c.items() if k in ('after_a_person','after_common_description_prefix')}
                    show.update({k:c[k]['captions'][i] for k in ('from_bos','from_a_person','from_common_description_prefix')})
                    blocks.append('<h4>'+h(arm+' / '+condition)+'</h4><pre>'+h(json.dumps(show,indent=2,ensure_ascii=False))+'</pre>')
    (out/'review.html').write_text('\n'.join(blocks),encoding='utf-8')
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jpg','.html','.py','.log'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','finish']);p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](a.config)
