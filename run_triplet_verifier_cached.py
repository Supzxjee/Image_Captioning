"""Whole-triplet development trial using existing image/text caches only."""
import argparse,json
from pathlib import Path
from collections import Counter
import torch
from captioning.triplet_verifier import TripletVerifier
from captioning.actor_compatibility import pair_geometry,actor_edge_metrics
from captioning.supervised_relations import contrast_edges,RELATIONS
from captioning.instance_contrast_alignment import supervised_alignment_loss
from captioning.selected_relation_prompts import build_selected_prompts
from run_actor_compatibility_cached import read_archive
from run_instance_alignment_diagnostic import metrics,fit_thresholds
from evaluate_frozen_residual import sha


def main():
    p=argparse.ArgumentParser()
    for name in ('v2-source','actor-source','candidate-source','scored','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--relation-only-text',action='store_true',help='Matched ablation: shared train-only text bank, no candidate-specific object text.')
    p.add_argument('--residual-head',default='C:/KhoaLuanTotNghiep/Code/output/clip_prior_residual_v1/residual_head.pt')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    old,oldviews,oldtext,oldlens=read_archive(args.v2_source,['train_pairs.json','val_pairs.json'])
    new,newviews,newtext,newlens=read_archive(args.actor_source,['train_pairs.json','val_pairs.json'])
    # Only the official training partitions are pooled. Every development row stays out.
    train=[];v=[];t=[];seen={};max_duplicate_feature_difference=0.
    for rows,views,text,count in ((old,oldviews,oldtext,oldlens[0]),(new,newviews,newtext,newlens[0])):
        for i,row in enumerate(rows[:count]):
            key=(row['image_id'],row['subject_id'],row['object_id'])
            if key in seen:
                previous=seen[key]
                if any(row[k]!=train[previous][k] for k in ('labels','subject_box','object_box','filename','object')):
                    raise ValueError('Duplicate training supervision disagrees.')
                difference=max(float((v[previous]-views[i]).abs().max()),float((t[previous]-text[i]).abs().max()))
                max_duplicate_feature_difference=max(max_duplicate_feature_difference,difference)
                if difference>1e-4:raise ValueError('Duplicate frozen features differ beyond numerical tolerance.')
                continue
            seen[key]=len(train);train.append(row);v.append(views[i]);t.append(text[i])
    dev=new[newlens[0]:];dv=newviews[newlens[0]:];dt=newtext[newlens[0]:]
    if {r['image_id'] for r in train}&{r['image_id'] for r in dev}:raise ValueError('Train/dev leakage.')
    if any(r['caption_split'] not in ('train','restval') for r in train+dev):raise ValueError('Caption heldout included.')
    views=torch.stack(v);text=torch.stack(t);geometry=pair_geometry(train)
    relation_bank=torch.nn.functional.normalize(text.mean(0),dim=-1)
    if args.relation_only_text:
        text=relation_bank[None].expand(len(train),-1,-1)
        dt=relation_bank[None].expand(len(dev),-1,-1)
    labels=torch.tensor([r['labels'] for r in train])
    edges=contrast_edges(train,max_per_positive=len(train))
    edge_tensor=torch.tensor([e[:4] for e in edges],dtype=torch.long).reshape(-1,4)
    kinds=torch.tensor([int(e[-1]=='actor') for e in edges])
    model=TripletVerifier();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[]
    for epoch in range(30):
        optimizer.zero_grad();logits=model(views,geometry,text)
        loss,bce,ranking=supervised_alignment_loss(logits,labels,edge_tensor,temperature=1.,margin=1.,contrast_weight=.2,edge_kinds=kinds)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        history.append({'epoch':epoch+1,'loss':float(loss.detach()),'bce':float(bce.detach()),'ranking':float(ranking.detach())})
    model.eval().requires_grad_(False)
    with torch.no_grad():
        train_score=model(views,geometry,text).sigmoid();dev_logits=model(dv,pair_geometry(dev),dt);dev_score=dev_logits.sigmoid()
    thresholds=fit_thresholds(train_score,labels)
    candidates,cv,ct,_=read_archive(args.candidate_source,['evaluation_pairs.json'])
    if args.relation_only_text:ct=relation_bank[None].expand(len(candidates),-1,-1)
    frozen_summary=json.loads((Path(args.scored)/'frozen_audit_summary.json').read_text())
    if sha(args.candidate_source)!=frozen_summary['source_sha256']:raise ValueError('Different full100 archive.')
    saved=json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    if len(saved)!=len(candidates) or any(any(s[k]!=r[k] for k in r) for s,r in zip(saved,candidates)):
        raise ValueError('Candidate order differs.')
    if {r['image_id'] for r in candidates}&{r['image_id'] for r in train}:raise ValueError('Candidate/train leakage.')
    with torch.no_grad():candidate_scores=model(cv,pair_geometry(candidates),ct).sigmoid()
    # Retain the frozen R50 proposer. New verifier cannot create additional raw proposals.
    if sha(args.residual_head)!=json.loads(Path('audit/actor_soft_fusion_full100_protocol.json').read_text())['residual_head_sha256']:
        raise ValueError('Frozen proposer checkpoint changed.')
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    modified=[]
    for i,row in enumerate(candidates):
        scores=candidate_scores[i].tolist()
        for c in range(6):
            if saved[i]['baseline_scores'][c]<rc['thresholds'][c]:scores[c]=-2.
        modified.append({**row,'scores':scores})
    old_prompts=json.loads((Path(args.scored)/'baseline_prompt_cache.json').read_text())
    inventory={f:[{'label':o['label'],'bbox':o['bbox'],'confidence':o['confidence']} for o in e['objects']] for f,e in old_prompts.items()}
    prompts=build_selected_prompts(modified,thresholds,object_inventory=inventory)
    summary={'variant':'relation_only_triplet_ablation_v1' if args.relation_only_text else 'text_conditioned_whole_triplet_development_v1','seed':42,'epochs':30,
        'train_pairs':len(train),'train_images':len({r['image_id'] for r in train}),'dev_pairs':len(dev),
        'training_labels':dict(Counter(str(int(v)) for v in labels.flatten())),
        'contrast_edges':dict(Counter(e[-1] for e in edges)),'parameters':sum(p.numel() for p in model.parameters()),
        'max_duplicate_feature_difference':max_duplicate_feature_difference,
        'thresholds':dict(zip(RELATIONS,thresholds)),'history':history,
        'dev_metrics':metrics(dev_score,torch.tensor([r['labels'] for r in dev]),thresholds),
        'dev_actor_edges':actor_edge_metrics(dev_logits,dev,exhaustive=True),
        'candidate_selected_claims':sum(len(e['relations']) for e in prompts.values()),
        'candidate_modes':dict(Counter(e['mode'] for e in prompts.values())),
        'sources':{'v2':sha(args.v2_source),'actor':sha(args.actor_source),'full100':sha(args.candidate_source)},
        'notes':'Sequential development trial; full100 and dev already inspected, not untouched heldout. '
                'Uses only known whole-triplet labels:0 means actor action absent, not object affordance impossible. '
                'Unknown missing roles remain masked. All confirmed actor and predicate contrasts used. '
                'Frozen A/B and predicate-object text embeddings; union excluded from verifier. No object whitelist or hand labels from audit used for training. '
                'Train-only thresholds; sigmoid not calibrated probability; fixed30ep/lr.001/margin1 logits. No caption claim.'}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'thresholds':thresholds,'relations':RELATIONS,
                'relation_only_text':args.relation_only_text,'relation_bank':relation_bank},output/'triplet_head.pt')
    for name,value in [('triplet_summary.json',summary),('triplet_prompt_cache.json',prompts),
                       ('candidate_predictions.json',[{**row,'triplet_activation':candidate_scores[i].tolist()} for i,row in enumerate(candidates)])]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2),flush=True)
    print('SELECTED',json.dumps({f:e['relations'] for f,e in prompts.items() if e['relations']},indent=2),flush=True)


if __name__=='__main__':main()
