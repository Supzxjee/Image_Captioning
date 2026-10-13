"""Train-only positive-weight evidence fusion; no hand-tuned child-image threshold."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from captioning.actor_compatibility import ActorCompatibility,actor_supervision,pair_geometry
from captioning.instance_contrast_alignment import ResidualInstanceAlignment
from captioning.selected_relation_prompts import build_selected_prompts
from run_actor_compatibility_cached import read_archive
from run_instance_alignment_diagnostic import fit_thresholds,metrics
from evaluate_frozen_residual import sha,newline_hashes


def edges_report(scores,rows):
    _,edges=actor_supervision(rows,exhaustive=True)
    i,r,j,c=edges.T
    gap=scores[i,r]-scores[j,c]
    return {'edges':len(edges),'wins':int((gap>0).sum()),'ties':int((gap==0).sum()),
            'win_rate':float((gap>0).float().mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-source',required=True);p.add_argument('--candidate-source',required=True)
    p.add_argument('--actor-directory',required=True);p.add_argument('--residual-head',required=True)
    p.add_argument('--scored',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    directory=Path(args.actor_directory)
    actor_summary=json.loads((directory/'actor_summary.json').read_text())
    if sha(args.train_source)!=actor_summary['train_source_sha256']:
        raise ValueError('Actor training source mismatch.')
    if sha(args.candidate_source)!=actor_summary['candidate_source_sha256'] or sha(args.residual_head)!=actor_summary['frozen_residual_checkpoint_sha256']:
        raise ValueError('Frozen source/head mismatch.')
    rows,views,text,lengths=read_archive(args.train_source,['train_pairs.json','val_pairs.json']);n=lengths[0]
    exclusions=set(json.loads(Path('audit/actor_verifier_dev_exclusions.json').read_text())['image_ids'])
    if {r['image_id'] for r in rows[n:]}&exclusions:
        raise ValueError('New dev overlaps exclusion set.')
    ac=torch.load(directory/'actor_head.pt',map_location='cpu',weights_only=True)
    if ac['architecture']!='actor_compatibility_v1':
        raise ValueError('Require nonlinear actor head.')
    actor=ActorCompatibility();actor.load_state_dict(ac['model']);actor.eval().requires_grad_(False)
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    residual=ResidualInstanceAlignment(scale=rc['scale']);residual.load_state_dict(rc['model']);residual.eval().requires_grad_(False)
    with torch.inference_mode():
        residual_scores,_,_=residual(views,text);actor_logits=actor(views,pair_geometry(rows),text)
    features=torch.stack([residual_scores,actor_logits],dim=-1).clone()
    labels,edges=actor_supervision(rows[:n],exhaustive=True);known=labels>=0
    mean=torch.stack([features[:n,c][known[:,c]].mean(0) for c in range(6)])
    std=torch.stack([features[:n,c][known[:,c]].std(0,unbiased=False).clamp_min(.001) for c in range(6)])
    normalized=(features-mean)/std
    raw_weights=torch.nn.Parameter(torch.zeros(6,2));bias=torch.nn.Parameter(torch.zeros(6))
    optimizer=torch.optim.Adam([raw_weights,bias],lr=.03)
    history=[]
    for epoch in range(100):
        optimizer.zero_grad()
        logits=(normalized[:n]*F.softplus(raw_weights)).sum(-1)+bias
        loss=F.binary_cross_entropy_with_logits(logits[known],labels[known].float())+.01*(F.softplus(raw_weights).square().mean()+bias.square().mean())
        loss.backward();optimizer.step();history.append(float(loss.detach()))
    with torch.no_grad():
        weights=F.softplus(raw_weights).detach();bias=bias.detach()
        fused=((normalized*weights).sum(-1)+bias).sigmoid()
    thresholds=fit_thresholds(fused[:n],labels)
    dev_labels,_=actor_supervision(rows[n:],exhaustive=True)
    candidate_rows,candidate_views,candidate_text,_=read_archive(args.candidate_source,['evaluation_pairs.json'])
    if {r['image_id'] for r in candidate_rows}&{r['image_id'] for r in rows[:n]}:
        raise ValueError('Candidate/train overlap.')
    with torch.no_grad():
        candidate_base,_,_=residual(candidate_views,candidate_text)
        candidate_actor=actor(candidate_views,pair_geometry(candidate_rows),candidate_text)
        candidate_features=torch.stack([candidate_base,candidate_actor],dim=-1)
        candidate_fused=((((candidate_features-mean)/std)*weights).sum(-1)+bias).sigmoid()
    saved=json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    if len(saved)!=len(candidate_rows) or any(any(s[k]!=r[k] for k in r) for s,r in zip(saved,candidate_rows)):
        raise ValueError('Candidate ordering changed.')
    if not torch.allclose(candidate_base,torch.tensor([r['scores'] for r in saved]),atol=1e-6):
        raise ValueError('Candidate frozen baseline differs.')
    modified=[]
    for i,row in enumerate(candidate_rows):
        # Keep the original relation proposer as a first-stage requirement.
        values=candidate_fused[i].tolist()
        for c in range(6):
            if candidate_base[i,c]<rc['thresholds'][c]:values[c]=-2.
        modified.append({**row,'scores':values})
    old=json.loads((Path(args.scored)/'selected_prompt_cache.json').read_text())
    inventory={f:[{'label':o['label'],'bbox':o['bbox'],'confidence':o['confidence']} for o in e['objects']] for f,e in old.items()}
    prompts=build_selected_prompts(modified,thresholds,object_inventory=inventory)
    before=residual_scores[n:]>=torch.tensor(rc['thresholds'])
    after=before&(fused[n:]>=torch.tensor(thresholds))
    def counts(mask):
        return {'positive':int((mask&(dev_labels==1)).sum()),'negative':int((mask&(dev_labels==0)).sum()),
                'unknown':int((mask&(dev_labels<0)).sum())}
    summary={'variant':'positive_weight_actor_soft_fusion_development_v1','epochs':100,'parameters':18,
        'train_source_sha256':sha(args.train_source),'actor_head_sha256':sha(directory/'actor_head.pt'),
        'residual_head_sha256':sha(args.residual_head),'weights':weights.tolist(),'bias':bias.tolist(),
        'thresholds':thresholds,'history':history,'actor_dev_edges':edges_report(actor_logits[n:],rows[n:]),
        'residual_dev_edges':edges_report(residual_scores[n:],rows[n:]),'fused_dev_edges':edges_report(fused[n:],rows[n:]),
        'fused_actor_endpoint_metrics':metrics(fused[n:],dev_labels,thresholds),
        'before_dev_accepts':counts(before),'after_dev_accepts':counts(after),
        'retained_dev_positives':int((before&after&(dev_labels==1)).sum()),
        'lost_dev_positives':int((before&~after&(dev_labels==1)).sum()),
        'removed_dev_negatives':int((before&~after&(dev_labels==0)).sum()),
        'candidate_selected_claims':sum(len(e['relations']) for e in prompts.values()),
        'notes':'Sequential development trial after examining hard-veto failure; not untouched heldout. '
                'Both heads frozen. Per-class two nonnegative weights and intercept, normalization/fit/thresholds train-only. '
                'Fixed100epochs/lr.03/ridge.01; no dev/child tuning. Fused sigmoid is not calibrated probability. '
                'Endpoint labels do not exhaustively evaluate all claims. Proposer threshold remains required.'}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for name,value in [('fusion_summary.json',summary),('soft_verified_prompt_cache.json',prompts),
                       ('candidate_fusion_predictions.json',[{**r,'actor_activation':candidate_actor[i].sigmoid().tolist(),
                                                             'fusion_activation':candidate_fused[i].tolist()} for i,r in enumerate(candidate_rows)])]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    torch.save({'weights':weights,'bias':bias,'mean':mean,'std':std,'thresholds':thresholds},output/'fusion_head.pt')
    print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2))
    print('SELECTED',json.dumps({f:e['relations'] for f,e in prompts.items() if e['relations']},indent=2))


if __name__=='__main__':main()
