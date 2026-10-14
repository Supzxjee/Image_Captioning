"""Train-only small fusion of frozen triplet/R50 evidence; no hard proposer gate."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F
from captioning.triplet_verifier import TripletVerifier
from captioning.instance_contrast_alignment import ResidualInstanceAlignment, supervised_alignment_loss
from captioning.actor_compatibility import pair_geometry, actor_edge_metrics
from captioning.supervised_relations import RELATIONS, contrast_edges
from captioning.selected_relation_prompts import build_selected_prompts
from run_actor_compatibility_cached import read_archive
from run_instance_alignment_diagnostic import fit_thresholds, metrics
from evaluate_frozen_residual import sha


def read_matched_transfer(path):
    # R55 predates the full-candidate archive layout; preserve its own provenance.
    with zipfile.ZipFile(path) as archive:
        def read(name):
            files=[f for f in archive.namelist() if f.endswith(name)]
            if len(files)!=1:raise ValueError('Ambiguous or missing archive entry: '+name)
            return archive.read(files[0])
        raw=read('data/evaluation_pairs.json');rows=json.loads(raw)
        metadata=json.loads(read('data/detector_pair_summary.json'))
        if hashlib.sha256(raw).hexdigest()!=metadata['pair_json_sha256']:raise ValueError('Transfer pair hash changed.')
        configuration=json.loads(read('manifest.json'))['configuration']
        if configuration['training'] is not False or configuration['model_revision']!='57c216476eefef5ab752ec549e440a49ae4ae5f3':
            raise ValueError('Transfer feature encoder changed.')
        records=sum([torch.load(io.BytesIO(read(f'features_{i}.pt')),map_location='cpu',weights_only=True) for i in range(2)],[])
    records.sort(key=lambda r:r['index'])
    if [r['index'] for r in records]!=list(range(len(rows))):raise ValueError('Transfer feature order invalid.')
    views=torch.stack([r['views'] for r in records]);text=torch.stack([r['text'] for r in records])
    for value in (views,text):
        if not torch.isfinite(value).all() or not torch.allclose(value.norm(dim=-1),torch.ones_like(value[...,0]),atol=1e-3):
            raise ValueError('Transfer features nonfinite or unnormalized.')
    return rows,views,text


def acceptance(scores, rows, thresholds):
    labels=torch.tensor([r['labels'] for r in rows]);mask=scores>=torch.tensor(thresholds)
    tp=int((mask&(labels==1)).sum());fp=int((mask&(labels==0)).sum())
    return {'positive':tp,'negative':fp,'unknown':int((mask&(labels<0)).sum()),
            'precision_on_known':tp/(tp+fp) if tp+fp else None,
            'recall_on_known':tp/int((labels==1).sum())}


def main():
    parser=argparse.ArgumentParser()
    for name in ('train-source','transfer-source','candidate-source','triplet-directory',
                 'residual-head','reference-directory','scored','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    protocol=json.loads(Path('audit/triplet_soft_prior_protocol.json').read_text())
    for path,key in ((args.train_source,'detector_source_sha256'),(args.transfer_source,'transfer_source_sha256'),
                     (args.candidate_source,'full100_source_sha256'),(Path(args.triplet_directory)/'triplet_head.pt','triplet_head_sha256'),
                     (args.residual_head,'residual_head_sha256')):
        if sha(path)!=protocol[key]:raise ValueError('Frozen protocol source/head mismatch: '+key)
    rows,views,text,lengths=read_archive(args.train_source,['train_pairs.json','val_pairs.json']);n=lengths[0]
    transfer,tv,tt=read_matched_transfer(args.transfer_source)
    candidates,cv,ct,_=read_archive(args.candidate_source,['evaluation_pairs.json'])
    if n!=protocol['train_pairs'] or len(rows)-n!=protocol['dev_pairs']:raise ValueError('Training/dev count changed.')
    train_ids={r['image_id'] for r in rows[:n]}
    all_gt_train={r['image_id'] for r in json.loads(Path('audit/triplet_detector_train_reference_pairs.json').read_text())}
    for evaluation in (rows[n:],transfer,candidates):
        if train_ids&{r['image_id'] for r in evaluation} or all_gt_train&{r['image_id'] for r in evaluation}:
            raise ValueError('Frozen head or fusion train/evaluation image overlap.')
    cp=torch.load(Path(args.triplet_directory)/'triplet_head.pt',map_location='cpu',weights_only=True)
    triplet=TripletVerifier();triplet.load_state_dict(cp['model']);triplet.eval().requires_grad_(False)
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    residual=ResidualInstanceAlignment(scale=rc['scale']);residual.load_state_dict(rc['model']);residual.eval().requires_grad_(False)
    def evidence(items,v,t):
        with torch.no_grad():
            logits=triplet(v,pair_geometry(items),t);base,_,_=residual(v,t)
        return torch.stack([logits,base],dim=-1)
    features=evidence(rows,views,text);transfer_features=evidence(transfer,tv,tt);candidate_features=evidence(candidates,cv,ct)
    reference=json.loads((Path(args.reference_directory)/'predictions.json').read_text())
    saved=json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    for items,old,feature,field in ((transfer,reference,transfer_features,'scores'),(candidates,saved,candidate_features,'baseline_scores')):
        if len(items)!=len(old) or any(any(row[k]!=r[k] for k in row) for row,r in zip(items,old)):
            raise ValueError('Evaluation candidate identity/order changed.')
        if not torch.allclose(feature[:,:,1],torch.tensor([r[field] for r in old]),atol=1e-5,rtol=1e-5):
            raise ValueError('Recomputed frozen R50 scores differ.')
    original_predictions=json.loads((Path(args.triplet_directory)/'candidate_predictions.json').read_text())
    if len(original_predictions)!=len(candidates) or not torch.allclose(candidate_features[:,:,0].sigmoid(),
          torch.tensor([r['triplet_activation'] for r in original_predictions]),atol=1e-5,rtol=1e-5):
        raise ValueError('Frozen triplet scores changed.')
    labels=torch.tensor([r['labels'] for r in rows[:n]]);known=labels>=0
    mean=torch.stack([features[:n,c][known[:,c]].mean(0) for c in range(6)])
    std=torch.stack([features[:n,c][known[:,c]].std(0,unbiased=False).clamp_min(.001) for c in range(6)])
    edges=contrast_edges(rows[:n],max_per_positive=n)
    edge_tensor=torch.tensor([e[:4] for e in edges],dtype=torch.long).reshape(-1,4)
    kinds=torch.tensor([int(e[-1]=='actor') for e in edges])
    inventory={f:[{'label':o['label'],'bbox':o['bbox'],'confidence':o['confidence']} for o in e['objects']]
               for f,e in json.loads((Path(args.scored)/'baseline_prompt_cache.json').read_text()).items()}
    old_review=json.loads((Path(args.scored)/'assistant_visual_review.json').read_text())['claims']
    spotcheck=json.loads(Path('audit/triplet_gate_bottleneck_result.json').read_text())['new_assistant_spotcheck']['claims']
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);runs={}
    for condition in protocol['conditions']:
        normalized=[(x-mean)/std for x in (features,transfer_features,candidate_features)]
        if condition=='triplet_only':
            for value in normalized:value[:,:,1]=0.
        raw_weights=torch.nn.Parameter(torch.zeros(6,2));bias=torch.nn.Parameter(torch.zeros(6))
        optimizer=torch.optim.Adam([raw_weights,bias],lr=protocol['learning_rate']);history=[]
        for epoch in range(protocol['epochs']):
            optimizer.zero_grad();weights=F.softplus(raw_weights)
            logits=(normalized[0][:n]*weights).sum(-1)+bias
            loss,bce,ranking=supervised_alignment_loss(logits,labels,edge_tensor,temperature=1.,margin=1.,
                contrast_weight=protocol['contrast_weight'],edge_kinds=kinds)
            loss=loss+protocol['ridge']*(weights.square().mean()+bias.square().mean())
            if not torch.isfinite(loss):raise ValueError('Nonfinite fusion loss.')
            loss.backward();optimizer.step()
            history.append({'epoch':epoch+1,'loss':float(loss.detach()),'bce':float(bce.detach()),'ranking':float(ranking.detach())})
        with torch.no_grad():
            weights=F.softplus(raw_weights).detach();offset=bias.detach()
            logits=[(x*weights).sum(-1)+offset for x in normalized];scores=[x.sigmoid() for x in logits]
        thresholds=fit_thresholds(scores[0][:n],labels)
        prompts=build_selected_prompts([{**r,'scores':scores[2][i].tolist()} for i,r in enumerate(candidates)],thresholds,object_inventory=inventory)
        selected={(f,r['subject_instance'],r['relation'],r['object_instance']) for f,e in prompts.items() for r in e['relations']}
        review=[{**c,'selected_by_fusion':tuple(c[k] for k in ('filename','subject_instance','relation','object_instance')) in selected}
                for c in old_review+spotcheck]
        run={'condition':condition,'protocol':protocol,'weights':weights.tolist(),'bias':offset.tolist(),'thresholds':thresholds,
             'train_pairs':n,'train_images':len(train_ids),'dev_pairs':len(rows)-n,'train_evaluation_overlap':0,
             'nominal_parameters':18,'effective_parameters':12 if condition=='triplet_only' else 18,
             'dev_metrics':metrics(scores[0][n:],torch.tensor([r['labels'] for r in rows[n:]]),thresholds),
             'dev_acceptance':acceptance(scores[0][n:],rows[n:],thresholds),
             'dev_actor_edges':actor_edge_metrics(logits[0][n:],rows[n:],exhaustive=True),
             'transfer_acceptance':acceptance(scores[1],transfer,thresholds),
             'transfer_metrics':metrics(scores[1],torch.tensor([r['labels'] for r in transfer]),thresholds),
             'transfer_actor_edges':actor_edge_metrics(logits[1],transfer,exhaustive=True),
             'full100_selected_claims':len(selected),'full100_relation_images':sum(bool(e['relations']) for e in prompts.values()),
             'full100_pre_cap_claims':sum(e['above_threshold_claims'] for e in prompts.values()),
             'known_assistant_review_crosscheck':review,
             'notes':'No hard proposer veto. Train-only in-sample stacking/calibration;frozen source heads;unknown labels masked. '
                     'Known-label matched transfer precision is not fullpool precision;review samples are purposive/method-visible;no caption claim.'}
        directory=output/condition;directory.mkdir(exist_ok=True)
        torch.save({'weights':weights,'bias':offset,'mean':mean,'std':std,'thresholds':thresholds,
                    'condition':condition,'protocol':protocol},directory/'fusion_head.pt')
        run['fusion_head_sha256']=sha(directory/'fusion_head.pt')
        for name,value in (('fusion_summary.json',run),('history.json',history),('prompt_cache.json',prompts),
            ('transfer_predictions.json',[{**r,'fusion_scores':scores[1][i].tolist()} for i,r in enumerate(transfer)]),
            ('candidate_predictions.json',[{**r,'fusion_scores':scores[2][i].tolist()} for i,r in enumerate(candidates)])):
            (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        runs[condition]=run
        print(condition,json.dumps({k:v for k,v in run.items() if k in ('transfer_acceptance','dev_acceptance','full100_selected_claims','full100_relation_images','weights')}),flush=True)
        print('NEW CONTRADICTIONS',json.dumps([c for c in review if c.get('assistant_status')=='contradicted' and c['selected_by_fusion']],indent=2),flush=True)
    (output/'paired_fusion_summary.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')


if __name__=='__main__':main()
