"""Paired region-content vs repeated union-CLS development trial."""
import argparse,json,zipfile
from pathlib import Path
from collections import Counter
import torch
from captioning.interaction_regions import InteractionRegionVerifier
from captioning.actor_compatibility import pair_geometry,actor_edge_metrics
from captioning.supervised_relations import RELATIONS,contrast_edges
from captioning.instance_contrast_alignment import supervised_alignment_loss
from captioning.selected_relation_prompts import build_selected_prompts
from run_actor_compatibility_cached import read_archive
from run_triplet_soft_prior_cached import read_matched_transfer,acceptance
from run_instance_alignment_diagnostic import metrics,fit_thresholds
from evaluate_frozen_residual import sha
from validate_interaction_regions import validate
from import_target_annotations import filter_annotations


def verified_target_tensors(train, filtered):
    labels=torch.tensor([r['labels'] for r in train]);edges=[];seen=set()
    for item in filtered['eligible_contrasts']:
        i,j,c=item['positive_row_index'],item['alternate_row_index'],item['relation_index']
        if not (0<=i<len(train) and 0<=j<len(train) and 0<=c<6):raise ValueError('Invalid annotation row/class index.')
        a,b=train[i],train[j]
        if (a['image_id'],a['subject_id'])!=(b['image_id'],b['subject_id']) or a['object_id']==b['object_id']:
            raise ValueError('Target contrast must hold image/subject fixed andchange object.')
        if a['labels'][c]!=1 or b['labels'][c]!=-1:raise ValueError('Review mustconfirm positive andpreviously unknown alternative.')
        if (i,j,c) in seen:raise ValueError('Duplicate target contrast.')
        seen.add((i,j,c));labels[j,c]=0;edges.append((i,c,j,c))
    if not edges:raise ValueError('No eligible reviewed target contrasts.')
    return labels,torch.tensor(edges,dtype=torch.long)


def main():
    parser=argparse.ArgumentParser()
    for name in ('region-source','train-source','transfer-source','candidate-source','scored','reference-directory','residual-head','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--target-review',help='Returned human JSON; switches to paired region classification vs target-ranking trial.')
    parser.add_argument('--annotation-package',help='Original local manifest/private mapping directory.')
    args=parser.parse_args();torch.set_num_threads(2)
    protocol=json.loads(Path('audit/interaction_region_protocol.json').read_text())
    for source,split in ((args.train_source,'train'),(args.transfer_source,'transfer'),(args.candidate_source,'full100')):
        if sha(source)!=protocol['splits'][split]['source_sha256']:raise ValueError('Old feature source changed.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);extracted=output/'validated_source';extracted.mkdir(exist_ok=True)
    # Copy only named members; do not follow archive paths or embedded instructions.
    names=['manifest.json','extraction_protocol.json','extraction_summary.json','regions_0.pt','regions_1.pt']
    names+=['data/'+split+'_pairs.json' for split in ('train','val','transfer','full100')]
    with zipfile.ZipFile(args.region_source) as archive:
        for name in names:
            if name not in archive.namelist():raise ValueError('Missing archive entry: '+name)
            target=extracted/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    reported=json.loads((extracted/'extraction_summary.json').read_text())
    for name,wanted in reported['feature_files_sha256'].items():
        if name not in ('regions_0.pt','regions_1.pt') or sha(extracted/name)!=wanted:raise ValueError('Region feature hash changed.')
    verified=validate(extracted)
    if json.loads(json.dumps(verified))!=reported:raise ValueError('Extraction validation differs from uploaded report.')
    records=sum([torch.load(extracted/f'regions_{i}.pt',map_location='cpu',weights_only=True) for i in range(2)],[])
    records.sort(key=lambda r:r['index'])
    old,v,t,lengths=read_archive(args.train_source,['train_pairs.json','val_pairs.json']);n=lengths[0]
    transfer,_,transfer_text=read_matched_transfer(args.transfer_source)
    candidates,_,candidate_text,_=read_archive(args.candidate_source,['evaluation_pairs.json'])
    text_by_split={'train':t[:n],'val':t[n:],'transfer':transfer_text,'full100':candidate_text}
    old_by_split={'train':old[:n],'val':old[n:],'transfer':transfer,'full100':candidates}
    cohorts={}
    for split in ('train','val','transfer','full100'):
        rows=json.loads((extracted/'data'/f'{split}_pairs.json').read_text())
        if rows!=old_by_split[split]:raise ValueError('New/old pair identities, geometry or labels differ.')
        selected=[r for r in records if r['split']==split]
        cohorts[split]=(rows,torch.stack([r['regions'] for r in selected]),torch.stack([r['valid'] for r in selected]),
                        torch.stack([r['mass'] for r in selected]),pair_geometry(rows),text_by_split[split])
    del records,old,v,t
    train=cohorts['train'][0];labels=torch.tensor([r['labels'] for r in train])
    edges=contrast_edges(train,max_per_positive=len(train));ee=torch.tensor([e[:4] for e in edges],dtype=torch.long).reshape(-1,4)
    ek=torch.tensor([int(e[-1]=='actor') for e in edges])
    target_edges=None;annotation_settings=None
    if args.target_review:
        if not args.annotation_package:raise ValueError('Original annotation package required.')
        package=Path(args.annotation_package);mapping=json.loads((package/'private_mapping.json').read_text(encoding='utf-8'))
        if sha(package/'cases.json')!=mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json')!=mapping['source_pairs_sha256']:
            raise ValueError('Annotation source/manifest changed.')
        filtered=filter_annotations(json.loads(Path(args.target_review).read_text(encoding='utf-8-sig')),mapping)
        labels,target_edges=verified_target_tensors(train,filtered)
        (output/'private_filtered_annotations.json').write_text(json.dumps(filtered,indent=2,ensure_ascii=False),encoding='utf-8')
        annotation_settings={'review_sha256':sha(args.target_review),'manifest_sha256':mapping['manifest_sha256'],
            'filter_counts':filtered['counts'],'target_edges':len(target_edges),'target_loss_weight':.2,
            'notes':'Bothregion conditions use same newly confirmed negative BCE labels andsame original actor/predicate edges. '
                    'Only target_rank adds .2 mean target hinge margin1. One selfreported human reviewer;source conflicts excluded,notresolved. '
                    'No source labels overwritten;remainingunknown masked;reviewed train images only.'}
    gate_protocol=json.loads(Path('audit/actor_soft_fusion_full100_protocol.json').read_text())
    if sha(args.residual_head)!=gate_protocol['residual_head_sha256']:raise ValueError('Frozen R50 head changed.')
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    saved=json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    reference=json.loads((Path(args.reference_directory)/'predictions.json').read_text())
    for rows,predicted in ((candidates,saved),(transfer,reference)):
        if len(rows)!=len(predicted) or any(any(row[k]!=p[k] for k in row) for row,p in zip(rows,predicted)):
            raise ValueError('Frozen reference order/identity changed.')
    inventory={f:[{'label':o['label'],'bbox':o['bbox'],'confidence':o['confidence']} for o in e['objects']]
               for f,e in json.loads((Path(args.scored)/'baseline_prompt_cache.json').read_text()).items()}
    reviewed=json.loads((Path(args.scored)/'assistant_visual_review.json').read_text())['claims']
    reviewed+=json.loads(Path('audit/triplet_gate_bottleneck_result.json').read_text())['new_assistant_spotcheck']['claims']
    runs={};settings={'seed':42,'epochs':30,'lr':.001,'weight_decay':.01,'contrast_weight':.2,
        'margin_logit':1,'temperature':1,'gradient_clip_norm':1,'threshold_rule':'train-only >=90%known precision and>=5accepts',
        'region_source_sha256':sha(args.region_source),'extraction_protocol':protocol,
        'notes':'Same initialization/train/text/masks/mass/geometry/parameter dimensions. Control repeatsunionCLS invalid slots. '
                'Primary acceptance hasno R50gate;secondaryconjunction frozen. Unknownmasked,no auditlabels. Reuseddevelopment;notcaptionquality.'}
    if annotation_settings:
        settings['annotation']=annotation_settings
        settings['notes']='Both ordered-region heads use identical new negative BCE labels andoriginal actor/predicate contrasts. Only target_rank adds target hinge. Sameinit/schedule/thresholdrule;noR50gate primary. Single-reviewer trainannotations;reuseddevelopment,notcaptionquality.'
    conditions=('classification_only','target_rank') if annotation_settings else ('union_only','regions')
    for condition in conditions:
        torch.manual_seed(settings['seed']);model=InteractionRegionVerifier()
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay']);history=[]
        def predict(split):return model(*cohorts[split][1:],union_only=condition=='union_only')
        for epoch in range(settings['epochs']):
            optimizer.zero_grad();logits=predict('train')
            loss,bce,ranking=supervised_alignment_loss(logits,labels,ee,temperature=1.,margin=1.,contrast_weight=.2,edge_kinds=ek)
            target_loss=logits.sum()*0
            if condition=='target_rank':
                i,r,j,c=target_edges.T
                target_loss=torch.relu(1.-logits[i,r]+logits[j,c]).mean()
                loss=loss+.2*target_loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            history.append({'epoch':epoch+1,'loss':float(loss.detach()),'bce':float(bce.detach()),'ranking':float(ranking.detach()),'target_ranking':float(target_loss.detach())})
        model.eval().requires_grad_(False)
        with torch.no_grad():logits={split:predict(split) for split in cohorts};scores={split:value.sigmoid() for split,value in logits.items()}
        thresholds=fit_thresholds(scores['train'],labels)
        transfer_gate=torch.tensor([r['scores'] for r in reference])>=torch.tensor(rc['thresholds'])
        candidate_gate=torch.tensor([r['baseline_scores'] for r in saved])>=torch.tensor(rc['thresholds'])
        gated_transfer=scores['transfer'].clone();gated_transfer[~transfer_gate]=-2.
        gated_candidate=scores['full100'].clone();gated_candidate[~candidate_gate]=-2.
        prompts=build_selected_prompts([{**r,'scores':scores['full100'][i].tolist()} for i,r in enumerate(candidates)],thresholds,object_inventory=inventory)
        gated_prompts=build_selected_prompts([{**r,'scores':gated_candidate[i].tolist()} for i,r in enumerate(candidates)],thresholds,object_inventory=inventory)
        selected={(f,r['subject_instance'],r['relation'],r['object_instance']) for f,e in prompts.items() for r in e['relations']}
        gated_selected={(f,r['subject_instance'],r['relation'],r['object_instance']) for f,e in gated_prompts.items() for r in e['relations']}
        review=[{**r,'selected_by_new_head':tuple(r[k] for k in ('filename','subject_instance','relation','object_instance')) in selected,
                 'selected_with_R50_gate':tuple(r[k] for k in ('filename','subject_instance','relation','object_instance')) in gated_selected} for r in reviewed]
        result={'condition':condition,'settings':settings,'parameters':sum(p.numel() for p in model.parameters()),
            'training_labels':dict(Counter(str(int(v)) for v in labels.flatten())),'contrast_edges':dict(Counter(e[-1] for e in edges)),
            'dev_metrics':metrics(scores['val'],torch.tensor([r['labels'] for r in cohorts['val'][0]]),thresholds),
            'dev_acceptance':acceptance(scores['val'],cohorts['val'][0],thresholds),'dev_actor_edges':actor_edge_metrics(logits['val'],cohorts['val'][0],True),
            'transfer_metrics':metrics(scores['transfer'],torch.tensor([r['labels'] for r in transfer]),thresholds),
            'transfer_acceptance':acceptance(scores['transfer'],transfer,thresholds),
            'transfer_actor_edges':actor_edge_metrics(logits['transfer'],transfer,True),
            'transfer_with_frozen_R50_gate':acceptance(gated_transfer,transfer,thresholds),
            'thresholds':thresholds,'full100_claims':len(selected),'full100_relation_images':sum(bool(e['relations']) for e in prompts.values()),
            'full100_gated_claims':sum(len(e['relations']) for e in gated_prompts.values()),'known_assistant_review_crosscheck':review}
        if annotation_settings:
            i,r,j,c=target_edges.T;gap=logits['train'][i,r]-logits['train'][j,c]
            result['reviewed_train_target_edges']={'edges':len(gap),'wins':int((gap>0).sum()),'mean_gap':float(gap.mean()),
                'notes':'Training fit only,not target accuracy on heldout.'}
        directory=output/condition;directory.mkdir(exist_ok=True)
        torch.save({'model':model.state_dict(),'thresholds':thresholds,'union_only':condition=='union_only','settings':settings},directory/'region_head.pt')
        result['head_sha256']=sha(directory/'region_head.pt')
        for name,value in (('region_summary.json',result),('history.json',history),('prompt_cache.json',prompts),('gated_prompt_cache.json',gated_prompts),
            ('transfer_predictions.json',[{**r,'scores':scores['transfer'][i].tolist()} for i,r in enumerate(transfer)]),
            ('candidate_predictions.json',[{**r,'scores':scores['full100'][i].tolist()} for i,r in enumerate(candidates)])):
            (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        runs[condition]=result
        print(condition,json.dumps({k:v for k,v in result.items() if k in ('dev_acceptance','transfer_acceptance','transfer_with_frozen_R50_gate','full100_claims','full100_relation_images')}),flush=True)
    (output/'paired_region_summary.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')


if __name__=='__main__':main()
