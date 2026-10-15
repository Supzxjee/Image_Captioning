"""Train small matched patch readers; keep all review-manifest images out of fit/cal."""
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from captioning.pair_query_attention import PairQueryAttention, patch_evidence
from captioning.actor_compatibility import pair_geometry
from captioning.instance_contrast_alignment import supervised_alignment_loss
from captioning.supervised_relations import RELATIONS, contrast_edges
from captioning.reviewed_targets import target_metrics, target_changes
from run_actor_compatibility_cached import read_archive
from run_instance_alignment_diagnostic import fit_thresholds, metrics, edge_metrics
from run_triplet_soft_prior_cached import acceptance
from import_target_annotations import filter_annotations
from extract_pair_query_patches import sha
from captioning.factorized_evidence import reviewed_labels, target_objective
from captioning.evidence_routing import selective_decisions
from run_factorized_evidence_cv import shared_target_policy


def partition(rows, train_length, excluded_filenames):
    candidates = sorted({r['image_id'] for r in rows[:train_length] if r['filename'] not in excluded_filenames})
    random.Random(1337).shuffle(candidates)
    calibration = set(candidates[:max(1, round(.2*len(candidates)))])
    fit = set(candidates)-calibration
    dev = {r['image_id'] for r in rows[train_length:]}
    review = {r['image_id'] for r in rows if r['filename'] in excluded_filenames}
    if fit & calibration or (fit|calibration) & (dev|review): raise ValueError('Image leakage.')
    return {'fit_images':sorted(fit), 'calibration_images':sorted(calibration), 'development_images':sorted(dev),
            'excluded_manifest_images_present':sorted(review)}


def run(args):
    torch.set_num_threads(2)
    root, package, output = map(Path, (args.patches, args.package, args.output))
    output.mkdir(parents=True, exist_ok=True)
    extraction = json.loads((root/'extraction_summary.json').read_text())
    if sha(root/'extraction_protocol.json') != extraction['protocol_sha256'] or sha(root/'image_provenance.json') != extraction['image_manifest_sha256']:
        raise ValueError('Patch extraction provenance differs.')
    ep = json.loads((root/'extraction_protocol.json').read_text())
    for split, digest in ep['source_pair_sha256'].items():
        if sha(Path('audit')/f'interaction_region_{split}_pairs.json') != digest: raise ValueError('Pair source changed.')
    source_protocol = json.loads(Path('audit/interaction_region_protocol.json').read_text())
    if sha(args.source) != source_protocol['splits']['train']['source_sha256']: raise ValueError('Old frozen feature source changed.')
    rows, views, text, lengths = read_archive(args.source, ['train_pairs.json','val_pairs.json'])
    if rows != sum([json.loads((Path('audit')/f'interaction_region_{s}_pairs.json').read_text()) for s in ('train','val')],[]):
        raise ValueError('Patch/text/labels must have identical ordering.')
    provenance = json.loads((package/'image_provenance.json').read_text())
    if sha(package/'image_provenance.json') != ep['review_provenance_sha256']: raise ValueError('Review image inventory changed.')
    mapping = json.loads((package/'private_mapping.json').read_text())
    if sha(package/'cases.json') != mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json') != mapping['source_pairs_sha256']:
        raise ValueError('Review manifest/source changed.')
    filtered = filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')), mapping)
    cases = filtered['eligible_contrasts']
    partitions = partition(rows, lengths[0], {r['filename'] for r in provenance})
    fit = [i for i,r in enumerate(rows) if r['image_id'] in set(partitions['fit_images'])]
    cal = [i for i,r in enumerate(rows) if r['image_id'] in set(partitions['calibration_images'])]
    dev = list(range(lengths[0], len(rows)))
    fit_rows, cal_rows, dev_rows = ([rows[i] for i in indices] for indices in (fit,cal,dev))
    labels = torch.tensor([r['labels'] for r in rows])
    edges = contrast_edges(fit_rows, max_per_positive=len(fit_rows))
    ee = torch.tensor([e[:4] for e in edges], dtype=torch.long).reshape(-1,4)
    ek = torch.tensor([int(e[-1]=='actor') for e in edges])
    generator = torch.Generator().manual_seed(1729)
    projection = torch.linalg.qr(torch.randn(768,64,generator=generator), mode='reduced').Q
    reference = Path(args.reference)
    fold_source = json.loads((reference/'protocol.json').read_text())
    fold_summary = json.loads((reference/'factorized_evidence_summary.json').read_text())
    if sha(reference/'protocol.json') != fold_summary['protocol_sha256'] or sha(args.review) != fold_source['review_sha256'] or sha(args.source) != fold_source['source_sha256']:
        raise ValueError('Frozen R101 folds/source changed.')
    protocol = {'variant':'pair_query_attention_v1','extraction_protocol_sha256':extraction['protocol_sha256'],
        'source_sha256':sha(args.source),'review_sha256':sha(args.review),'review_manifest_sha256':mapping['manifest_sha256'],
        'partition':partitions,'fit_pairs':len(fit),'calibration_pairs':len(cal),'development_pairs':len(dev),
        'review_cases':len(cases),'review_filter_counts':filtered['counts'],
        'seeds':[42], 'epochs':30, 'optimizer':'AdamW', 'learning_rate':.001, 'weight_decay':.01,
        'hidden':32,'compression':'L2 normalize; fixed QR Gaussian 768x64 seed1729; L2 normalize',
        'patch_inputs':'64D content +5 ordered region supports +XY; subject/object weighted patch means +10D box geometry;512D candidate text',
        'conditions':['clip_prior','mean_pool','query_blind','attention'],
        'loss':'known-label BCE +0.2 type-balanced actor/predicate hinge margin1; unknowns masked',
        'thresholds':'separate source-training image calibration only; >=90%known precision,>=5known accepts',
        'target_adaptation':{'folds':fold_source['folds'],'reference_protocol_sha256':sha(reference/'protocol.json'),
            'epochs':30,'lr':.0003,'weight_decay':.01,'ranking_weight':.2,
            'loss':'equal-image known reviewed BCE +0.2 logistic target contrast; only inner fit images',
            'calibration':'R101 inner calibration cases only; shared90%/5-known tails; no actor gate'},
        'notes':'Source stage uses no human labels for training/calibration; all61 manifest images excluded. Separate target adaptation uses inner-fold human labels only. Existing dev and human cases repeatedly inspected; not independent. Same instantiated head; mean_pool has inactive key parameters. Query-blind control trains same modules but selects patches with a zero pair/text query.'}
    pp = output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol: raise ValueError('Changed training protocol; use fresh output.')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Verify and compress in bounded shards; avoid a 197x768 full float32 copy.
    records = []; descriptors = []; endpoints = []; indices = []
    model_path=Path(args.model)
    if sha(model_path/'pytorch_model.bin') != ep['weights_sha256']: raise ValueError('CLIP weights changed.')
    state=torch.load(model_path/'pytorch_model.bin',map_location='cpu',weights_only=True)
    visual_projection=state['visual_projection.weight'].float().clone(); del state
    cls_similarities=[]
    for name, digest in sorted(extraction['feature_sha256'].items()):
        if Path(name).name != name or sha(root/name) != digest: raise ValueError('Patch shard changed.')
        records = torch.load(root/name, weights_only=True, map_location='cpu')
        tokens = torch.stack([r['tokens'] for r in records])
        support = torch.stack([r['region_support'] for r in records])
        shard_indices=[r['index'] for r in records]
        projected_cls=F.normalize(tokens[:,0].float() @ visual_projection.T,dim=-1)
        cls_similarities.extend((projected_cls*views[shard_indices,2]).sum(-1).tolist())
        x, end = patch_evidence(tokens, support, projection)
        descriptors.append(x); endpoints.append(end); indices.extend(r['index'] for r in records)
    if indices != list(range(len(rows))): raise ValueError('Missing or duplicate patch rows.')
    if min(cls_similarities)<.999: raise ValueError('New patches do not reproduce the original union CLIP features.')
    patches, endpoints = torch.cat(descriptors), torch.cat(endpoints)
    del descriptors, records, tokens, support
    geometry = pair_geometry(rows)
    torch.save({'projection':projection,'patches':patches,'endpoints':endpoints,'geometry':geometry,'text':text}, output/'private_compressed_inputs.pt')
    results, details, histories, saved_scores, changes = {}, {}, {}, {}, {}
    def evaluate(name, logits):
        probabilities = logits.sigmoid()
        thresholds = fit_thresholds(probabilities[cal], labels[cal])
        decisions = (probabilities >= torch.tensor(thresholds)).long()
        target, detail = target_metrics(logits[:lengths[0]], decisions[:lengths[0]], rows[:lengths[0]], cases)
        result = {'parameters':0 if name=='clip_prior' else sum(p.numel() for p in model.parameters()),
            'thresholds':thresholds,'development_metrics':metrics(probabilities[dev],labels[dev],thresholds),
            'development_acceptance':acceptance(probabilities[dev],dev_rows,thresholds),
            'development_contrasts':edge_metrics(logits[dev],dev_rows), 'reviewed_targets':target}
        results[name], details[name], saved_scores[name] = result, detail, logits
        return thresholds
    evaluate('clip_prior',10*torch.einsum('bd,bkd->bk',views[:,2],text)-3)
    for mode in ('mean_pool','query_blind','attention'):
        torch.manual_seed(42); model = PairQueryAttention()
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        history=[]; started=time.monotonic()
        for epoch in range(30):
            optimizer.zero_grad()
            logits = model(patches[fit],endpoints[fit],geometry[fit],text[fit],mode=mode)
            loss,bce,rank = supervised_alignment_loss(logits,labels[fit],ee,temperature=1,margin=1,contrast_weight=.2,edge_kinds=ek)
            if not torch.isfinite(loss): raise ValueError('Nonfinite patch-reader loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            history.append({'epoch':epoch+1,'loss':float(loss.detach()),'bce':float(bce.detach()),'ranking':float(rank.detach())})
            if (epoch+1)%10==0: print(f'{mode} epoch {epoch+1}/30 loss {history[-1]["loss"]:.4f} | {time.monotonic()-started:.1f}s',flush=True)
        model.eval().requires_grad_(False)
        with torch.no_grad():
            logits = torch.cat([model(patches[i:i+100],endpoints[i:i+100],geometry[i:i+100],text[i:i+100],mode=mode) for i in range(0,len(rows),100)])
        thresholds = evaluate(mode, logits); histories[mode]=history
        with torch.no_grad():
            _,weights=model(patches[dev[:50]],endpoints[dev[:50]],geometry[dev[:50]],text[dev[:50]],mode=mode,return_weights=True)
        results[mode]['reader_diagnostic_first50_dev']={
            'mean_entropy':float(-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean()),
            'mean_maximum_weight':float(weights.amax(-1).mean()),
            'mean_predicate_weight_l1':float((weights[:,1:]-weights[:,:1]).abs().sum(-1).mean()),
            'note':'Internal weights, not causal explanations or contact labels.'}
        torch.save({'model':model.state_dict(),'mode':mode,'thresholds':thresholds,'protocol_sha256':sha(pp)},output/f'{mode}_seed42.pt')
        counts, changed = target_changes(details['clip_prior'], details[mode])
        results[mode]['paired_target_changes_vs_clip']=counts; changes[mode]=changed
    counts, changed = target_changes(details['mean_pool'],details['attention'])
    results['attention']['paired_target_changes_vs_mean_pool']=counts; changes['attention_vs_mean_pool']=changed
    summary = {'variant':protocol['variant'],'protocol_sha256':sha(pp),'partition_counts':{k:len(v) for k,v in partitions.items()},
        'fit_pairs':len(fit),'calibration_pairs':len(cal), 'minimum_original_union_cls_cosine':min(cls_similarities),
        'fit_labels':dict(Counter(map(str,labels[fit].flatten().tolist()))),
        'fit_contrast_types':dict(Counter(e[-1] for e in edges)), 'review_cases':len(cases),
        'results':results,'production_promotion':False,'caption_decoder_changed':False,
        'limitations':['Single seed','Repeatedly inspected development data','No fresh independent evaluation','Coarse contextualized14x14patches; attention is not ground-truth contact','No caption metrics measured']}
    for name,value in [('pair_query_summary.json',summary),('private_case_predictions.json',details),('private_target_changes.json',changes),('history.json',histories)]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    torch.save(saved_scores,output/'private_logits.pt')
    # Target-specific supervision is a separate, prespecified nested-CV stage.
    # The source-pretrained heads have never seen any of these61 manifest images.
    adapted, adapted_details, fold_records = {}, {}, []
    n = lengths[0]
    for mode in ('mean_pool','query_blind','attention'):
        oof_scores=torch.zeros(n,6); oof_decisions=torch.full((n,6),-1); coverage=torch.zeros(n,dtype=torch.long)
        for fold in fold_source['folds']:
            fit_ids,cal_ids,test_ids=(set(fold[k]) for k in ('fit_images','calibration_images','evaluation_images'))
            if fit_ids&cal_ids or (fit_ids|cal_ids)&test_ids: raise ValueError('Nested image leakage.')
            fit_cases=[c for c in cases if rows[c['positive_row_index']]['image_id'] in fit_ids]
            cal_cases=[c for c in cases if rows[c['positive_row_index']]['image_id'] in cal_ids]
            selected=sorted({c[k] for c in fit_cases for k in ('positive_row_index','alternate_row_index')})
            remap={old:new for new,old in enumerate(selected)}
            local_cases=[{**c,'positive_row_index':remap[c['positive_row_index']], 'alternate_row_index':remap[c['alternate_row_index']]} for c in fit_cases]
            local_rows=[rows[i] for i in selected]
            model=PairQueryAttention()
            cp=torch.load(output/f'{mode}_seed42.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(cp['model'],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            history=[]
            for epoch in range(30):
                optimizer.zero_grad(); logits=model(patches[selected],endpoints[selected],geometry[selected],text[selected],mode=mode)
                loss,bce,rank=target_objective(logits,local_rows,local_cases,ranking_weight=.2)
                if not torch.isfinite(loss): raise ValueError('Nonfinite target adaptation.')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                history.append({'epoch':epoch+1,'loss':float(loss.detach()),'bce':float(bce.detach()),'ranking':float(rank.detach())})
            model.eval().requires_grad_(False)
            with torch.no_grad():
                scores=torch.cat([model(patches[i:min(i+100,n)],endpoints[i:min(i+100,n)],geometry[i:min(i+100,n)],text[i:min(i+100,n)],mode=mode) for i in range(0,n,100)])
            temperature,scalar_policy,policy=shared_target_policy(scores,reviewed_labels(rows[:n],cal_cases))
            decision=selective_decisions((scores/temperature).sigmoid(),torch.ones_like(scores).bool(),policy)
            test_indices=[i for i,r in enumerate(rows[:n]) if r['image_id'] in test_ids]
            oof_scores[test_indices]=scores[test_indices];oof_decisions[test_indices]=decision[test_indices];coverage[test_indices]+=1
            record={'fold':fold['fold'],'mode':mode,'fit_cases':len(fit_cases),'calibration_cases':len(cal_cases),'temperature':temperature,'policy':scalar_policy}
            fold_records.append(record);histories[f"adapt_{mode}_{fold['fold']}"]=history
            torch.save({'model':model.state_dict(),'mode':mode,'fold':fold,'temperature':temperature,'policy':policy,'protocol_sha256':sha(pp)},output/f"adapt_{mode}_fold{fold['fold']}.pt")
        if any(int(coverage[c[k]])!=1 for c in cases for k in ('positive_row_index','alternate_row_index')):raise ValueError('Target OOF coverage differs.')
        result,detail=target_metrics(oof_scores,oof_decisions,rows[:n],cases)
        adapted[mode]=result;adapted_details[mode]=detail
        saved_scores[f'adapt_{mode}']={'scores':oof_scores,'decisions':oof_decisions,'coverage':coverage}
        print(f'Target adaptation {mode}: {result["all"]["ranking_wins"]}/{len(cases)}',flush=True)
    comparisons={}
    for mode in ('query_blind','attention'):
        counts,changed=target_changes(adapted_details['mean_pool'],adapted_details[mode]);comparisons[mode]=counts
        changes[f'adapt_{mode}_vs_mean_pool']=changed
    summary['target_adaptation']={'results':adapted,'paired_changes_vs_mean_pool':comparisons,'fold_records':fold_records,
        'notes':'Target-only acceptance; no action gate/cap. Learned on reviewed inner fit cases; each outer reviewed image predicted once. Reused development data.'}
    details['adapted']=adapted_details
    for name,value in [('pair_query_summary.json',summary),('private_case_predictions.json',details),('private_target_changes.json',changes),('history.json',histories)]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    torch.save(saved_scores,output/'private_logits.pt')
    summary['compressed_inputs_sha256']=sha(output/'private_compressed_inputs.pt')
    summary['checkpoint_sha256']={p.name:sha(p) for p in sorted(output.glob('*.pt')) if not p.name.startswith('private_')}
    (output/'pair_query_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','patches','package','review','reference','model','output'): parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
