"""Replay all saved readers, calibration decisions and reviewed target metrics."""
import argparse
import json
from pathlib import Path
import torch
from captioning.pair_query_attention import PairQueryAttention
from captioning.reviewed_targets import target_metrics, target_changes
from captioning.factorized_evidence import reviewed_labels
from captioning.evidence_routing import selective_decisions
from run_factorized_evidence_cv import shared_target_policy
from run_pair_query_attention import partition
from run_instance_alignment_diagnostic import fit_thresholds, metrics, edge_metrics
from run_triplet_soft_prior_cached import acceptance
from run_actor_compatibility_cached import read_archive
from import_target_annotations import filter_annotations
from extract_pair_query_patches import sha


def verify(args):
    torch.set_num_threads(2)
    output,package=Path(args.output),Path(args.package)
    summary=json.loads((output/'pair_query_summary.json').read_text())
    protocol=json.loads((output/'protocol.json').read_text())
    if sha(output/'protocol.json')!=summary['protocol_sha256'] or sha(args.source)!=protocol['source_sha256'] or sha(args.review)!=protocol['review_sha256']:
        raise ValueError('Protocol/source changed.')
    for name,digest in summary['checkpoint_sha256'].items():
        if sha(output/name)!=digest:raise ValueError('Checkpoint changed.')
    if sha(output/'private_compressed_inputs.pt')!=summary['compressed_inputs_sha256']:raise ValueError('Inputs changed.')
    data=torch.load(output/'private_compressed_inputs.pt',weights_only=True,map_location='cpu')
    stored=torch.load(output/'private_logits.pt',weights_only=True,map_location='cpu')
    rows,views,text,lengths=read_archive(args.source,['train_pairs.json','val_pairs.json']);n=lengths[0]
    if not torch.equal(text,data['text']):raise ValueError('Cached text changed.')
    labels=torch.tensor([r['labels'] for r in rows])
    provenance=json.loads((package/'image_provenance.json').read_text())
    actual=partition(rows,n,{p['filename'] for p in provenance})
    if actual!=protocol['partition']:raise ValueError('Image exclusion changed.')
    mapping=json.loads((package/'private_mapping.json').read_text())
    cases=filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')),mapping)['eligible_contrasts']
    cal=[i for i,r in enumerate(rows) if r['image_id'] in actual['calibration_images']]
    dev=list(range(n,len(rows)));dev_rows=rows[n:]
    source_details={}
    def predict(cp,count):
        model=PairQueryAttention();model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
        with torch.no_grad():
            return torch.cat([model(data['patches'][i:i+100],data['endpoints'][i:i+100],data['geometry'][i:i+100],text[i:i+100],mode=cp['mode']) for i in range(0,count,100)])[:count]
    for mode in ('clip_prior','mean_pool','query_blind','attention'):
        if mode=='clip_prior':logits=10*torch.einsum('bd,bkd->bk',views[:,2],text)-3
        else:
            cp=torch.load(output/f'{mode}_seed42.pt',weights_only=True,map_location='cpu')
            if cp['protocol_sha256']!=summary['protocol_sha256']:raise ValueError('Wrong source checkpoint protocol.')
            logits=predict(cp,len(rows))
        if not torch.equal(logits,stored[mode]):raise ValueError('Source logits do not replay exactly: '+mode)
        thresholds=fit_thresholds(logits[cal].sigmoid(),labels[cal])
        decisions=(logits.sigmoid()>=torch.tensor(thresholds)).long()
        target,detail=target_metrics(logits[:n],decisions[:n],rows[:n],cases)
        wanted=summary['results'][mode]
        if thresholds!=wanted['thresholds'] or target!=wanted['reviewed_targets']:raise ValueError('Source calibration/targets differ.')
        if metrics(logits[dev].sigmoid(),labels[dev],thresholds)!=wanted['development_metrics'] or acceptance(logits[dev].sigmoid(),dev_rows,thresholds)!=wanted['development_acceptance']:
            raise ValueError('Development metrics differ.')
        if edge_metrics(logits[dev],dev_rows)!=wanted['development_contrasts']:raise ValueError('Development contrasts differ.')
        source_details[mode]=detail
    adapted_details={}
    for mode in ('mean_pool','query_blind','attention'):
        oof=stored['adapt_'+mode]
        replay=torch.zeros(n,6);decisions=torch.full((n,6),-1);coverage=torch.zeros(n,dtype=torch.long)
        for fold in protocol['target_adaptation']['folds']:
            cp=torch.load(output/f"adapt_{mode}_fold{fold['fold']}.pt",weights_only=True,map_location='cpu')
            if cp['fold']!=fold or cp['protocol_sha256']!=summary['protocol_sha256']:raise ValueError('Wrong target checkpoint/fold.')
            # The training runner's last prefix batch ends exactly at n, not total rows.
            model=PairQueryAttention();model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
            with torch.no_grad():
                scores=torch.cat([model(data['patches'][i:min(i+100,n)],data['endpoints'][i:min(i+100,n)],data['geometry'][i:min(i+100,n)],text[i:min(i+100,n)],mode=mode) for i in range(0,n,100)])
            cal_cases=[c for c in cases if rows[c['positive_row_index']]['image_id'] in fold['calibration_images']]
            temp,_,policy=shared_target_policy(scores,reviewed_labels(rows[:n],cal_cases))
            if temp!=cp['temperature'] or policy!=cp['policy']:raise ValueError('Target calibration differs.')
            pred=selective_decisions((scores/temp).sigmoid(),torch.ones_like(scores).bool(),policy)
            ii=[i for i,r in enumerate(rows[:n]) if r['image_id'] in fold['evaluation_images']]
            replay[ii]=scores[ii];decisions[ii]=pred[ii];coverage[ii]+=1
        if not all(torch.equal(a,b) for a,b in [(replay,oof['scores']),(decisions,oof['decisions']),(coverage,oof['coverage'])]):raise ValueError('Target OOF replay differs.')
        result,detail=target_metrics(replay,decisions,rows[:n],cases)
        if result!=summary['target_adaptation']['results'][mode]:raise ValueError('Target metric replay differs.')
        adapted_details[mode]=detail
    for mode in ('query_blind','attention'):
        counts,_=target_changes(adapted_details['mean_pool'],adapted_details[mode])
        if counts!=summary['target_adaptation']['paired_changes_vs_mean_pool'][mode]:raise ValueError('Paired changes differ.')
    report={'saved_heads_strictly_reloaded':18,'source_and_oof_logits_replayed_exactly':True,
        'thresholds_and_target_calibration_replayed_exactly':True,'metrics_and_paired_changes_replayed':True,
        'all_review_manifest_images_excluded_from_source_fit_and_calibration':True,
        'source_protocol_sha256':summary['protocol_sha256']}
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','package','review','output'):parser.add_argument('--'+name,required=True)
    verify(parser.parse_args())
