"""Reuse fixed42 and run paired7/123 controls without choosing a winning seed."""
import argparse,json,os,statistics,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path('audit/triplet_verifier_seed_stability_protocol.json').read_text())
    root=Path('C:/KhoaLuanTotNghiep/Code/output');downloads=Path('C:/Users/leduc/Downloads')
    from evaluate_frozen_residual import sha
    inputs={'v2':downloads/'instance_contrast_alignment_v2_diagnostic_1024_review.zip',
            'actor':downloads/'actor_all_endpoints_features_v1_review.zip',
            'candidate':downloads/'actor_soft_fusion_full100_features_review.zip',
            'transfer':downloads/'residual_detector_boxes_500_features_review.zip'}
    for name,path in inputs.items():
        if sha(path)!=protocol[name+'_source_sha256']:raise ValueError(name+' source changed.')
    reuse={'candidate_text':(root/'text_conditioned_triplet_verifier_v1',root/'triplet_verifier_detector_transfer_v1'),
           'relation_bank':(root/'triplet_relation_only_ablation_v1',root/'triplet_relation_only_detector_transfer_v1')}
    runs=[]
    for seed in protocol['seeds']:
        for method in ('candidate_text','relation_bank'):
            if seed==42:
                head_dir,transfer_dir=reuse[method]
                if sha(head_dir/'triplet_head.pt')!=protocol['reuse_seed42_heads'][method]:raise ValueError('Fixed42 head changed.')
            else:
                head_dir=output/f'{method}_seed{seed}';transfer_dir=output/f'{method}_seed{seed}_transfer'
                command=[sys.executable,'run_triplet_verifier_cached.py','--v2-source',str(inputs['v2']),
                    '--actor-source',str(inputs['actor']),'--candidate-source',str(inputs['candidate']),
                    '--scored',str(root/'actor_soft_fusion_full100_audit'),'--output',str(head_dir),'--seed',str(seed)]
                if method=='relation_bank':command+=['--relation-only-text']
                print(f'Training {method}, seed{seed}',flush=True)
                with (output/f'{method}_seed{seed}_training.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                command=[sys.executable,'evaluate_triplet_transfer_cached.py','--source',str(inputs['transfer']),
                    '--head-directory',str(head_dir),'--reference-directory',str(root/'residual_detector_boxes_500_evaluation'),
                    '--v2-source',str(inputs['v2']),'--actor-source',str(inputs['actor']),'--output',str(transfer_dir)]
                with (output/f'{method}_seed{seed}_transfer.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            train=json.loads((head_dir/'triplet_summary.json').read_text());transfer=json.loads((transfer_dir/'transfer_summary.json').read_text())
            if train['seed']!=seed or train['epochs']!=protocol['epochs'] or train['train_pairs']!=1776:raise ValueError('Training config changed.')
            if transfer['train_image_overlap']!=0 or transfer['threshold_refit']:raise ValueError('Invalid transfer protocol.')
            run={'seed':seed,'method':method,'dev_AP':train['dev_metrics']['macro_AP_on_known_labels'],
                 'dev_actor_wins':train['dev_actor_edges']['wins'],'full100_claims':train['candidate_selected_claims'],
                 'transfer':transfer['after'],'lost_positives':transfer['lost_positives'],
                 'removed_negatives':transfer['removed_negatives'],'head_sha256':sha(head_dir/'triplet_head.pt')}
            runs.append(run);print(json.dumps(run),flush=True)
    aggregate={}
    for method in ('candidate_text','relation_bank'):
        values=[r for r in runs if r['method']==method];aggregate[method]={}
        measures={'dev_AP':[r['dev_AP'] for r in values],
                  'transfer_precision':[r['transfer']['precision_on_known'] for r in values],
                  'transfer_recall':[r['transfer']['recall_on_known'] for r in values],
                  'retained_positives':[r['transfer']['positive'] for r in values],
                  'retained_negatives':[r['transfer']['negative'] for r in values]}
        for key,series in measures.items():
            aggregate[method][key]={'mean':statistics.mean(series),'sample_sd':statistics.stdev(series),'values':series}
    paired=[]
    for seed in protocol['seeds']:
        a=next(r for r in runs if r['seed']==seed and r['method']=='candidate_text')
        b=next(r for r in runs if r['seed']==seed and r['method']=='relation_bank')
        paired.append({'seed':seed,'dev_AP_delta':a['dev_AP']-b['dev_AP'],
            'transfer_positive_delta':a['transfer']['positive']-b['transfer']['positive'],
            'transfer_negative_delta':a['transfer']['negative']-b['transfer']['negative']})
    result={'protocol':protocol,'runs':runs,'aggregate':aggregate,'paired_deltas':paired,
            'notes':'All three seeds reported; no seed or epoch selection, no checkpoint promotion. Reused development sets. Sample SD across seeds is not an image confidence interval.'}
    (output/'seed_stability_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PAIRED DELTAS',json.dumps(paired),flush=True)


if __name__=='__main__':main()
