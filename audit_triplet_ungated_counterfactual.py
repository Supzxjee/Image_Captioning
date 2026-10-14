"""Frozen full100 counterfactual; never modifies the production prompt cache."""
import argparse
import json
from pathlib import Path

import torch
from evaluate_frozen_residual import sha
from captioning.selected_relation_prompts import build_selected_prompts


def main():
    parser=argparse.ArgumentParser()
    for name in ('head-directory','scored','source','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    head=Path(args.head_directory);scored=Path(args.scored);output=Path(args.output)
    summary=json.loads((head/'triplet_summary.json').read_text())
    frozen=json.loads((scored/'frozen_audit_summary.json').read_text())
    if sha(args.source)!=summary['sources']['full100'] or summary['sources']['full100']!=frozen['source_sha256']:
        raise ValueError('Full100 source changed.')
    predictions=json.loads((head/'candidate_predictions.json').read_text())
    reference=json.loads((scored/'all_candidate_predictions.json').read_text())
    if len(predictions)!=len(reference) or any(any(row[k]!=saved[k] for k in row if k!='triplet_activation')
                                              for row,saved in zip(predictions,reference)):
        raise ValueError('Candidate identity/order changed.')
    cp=torch.load(head/'triplet_head.pt',map_location='cpu',weights_only=True)
    gated=json.loads((head/'triplet_prompt_cache.json').read_text())
    inventory={f:[{'label':o['label'],'bbox':o['bbox'],'confidence':o['confidence']} for o in e['objects']]
               for f,e in json.loads((scored/'baseline_prompt_cache.json').read_text()).items()}
    prompts=build_selected_prompts([{**r,'scores':r['triplet_activation']} for r in predictions],cp['thresholds'],object_inventory=inventory)
    def keys(cache):
        return {(f,r['subject_instance'],r['relation'],r['object_instance']) for f,e in cache.items() for r in e['relations']}
    original=keys(gated);counterfactual=keys(prompts)
    review=json.loads((scored/'assistant_visual_review.json').read_text())
    reviewed=[]
    for claim in review['claims']:
        key=tuple(claim[k] for k in ('filename','subject_instance','relation','object_instance'))
        reviewed.append({**claim,'gated_selected':key in original,'ungated_selected':key in counterfactual})
    result={'head_sha256':sha(head/'triplet_head.pt'),'source_sha256':summary['sources']['full100'],
            'training_performed':False,'threshold_refit':False,'pipeline_changed':False,'max_relations':3,
            'gated_claims':len(original),'ungated_claims':len(counterfactual),
            'gated_relation_images':sum(bool(e['relations']) for e in gated.values()),
            'ungated_relation_images':sum(bool(e['relations']) for e in prompts.values()),
            'ungated_pre_cap_claims':sum(e['above_threshold_claims'] for e in prompts.values()),
            'added_after_cap':len(counterfactual-original),'removed_by_cap_competition':len(original-counterfactual),
            'known_review_membership':reviewed,
            'notes':'Counterfactual only: existing frozen verifier scores/thresholds, remove R50 hard gate. '
                    'Full100 is reused development; prior assistant reviews are selected, method-visible and incomplete. '
                    'Do not infer accuracy for unreviewed/new claims; margin-based cap can displace previously accepted claims.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'counterfactual_prompt_cache.json').write_text(json.dumps(prompts,indent=2),encoding='utf-8')
    (output/'counterfactual_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='known_review_membership'},indent=2),flush=True)
    print('PREVIOUS CONTRADICTIONS',json.dumps([c for c in reviewed if c['assistant_status']=='contradicted' and c['ungated_selected']],indent=2),flush=True)


if __name__=='__main__':main()
