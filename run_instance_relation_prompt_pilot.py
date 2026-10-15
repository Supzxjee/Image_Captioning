"""Local cache-only pilot; no notebook/model inference or caption test tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captioning.hoi_transfer_eval import match_inventory,node_key
from captioning.instance_relation_prompts import METHOD,pair_inventory,fit_gates,metrics,emit_prompts
from captioning.hoi_caption_reranking import VERBS


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(a):
    torch.set_num_threads(2)
    repo=Path(__file__).resolve().parent;root=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((root/'protocol.json').read_text())
    hashes=json.loads((root/'cache_sha256.json').read_text())
    meta=json.loads(Path(a.metadata).read_text())
    assert sha(a.metadata)==old['metadata_sha256']
    # Only IDs are consulted for leakage exclusions, never test references/scores.
    heldout=set(json.loads(Path(a.heldout_ids).read_text()))
    cal=set(old['calibration_images'])-heldout;dev=set(old['development_images'])-heldout
    assert cal and dev and not cal&dev
    rows=[]
    for name in ('train','val'):
        rows+=json.loads((repo/f'audit/interaction_region_{name}_pairs.json').read_text())
    correspondence=[[] for _ in meta['objects']]
    for _,obj,verb in meta['correspondence']:correspondence[obj].append(verb)
    cells={'calibration':[],'development':[]};labels={k:[] for k in cells};graphs={}
    for image_id in sorted(cal|dev):
        path=root/f'private_image_{image_id}.pt'
        assert sha(path)==hashes[path.name]
        prediction=torch.load(path,map_location='cpu',weights_only=True)
        source=[r for r in rows if r['image_id']==image_id]
        mapping,_=match_inventory(source,prediction,meta['objects'])
        inventory=pair_inventory(prediction,meta['objects'],meta['verbs'],correspondence)
        lookup={(c['subject_index'],c['object_index'],c['verb']):c for c in inventory}
        graphs[image_id]={'filename':source[0]['filename'],'inventory':inventory}
        split='calibration' if image_id in cal else 'development'
        for r in source:
            s=mapping.get(node_key(r['subject'],r['subject_box']))
            o=mapping.get(node_key(r['object'],r['object_box']))
            for j,verb in enumerate(VERBS):
                c=lookup.get((s['index'],o['index'],verb)) if s and o else None
                cells[split].append(c);labels[split].append(r['labels'][j])
    policies={};results={}
    for name,local in [('raw_only',False),('pair_local',True)]:
        policies[name]={}
        for verb in VERBS:
            selected=[i for i,c in enumerate(cells['calibration']) if c and c['verb']==verb]
            policies[name][verb]=fit_gates([cells['calibration'][i] for i in selected],
                [labels['calibration'][i] for i in selected],local_margin=local)
        results[name]=metrics(cells['development'],labels['development'],policies[name])
    # Prompt export sees predictions/policies only, not source labels or matching.
    exported=[{'image_id':i,'filename':graphs[i]['filename'],
               'prompts':emit_prompts(graphs[i]['inventory'],policies['pair_local'])} for i in sorted(dev)]
    protocol={'method':METHOD,'calibration_images':sorted(cal),'development_images':sorted(dev),
        'excluded_test_and_caption_validation_ids_sha256':sha(a.heldout_ids),'source_cache_protocol_sha256':sha(root/'protocol.json'),
        'precision_target':.9,'minimum_known':5,'margin_grid':[None,-2,-1,0,.5,1,2],
        'selection':'Calibration TP maximum subject to empirical precision; tie fewer FP then raw-only.',
        'test_used_for_fitting':False,'model_inference_performed':False,'caption_metrics_measured':False,
        'scope':'Reused source development diagnostic. Source actor-absent negatives are not a complete target-identity benchmark.',
        'unknown':'Abstain; no negative prompt or contradiction label.'}
    baseline=results['raw_only'];new=results['pair_local']
    # Fixed before execution; success cannot be obtained by dropping all positives.
    go=(new['known_precision'] is not None and new['known_precision']>=.9
        and new['accepted_true']>=baseline['accepted_true'] and new['accepted_false']<baseline['accepted_false'])
    summary={'protocol':protocol,'policies':policies,'development':results,
        'emitted_prompt_instances':sum(len(r['prompts']) for r in exported),
        'images_with_emitted_prompts':sum(bool(r['prompts']) for r in exported),
        'promotion_gate_passed':go,'production_promotion':False,'novelty_established':False}
    write(out/'protocol.json',protocol);write(out/'prompt_cache.json',exported);write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','metadata','heldout-ids','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
