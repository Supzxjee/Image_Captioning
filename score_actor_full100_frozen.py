"""Frozen residual/actor/fusion scoring and paired visual review on a locked100-image pool."""
import argparse,csv,json,zipfile,random,html
from pathlib import Path
from collections import Counter
import torch
from captioning.actor_compatibility import ActorCompatibility,pair_geometry
from captioning.instance_contrast_alignment import ResidualInstanceAlignment
from captioning.detector_candidates import person_object_pairs
from captioning.selected_relation_prompts import build_selected_prompts
from run_actor_compatibility_cached import read_archive
from evaluate_frozen_residual import sha,newline_hashes


def main():
    p=argparse.ArgumentParser()
    for name in ('source','residual-head','actor-head','fusion-head','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--protocol',default='audit/actor_soft_fusion_full100_protocol.json')
    p.add_argument('--exclusions',default='audit/actor_soft_fusion_full100_exclusions.json')
    args=p.parse_args();protocol=json.loads(Path(args.protocol).read_text())
    for name in ('residual','actor','fusion'):
        if sha(getattr(args,name+'_head'))!=protocol[name+'_head_sha256']:raise ValueError(name+' checkpoint changed.')
    if protocol['exclusions_sha256'] not in newline_hashes(args.exclusions):raise ValueError('Exclusions changed.')
    exclusions=set(json.loads(Path(args.exclusions).read_text())['image_ids'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.source) as z:
        def read(name):
            files=[f for f in z.namelist() if f.endswith(name)]
            if len(files)!=1:raise ValueError(name)
            return z.read(files[0])
        selection_raw=read('data/image_list.json');selection=json.loads(selection_raw)
        summary=json.loads(read('data/full_candidate_summary.json'))
        import hashlib
        if hashlib.sha256(selection_raw).hexdigest()!=summary['image_list_sha256']:raise ValueError('Selection hash mismatch.')
        if selection['exclusions_sha256']!=protocol['exclusions_sha256'] or selection['caption_json_sha256']!=protocol['caption_json_sha256']:
            raise ValueError('Selection sources changed.')
        eligible=selection['eligible_image_ids']
        if eligible!=sorted(set(eligible)) or set(eligible)&exclusions:raise ValueError('Eligible pool invalid.')
        shuffled=eligible.copy();random.Random(protocol['selection_seed']).shuffle(shuffled)
        images=selection['images']
        if len(images)!=protocol['images'] or [r['image_id'] for r in images]!=shuffled[:protocol['images']]:
            raise ValueError('Image selection differs from locked recipe.')
        if any(r['caption_split'] not in ('train','restval') for r in images):raise ValueError('Caption heldout included.')
        config=json.loads(read('manifest.json'))['configuration']
        if config['protocol_sha256'] not in newline_hashes(args.protocol):raise ValueError('Extraction protocol changed.')
        if config['model']!=protocol['model']:raise ValueError('Feature model changed.')
        inventory=json.loads(read('data/object_inventory.json'))
        if set(inventory)!={r['filename'] for r in images}:raise ValueError('Inventory coverage mismatch.')
        if any(summary[k]!=protocol[k] for k in ('min_confidence','max_objects','duplicate_iou')):raise ValueError('Detector rules changed.')
        if summary['coordinate_format']!='pixel_xyxy':raise ValueError('Wrong box convention.')
        if summary['detection_cache_sha256']!=protocol['detection_cache_sha256']:raise ValueError('Detector cache changed.')
        for filename in inventory:
            for suffix in ('full','marked'):
                name=Path(filename).stem+'_'+suffix+'.jpg';(output/name).write_bytes(read('data/'+name))
    rows,views,text,_=read_archive(args.source,['evaluation_pairs.json'])
    expected=[r for image in images for r in person_object_pairs(image,inventory[image['filename']])]
    if rows!=expected:raise ValueError('Candidate reconstruction mismatch.')
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    ac=torch.load(args.actor_head,map_location='cpu',weights_only=True)
    fc=torch.load(args.fusion_head,map_location='cpu',weights_only=True)
    residual=ResidualInstanceAlignment(scale=rc['scale']);residual.load_state_dict(rc['model']);residual.eval().requires_grad_(False)
    actor=ActorCompatibility();actor.load_state_dict(ac['model']);actor.eval().requires_grad_(False)
    torch.set_num_threads(2)
    with torch.no_grad():
        base,_,_=residual(views,text);logits=actor(views,pair_geometry(rows),text)
        fused=((((torch.stack([base,logits],dim=-1)-fc['mean'])/fc['std'])*fc['weights']).sum(-1)+fc['bias']).sigmoid()
    old_rows=[{**row,'scores':base[i].tolist()} for i,row in enumerate(rows)]
    new_rows=[]
    for i,row in enumerate(rows):
        values=fused[i].tolist()
        for c in range(6):
            if base[i,c]<rc['thresholds'][c]:values[c]=-2.
        new_rows.append({**row,'scores':values})
    old=build_selected_prompts(old_rows,rc['thresholds'],object_inventory=inventory)
    new=build_selected_prompts(new_rows,fc['thresholds'],object_inventory=inventory)
    review=[];cards=[]
    for filename in inventory:
        claims={}
        for method,cache in (('baseline',old),('fusion',new)):
            for claim in cache[filename]['relations']:
                key=(claim['subject_instance'],claim['relation'],claim['object_instance'])
                claims.setdefault(key,{'filename':filename,**claim,'baseline_selected':False,'fusion_selected':False})[method+'_selected']=True
        review.extend(claims.values())
        cards.append('<article><h3>'+html.escape(filename)+'</h3><img src="'+Path(filename).stem+'_full.jpg"><img src="'+Path(filename).stem+'_marked.jpg"><pre>'+html.escape(json.dumps({'baseline':old[filename]['relations'],'fusion':new[filename]['relations']},indent=2))+'</pre></article>')
    with (output/'paired_claim_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['filename','subject_instance','relation','object_instance','subject','object','baseline_selected','fusion_selected','human_status','human_notes']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(review)
    blind=review.copy();random.Random(42).shuffle(blind)
    membership={}
    with (output/'blind_claim_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['claim_id','filename','subject_instance','relation','object_instance','subject','object','human_status','human_notes']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for i,claim in enumerate(blind):
            identity=f'claim_{i:04d}'
            writer.writerow({**claim,'claim_id':identity})
            membership[identity]={'baseline_selected':claim['baseline_selected'],'fusion_selected':claim['fusion_selected']}
    (output/'claim_method_membership.json').write_text(json.dumps(membership,indent=2),encoding='utf-8')
    with (output/'image_coverage_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['filename','supported_instance_relations_json','annotation_complete','human_notes'])
        writer.writeheader();writer.writerows({'filename':filename} for filename in inventory)
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;margin:24px}img{max-width:48%}pre{white-space:pre-wrap}article{border-bottom:1px solid #aaa}</style><h1>Paired frozen-method review; annotation pending</h1>'+''.join(cards),encoding='utf-8')
    result={'variant':protocol['variant'],'images':len(images),'pairs':len(rows),'excluded_overlap':0,
            'baseline_selected_claims':sum(len(e['relations']) for e in old.values()),
            'fusion_selected_claims':sum(len(e['relations']) for e in new.values()),
            'fusion_modes':dict(Counter(e['mode'] for e in new.values())),'source_sha256':sha(args.source),
            'notes':'Frozen heads/weights/thresholds; no training/refit. Blind selected-claim annotation supports precision only; image-level exhaustive supported-instance annotations are required for recall. No automatic precision/recall or quality claim.'}
    for name,value in [('baseline_prompt_cache.json',old),('fusion_prompt_cache.json',new),('frozen_audit_summary.json',result),
                       ('all_candidate_predictions.json',[{**r,'baseline_scores':base[i].tolist(),'fusion_activation':fused[i].tolist()} for i,r in enumerate(rows)])]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
