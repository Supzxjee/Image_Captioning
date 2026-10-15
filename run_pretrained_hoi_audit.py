"""Run frozen official UPT HICO-DET inference; no local relation-head fitting."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
from PIL import Image
from captioning.upt_reference import load_reference,prepare_image,infer,UPT_REVISION,POCKET_REVISION
from captioning.hoi_transfer_eval import match_inventory,target_case,aggregate_cases
from run_actor_compatibility_cached import read_archive
from import_target_annotations import filter_annotations
from extract_pair_query_patches import sha


def run(args):
    torch.set_num_threads(4)
    root,pocket,package,output=map(Path,(args.upt,args.pocket,args.package,args.output))
    output.mkdir(parents=True,exist_ok=True)
    expected={root:UPT_REVISION,pocket:POCKET_REVISION,root/'detr':'820d17abc697e8dd5f49d4cd1fdbde412545bac5',root/'hicodet':'9dd59c1f6ee2ddc011567765d0bc644858391fea'}
    for directory,revision in expected.items():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()!=revision:raise ValueError('Reference revision changed.')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=directory,text=True).strip():raise ValueError('Upstream tracked code was modified.')
    if sha(args.checkpoint)!='12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c':raise ValueError('Checkpoint bytes changed.')
    metadata_path=root/'hicodet/instances_train2015.json';metadata=json.loads(metadata_path.read_text())
    correspondence=[[] for _ in metadata['objects']]
    for _,obj,verb in metadata['correspondence']:correspondence[obj].append(verb)
    mapping=json.loads((package/'private_mapping.json').read_text())
    if sha(package/'cases.json')!=mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json')!=mapping['source_pairs_sha256']:raise ValueError('Review manifest changed.')
    rows,views,text,lengths=read_archive(args.source,['train_pairs.json','val_pairs.json'])
    if rows[:lengths[0]]!=json.loads(Path('audit/interaction_region_train_pairs.json').read_text()):raise ValueError('Review/source order changed.')
    wanted_source=json.loads(Path('audit/interaction_region_protocol.json').read_text())['splits']['train']['source_sha256']
    if sha(args.source)!=wanted_source:raise ValueError('CLIP reference source changed.')
    cases=filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')),mapping)['eligible_contrasts']
    image_ids=sorted({rows[c['positive_row_index']]['image_id'] for c in cases})
    provenance={r['filename']:r for r in json.loads((package/'image_provenance.json').read_text())}
    protocol={'variant':'pretrained_upt_hico_target_audit_v1','reference_revisions':{str(k):v for k,v in expected.items()},
        'checkpoint_sha256':sha(args.checkpoint),'checkpoint_url':'https://drive.google.com/file/d/1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z/view',
        'upstream_url':'https://github.com/fredzzhang/upt','training_dataset':'HICO-DET; no local fine-tuning',
        'metadata_sha256':sha(metadata_path),'source_sha256':sha(args.source),'review_sha256':sha(args.review),
        'manifest_sha256':mapping['manifest_sha256'],'minimum_iou':.5,'matching':'class-consistent one-to-one maximum-total-IoU Hungarian;no relation labels',
        'primary':'raw trained verb logits where all three instances localize and both object/verb combinations occur in HICO ontology',
        'secondary':'official sigmoid times native class/detection prior; detection-confidence-only control',
        'go_rule':'coverage>=80% of34 and raw wins>=same-subset CLIP wins+3; only go to larger audit, not production',
        'images':image_ids,'cases':len(cases),'training_performed':False,'caption_decoder_changed':False,
        'limitations':['Inspected single-reviewer development pool','HICO relation training differs; COCO detector pretraining may have seen image objects','Not a same-training-budget architecture ablation','Missing/ontology-unsupported is unknown, not false']}
    pp=output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Changed audit protocol; use fresh output.')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model,runtime=load_reference(root,pocket,args.checkpoint,correspondence)
    clip_logits=10*torch.einsum('bd,bkd->bk',views[:,2],text)-3
    details=[];inventory_counts=[];cache_hashes={};started=time.monotonic()
    for image_id in image_ids:
        candidates=[r for r in rows[:lengths[0]] if r['image_id']==image_id]
        filename=candidates[0]['filename'];image_path=package/'images'/filename
        if sha(image_path)!=provenance[filename]['sha256']:raise ValueError('Reviewed image bytes changed.')
        with Image.open(image_path) as image:
            if list(image.size)!=provenance[filename]['image_size']:raise ValueError('Image size changed.')
            pixels=prepare_image(image)
        prediction=infer(model,pixels)
        if not torch.isfinite(prediction['logits']).all():raise ValueError('Nonfinite HOI output.')
        cache=output/f'private_image_{image_id}.pt';torch.save(prediction,cache);cache_hashes[cache.name]=sha(cache)
        matches,counts=match_inventory(candidates,prediction,metadata['objects'],.5)
        inventory_counts.append({'image_id':image_id,**counts})
        current=[c for c in cases if rows[c['positive_row_index']]['image_id']==image_id]
        details.extend(target_case(c,rows,prediction,matches,metadata['verbs'],correspondence,clip_logits) for c in current)
        print(f'Frozen UPT {len(inventory_counts)}/{len(image_ids)} images | {time.monotonic()-started:.1f}s',flush=True)
    summary={'variant':protocol['variant'],'protocol_sha256':sha(pp),'runtime':runtime,'evaluation':aggregate_cases(details),
        'image_count':len(image_ids),'case_count':len(cases),'cache_sha256':cache_hashes,'production_promotion':False,
        'training_performed':False,'caption_metrics_measured':False,'seconds':time.monotonic()-started}
    for name,value in [('pretrained_hoi_summary.json',summary),('private_case_predictions.json',details),('private_inventory_matches.json',inventory_counts)]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('upt','pocket','checkpoint','source','package','review','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
