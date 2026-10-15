"""Replay matching/metrics, official scores, preprocessing and a frozen forward pass."""
import argparse,importlib.util,json
from pathlib import Path
import torch
from PIL import Image
from captioning.upt_reference import load_reference,prepare_image,infer
from captioning.hoi_transfer_eval import match_inventory,target_case,aggregate_cases,exploratory_pairwise_fallback
from run_actor_compatibility_cached import read_archive
from import_target_annotations import filter_annotations
from extract_pair_query_patches import sha


def run(args):
    torch.set_num_threads(4)
    output,package,root=map(Path,(args.output,args.package,args.upt))
    summary=json.loads((output/'pretrained_hoi_summary.json').read_text());protocol=json.loads((output/'protocol.json').read_text())
    if sha(output/'protocol.json')!=summary['protocol_sha256'] or sha(args.checkpoint)!=protocol['checkpoint_sha256'] or sha(args.source)!=protocol['source_sha256'] or sha(args.review)!=protocol['review_sha256']:raise ValueError('Provenance changed.')
    metadata=json.loads((root/'hicodet/instances_train2015.json').read_text());corr=[[] for _ in metadata['objects']]
    for _,obj,verb in metadata['correspondence']:corr[obj].append(verb)
    rows,views,text,lengths=read_archive(args.source,['train_pairs.json','val_pairs.json'])
    mapping=json.loads((package/'private_mapping.json').read_text())
    cases=filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')),mapping)['eligible_contrasts']
    clip=10*torch.einsum('bd,bkd->bk',views[:,2],text)-3
    predictions={};details=[];official_cells=0
    for image_id in protocol['images']:
        path=output/f'private_image_{image_id}.pt'
        if sha(path)!=summary['cache_sha256'][path.name]:raise ValueError('Frozen cache changed.')
        record=torch.load(path,weights_only=True,map_location='cpu');predictions[image_id]=record
        h,o=record['subject_indices'],record['object_indices'];lookup={(int(a),int(b)):i for i,(a,b) in enumerate(zip(h,o))}
        official=record['official_output'];prior=record['prior'].prod(0)
        if len(official['scores'])!=int((prior>0).sum()):raise ValueError('Official ontology support differs.')
        x,y=torch.nonzero(prior).unbind(1)
        expected=record['logits'][x,y].sigmoid()*prior[x,y]
        if not torch.equal(expected,official['scores']) or not torch.equal(torch.stack([h[x],o[x]]),official['pairing']) or not torch.equal(y,official['labels']):
            raise ValueError('Official postprocessing scores/pair order differ.')
        official_cells+=len(official['scores'])
        candidates=[r for r in rows[:lengths[0]] if r['image_id']==image_id]
        assigned,_=match_inventory(candidates,record,metadata['objects'],protocol['minimum_iou'])
        details.extend(target_case(c,rows,record,assigned,metadata['verbs'],corr,clip) for c in cases if rows[c['positive_row_index']]['image_id']==image_id)
    if details!=json.loads((output/'private_case_predictions.json').read_text()) or aggregate_cases(details)!=summary['evaluation']:raise ValueError('Target evaluation differs.')
    model,runtime=load_reference(root,args.pocket,args.checkpoint,corr)
    # Compare against the pinned upstream resize/normalization directly.
    spec=importlib.util.spec_from_file_location('_upt_transforms',root/'detr/datasets/transforms.py')
    transforms=importlib.util.module_from_spec(spec);spec.loader.exec_module(transforms)
    from torchvision.transforms import functional as TF
    checked=0
    for image_id in protocol['images']:
        filename=next(r['filename'] for r in rows if r['image_id']==image_id)
        with Image.open(package/'images'/filename) as source:
            image=source.convert('RGB');ours=prepare_image(image)
            resized,_=transforms.resize(image,None,800,max_size=1333)
            upstream=TF.normalize(TF.to_tensor(resized),[.485,.456,.406],[.229,.224,.225])
        if not torch.equal(ours,upstream):raise ValueError('Preprocessing differs from upstream.')
        checked+=1
        if checked==1:
            replay=infer(model,ours);original=predictions[image_id]
            for key in ('boxes','scores','labels','logits','prior','subject_indices','object_indices','size'):
                if not torch.equal(replay[key],original[key]):raise ValueError('Frozen forward pass differs: '+key)
    exploratory=exploratory_pairwise_fallback(details)
    (output/'exploratory_pairwise_fallback.json').write_text(json.dumps(exploratory,indent=2),encoding='utf-8')
    report={'strict_checkpoint_tensors':runtime['state_tensors'],'all22_cache_hashes_checked':True,
        'all_case_matching_and_metrics_replayed_exactly':True,'official_output_cells_replayed_exactly':official_cells,
        'images_preprocessing_bit_exact_to_upstream':checked,'one_full_model_forward_replayed_exactly':True,
        'exploratory_fallback':exploratory}
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('upt','pocket','checkpoint','source','package','review','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
