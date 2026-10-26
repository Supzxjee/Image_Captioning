"""User-run frozen-A feasibility only. No optimizer/backward/training entry point."""
import argparse
import json
import os
from pathlib import Path
import time
import zipfile
from diagnostic_data import A_SHA,BASE_SHA,DATA_SHA,LEGACY_SHA,digest,save
from role_logic import POLICY,order,identity,compare,select_review
from role_parser import PARSER_POLICY,parse_caption

RECEIPT_SHA='867ecb73d880e791c35065863504cee9fd44a3bad63bac326a3c87e5106be51e'
MANIFEST_SHA='f33e9682fa53301efaba458c839d3ce627ebb65858f10f083bd5e4aaed17e6d3'


def budget(cfg):
    if time.time()-cfg['started_at']>POLICY['max_preparation_seconds']:
        raise TimeoutError('30-minute preparation/generation budget reached; preserve partial review, no automatic retry.')


def source_check(cfg):
    for name,expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise ValueError('Source changed: '+name)


def prepare(cfg):
    import h5py
    import numpy as np
    import torch
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.visual_cache import VisualCache
    from hidden_data import locked_groups
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    cfg['started_at']=time.time();save(out/'initial_config.json',cfg);source_check(cfg)
    for key,expected in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=expected:raise ValueError('Different '+key)
        budget(cfg)
    metadata=json.loads(Path('role_metadata.json').read_text(encoding='utf-8'))
    if metadata['schema']!='role-interval-fit-metadata-v1':raise ValueError('Wrong annotation metadata.')
    dataset=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));groups=locked_groups(dataset)
    rows={int(Path(r['filename']).stem.rsplit('_',1)[1]):r for r in dataset['images']}
    if len(rows)!=len(dataset['images']):raise ValueError('Duplicate COCO IDs.')
    reserved=set(sum(metadata['reserved'].values(),[]))
    official_eval={i for i,r in rows.items() if r['split'] not in ('train','restval')}
    fit=metadata['fit'];fit_ids={r['coco_id'] for r in fit}
    if len(fit_ids)!=len(fit) or fit_ids&(reserved|official_eval):raise ValueError('FIT split overlap.')
    for r in fit:
        actual=rows.get(r['coco_id'])
        if actual is None or actual['filename']!=r['filename'] or actual['split']!=r['caption_split']:raise ValueError('FIT/dataset join differs.')
        nodes={n[0]:n[1] for n in r['instances']};labels={(a,p):v for a,p,v in r['actor_labels']}
        if len(nodes)!=len(r['instances']) or len(labels)!=len(r['actor_labels']):raise ValueError('Duplicate graph annotation.')
        if any(v not in (-1,0,1) or p not in POLICY['predicates'] for (_,p),v in labels.items()):raise ValueError('Invalid actor label.')
        if any(nodes.get(s)!='person' or o not in nodes or labels.get((s,p))!=1 for s,p,o in r['positive_edges']):raise ValueError('Invalid positive binding.')
    receipts=[p for p in Path(cfg['stale_root']).rglob('receipt.json') if digest(p)==RECEIPT_SHA]
    if len(receipts)!=1:raise ValueError('Attach one audited stale preparation output; exact receipt not found or duplicated.')
    oldroot=receipts[0].parent
    if digest(oldroot/'manifest.json')!=MANIFEST_SHA:raise ValueError('Different factual replay metadata.')
    replay=json.loads((oldroot/'manifest.json').read_text(encoding='utf-8'))['rows']
    # Feasibility reads ONLY audited replay metadata, not its 2.6GB HDF5 payloads.
    replay=[r for r in replay if r['coco_id'] not in fit_ids|reserved|official_eval]
    requested=fit+replay;names=list(dict.fromkeys(r['filename'] for r in requested))
    print('Loading unchanged legacy prompt embeddings once.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names);del bundle
    visual=VisualCache([cfg['visual_cache']]);valid=lambda r:r['filename'] in legacy and r['coco_id'] in visual.index
    eligible=sorted([r for r in fit if valid(r)],key=lambda r:order(r['coco_id']))
    replay=sorted([r for r in replay if valid(r)],key=lambda r:order(r['coco_id']))
    n=min(2000,len(eligible),len(replay))//4*4
    inventory=dict(fit_total=len(fit),fit_available=len(eligible),replay_available=len(replay),planned_n=n,
        replay_hdf5_verified=False,replay_use='Metadata planning only; future training must verify all payloads.',
        reserved_images=len(reserved),official_eval_overlap=0,source_sha256=metadata['source_sha256'],sources=metadata['sources'])
    save(out/'inventory.json',inventory)
    if n<POLICY['min_pool']:raise RuntimeError('Insufficient disjoint FIT/replay cache coverage; stop without changing pool policy.')
    selected=eligible[:POLICY['sample_images']]
    cfg.update(selected=selected,planned_fit_ids=[r['coco_id'] for r in eligible[:n]],planned_replay_ids=[r['coco_id'] for r in replay[:n]],
        policy=POLICY,parser_policy=PARSER_POLICY,metadata_sha256=digest('role_metadata.json'),
        input_sha256=dict(checkpoint=A_SHA,dataset_json=DATA_SHA,legacy_prompt=LEGACY_SHA,replay_receipt=RECEIPT_SHA,replay_manifest=MANIFEST_SHA),
        test_used=False,training=False,selection_sha256=identity([r['coco_id'] for r in selected]))
    checked={};stage=out/'inputs.h5'
    try:
        with h5py.File(stage,'w') as f:
            for name,shape,dtype in [('visual',(197,768),'f4'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:
                f.create_dataset(name,(len(selected),*shape),dtype=dtype)
            f.create_dataset('coco_ids',data=[r['coco_id'] for r in selected])
            # Read physical row order to avoid 128 random cloud shard reads.
            for index in sorted(range(len(selected)),key=lambda j:visual.index[selected[j]['coco_id']]):
                budget(cfg);r=selected[index];fid,_=visual.index[r['coco_id']]
                if fid not in checked:
                    with h5py.File(visual.paths[fid],'r') as h:meta=json.loads(h.attrs['metadata_json'])
                    expected=dict(dataset_json_sha256=DATA_SHA,model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',
                        preprocessing='bilinear',image_size=[224,224],antialias=True,compute_dtype='float32',autocast=False,
                        mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711])
                    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Different visual preprocessing.')
                    checked[fid]=dict(path=visual.paths[fid],metadata=meta)
                entry=legacy[r['filename']];p=entry['tokens'].float().numpy();mask=entry['mask'].numpy()
                if p.shape!=(20,512) or mask.shape!=(20,) or not np.isfinite(p).all() or not np.isin(mask,[0,1]).all() or not mask.any():raise ValueError('Invalid original prompt shape/value.')
                f['visual'][index]=visual.read(r['coco_id']);f['prompt'][index]=p;f['mask'][index]=mask
                print(f'Staged {index+1}/128 COCO {r["coco_id"]}; frozen feasibility only',flush=True)
            f.attrs['complete']=True
    finally:visual.close()
    cfg.update(stage_sha256=digest(stage),visual_cache_metadata=checked,preparation_seconds=time.time()-cfg['started_at'])
    budget(cfg);save(out/'protocol.json',cfg)
    print('Preparation READY. No training; two workers generate 128 fixed caption pairs.',flush=True)


def build(cfg):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False);weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Expected A wrapper.')
    model=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(state['caption_word2idx'])))
    model.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True)
    if model_digest(model)!=BASE_SHA:raise ValueError('Different A tensors.')
    return model.requires_grad_(False).cuda().eval(),state['caption_word2idx']


def worker(cfg):
    import h5py
    import torch
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst import rollout,decode
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    source_check(cfg);rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2.')
    torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True);budget(cfg)
    out=Path(cfg['output'])
    if digest(out/'inputs.h5')!=cfg['stage_sha256'] or digest(cfg['checkpoint'])!=A_SHA:raise ValueError('Inputs changed.')
    model,vocab=build(cfg);tok=CaptionTokenizer([]);tok.word2idx=vocab;tok.idx2word={i:w for w,i in vocab.items()}
    records=[];started=time.time()
    with h5py.File(out/'inputs.h5','r') as f,torch.inference_mode():
        if not f.attrs.get('complete') or f['coco_ids'][:].tolist()!=[r['coco_id'] for r in cfg['selected']]:raise ValueError('Incomplete/misordered stage.')
        for index in range(rank,len(cfg['selected']),2):
            budget(cfg);r=cfg['selected'][index];seed=POLICY['seed']+r['coco_id']
            memory,pad=model.build_memory(torch.from_numpy(f['visual'][index])[None].cuda(),torch.from_numpy(f['prompt'][index])[None].cuda(),torch.from_numpy(f['mask'][index]).long()[None].cuda())
            greedy,_=rollout(model,memory,pad,tok,sample=False,max_len=30)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            sample,_=rollout(model,memory,pad,tok,sample=True,max_len=30)
            records.append(dict(coco_id=r['coco_id'],seed=seed,greedy=decode(greedy,tok)[0],sample=decode(sample,tok)[0],greedy_ids=greedy[0].cpu().tolist(),sample_ids=sample[0].cpu().tolist()))
            save(out/f'partial_rank{rank}.json',records)
            print(f'GPU{rank}: {len(records)}/64 fixed pairs; no optimization',flush=True)
    if model_digest(model)!=BASE_SHA or any(p.requires_grad for p in model.parameters()):raise ValueError('A changed.')
    budget(cfg);save(out/f'complete_rank{rank}.json',dict(records=records,records_sha256=identity(records),rank=rank,
        protocol_sha256=identity(cfg),selection_sha256=cfg['selection_sha256'],stage_sha256=cfg['stage_sha256'],
        implementation_sha256=cfg['implementation_sha256'],base_sha256=BASE_SHA,runtime=runtime_manifest(),seconds=time.time()-started,optimizer_used=False))


def finish(cfg):
    import spacy
    from role_review import write_review
    source_check(cfg);out=Path(cfg['output'])
    if digest(out/'inputs.h5')!=cfg['stage_sha256']:raise ValueError('Stage changed after generation.')
    if spacy.__version__!=PARSER_POLICY['spacy']:raise ValueError('Different parser software.')
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    if nlp.meta['version']!=PARSER_POLICY['model_version']:raise ValueError('Different parser weights.')
    records=[]
    for rank in (0,1):
        packet=json.loads((out/f'complete_rank{rank}.json').read_text())
        if packet['base_sha256']!=BASE_SHA or packet['optimizer_used']:raise ValueError('Invalid worker.')
        expected_binding=dict(rank=rank,protocol_sha256=identity(cfg),selection_sha256=cfg['selection_sha256'],
            stage_sha256=cfg['stage_sha256'],implementation_sha256=cfg['implementation_sha256'],records_sha256=identity(packet['records']))
        if any(packet.get(k)!=v for k,v in expected_binding.items()):raise ValueError('Worker protocol/stage/selection/source binding differs.')
        if [r['coco_id'] for r in packet['records']]!=[r['coco_id'] for r in cfg['selected'][rank::2]]:raise ValueError('Worker rank assignment differs.')
        if any(r['seed']!=POLICY['seed']+r['coco_id'] for r in packet['records']):raise ValueError('Sample seeds differ.')
        records+=packet['records']
    expected={r['coco_id']:r for r in cfg['selected']}
    if len(records)!=128 or {r['coco_id'] for r in records}!=set(expected):raise ValueError('Not exactly the fixed 128 pairs.')
    cases=[]
    for r in sorted(records,key=lambda r:order(r['coco_id'])):
        budget(cfg);scene=expected[r['coco_id']]
        parsed={key:parse_caption(nlp,r[key]) for key in ('sample','greedy')}
        cases.append(dict(r,scene=scene,parsed=parsed,comparison=compare(parsed['sample']['graph'],parsed['greedy']['graph'],scene)))
    save(out/'cases.json',cases);selected=select_review(cases)
    active=sum(c['comparison']['role_advantage']!=0 for c in cases);operative=sum(c['comparison']['operative'] for c in cases)
    summary=dict(status='WAIT_HUMAN_REVIEW' if active>=16 and operative>=8 else 'FEASIBILITY_NO_GO',images=len(cases),active_pairs=active,operative_pairs=operative,
        interval_operates=sum(c['comparison']['interval_operates'] for c in cases),binding_operates=sum(c['comparison']['binding_operates'] for c in cases),
        search_capped_pairs=sum(c['comparison']['search_capped'] for c in cases),
        binding_claim_validated=False,review_cases=len(selected),human_review_pending=True,training=False,
        scope='FIT diagnostic only. Unknown is not false. Derivation audit is not full-caption truth/preference or held-out efficacy. No automatic training.',
        elapsed_preparation_generation_scoring_seconds=time.time()-cfg['started_at'])
    save(out/'summary.json',summary)
    write_review(out/'review',selected,cfg['images'])
    save(out/'receipt.json',dict(status='FEASIBILITY_PACKET_COMPLETE',summary=summary,
        policy=POLICY,parser_policy=PARSER_POLICY,selection_sha256=cfg['selection_sha256'],
        payload_sha256={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.html','.jpg') and p.name!='receipt.json'}))
    print(json.dumps(summary,indent=2),flush=True)


def archive(cfg):
    out=Path(cfg['output'])
    if not out.exists():return
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.html','.jpg','.log'):z.write(p,p.relative_to(out))
        for name in cfg['implementation_sha256']:z.write(Path(__file__).parent/name,'source/'+name)
    print('SEND:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','finish','archive']);ap.add_argument('--config',required=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));globals()[args.mode](cfg)
