"""User-run frozen-A observation. No optimizer, backward, or new prompt inputs."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from diagnostic_data import *


def prepare(cfg):
    import torch
    from transformers import CLIPTokenizerFast
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_consistency import ALIASES
    from hidden_data import locked_groups
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    save(out/'policy.json',POLICY)
    for key,sha in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=sha:raise ValueError('Wrong pinned input: '+key)
    root=Path(cfg['noun_manifest']);receipt=root/'manifest_receipt.json'
    if digest(receipt)!=NOUN_SHA:raise ValueError('Wrong sealed noun-box receipt.')
    receipt=json.loads(receipt.read_text())
    for name,sha in receipt['payload_sha256'].items():
        if Path(name).name!=name or digest(root/name)!=sha:raise ValueError('Wrong noun payload: '+name)
    dataset=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
    groups=locked_groups(dataset)['train'];masks=json.loads((root/'mask_bank.json').read_text())
    byimage=defaultdict(dict)
    for s in json.loads((root/'spans.json').read_text()):
        byimage[s['coco_id']][s['category']]=masks[s['mask_key']]
    names=[r['filename'] for r in groups]
    print('Loading original legacy cache once. No text reconstructed from reference captions.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names)
    if set(legacy)!=set(names):raise ValueError('Original legacy cache missing TRAIN images.')
    save(out/'cache_schema.json',dict(bundle_keys=list(bundle),metadata=bundle.get('metadata',{}),
        first_entry_keys=list(legacy[names[0]]),first_token_shape=list(legacy[names[0]]['tokens'].shape)))
    external={}
    text_source=cfg.get('prompt_text_json')
    if text_source:
        path=Path(text_source)
        if not path.is_file():raise FileNotFoundError('Attach the ORIGINAL prompt_cache.json used to encode legacy embeddings: '+str(path))
        external=text_index(json.loads(path.read_text(encoding='utf-8')))
        cfg['prompt_text_sha256']=digest(path)
        save(out/'text_source.json',dict(path=str(path),sha256=cfg['prompt_text_sha256'],entries=len(external),
            status='UNVERIFIED_UNTIL_SELECTED_TEXT_EMBEDDING_BINDING_PASSES'))
    tok=CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV)
    rejects=Counter();candidates=[];text_missing=0;class_counts=Counter();missing_examples=[]
    # All eligibility decisions are metadata-only and completed before caption-model loading.
    for r in groups:
        entry=legacy[r['filename']];text=prompt_text(entry)
        other=external.get(r['filename'])
        if text and other and text!=other:raise ValueError('Original text sources conflict: '+r['filename'])
        text=text or other
        if text is None:
            text_missing+=1;rejects['missing_original_prompt_text']+=1
            if len(missing_examples)<3:missing_examples.append(dict(filename=r['filename'],keys=list(entry)))
            continue
        length=entry['tokens'].shape[0]
        enc=tok(text,padding='max_length',truncation=True,max_length=length,return_offsets_mapping=True)
        if not np.array_equal(np.asarray(enc['attention_mask']),entry['mask'].cpu().numpy()):
            rejects['tokenizer_mask_mismatch']+=1;continue
        available=byimage[r['coco_id']];phrases=[];seen=set()
        for p in phrase_positions(text,ALIASES,enc):
            label=p['category']
            if label not in available or label in seen:continue
            box=available[label];b=box['normalized_xyxy']
            if len(b)!=4 or not 0<=b[0]<b[2]<=1 or not 0<=b[1]<b[3]<=1:raise ValueError('Invalid sealed box.')
            # Only singular/paired aliases admitted by the preexisting conservative v1.1 policy.
            from manifest_policy_v11 import POLICY as noun_policy,context_exclusion
            from audit import spans
            raw,matches=spans(text,ALIASES)
            matches=[s for s in matches if s[2]==label and s[3]==p['surface'].lower()]
            if not matches:continue
            start,end,_,surface=matches[0]
            norm=[re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',w) for w in raw]
            if surface in noun_policy['ambiguity_exclusions'] or (surface in noun_policy['plural_surfaces'] and surface not in noun_policy['paired_surface_exceptions']):continue
            if context_exclusion(norm,start,end,label,raw_tokens=raw):continue
            seen.add(label);phrases.append(dict(p,box=b,mask=box['patch_area_fraction'],annotation_id=box['annotation_id']))
        if len(phrases)<2:rejects['fewer_than_two_unique_groundable_prompt_classes']+=1;continue
        pairs=axis_pairs(phrases)
        if not pairs:rejects['no_separated_axis_pair_gap_at_least_one_patch']+=1;continue
        candidates.append(dict(r,prompt=text,input_ids=enc['input_ids'],phrases=phrases,pairs=pairs))
        class_counts.update(p['category'] for p in phrases)
    candidates.sort(key=lambda r:order(r['coco_id']));selected=candidates[:POLICY['max_images']]
    coverage=dict(train_population=len(groups),original_text_missing=text_missing,missing_examples=missing_examples,
        eligible_images=len(candidates),selected_images=len(selected),rejections=dict(rejects),
        eligible_fraction=len(candidates)/len(groups),eligible_classes=dict(class_counts),
        selected_classes=dict(Counter(p['category'] for r in selected for p in r['phrases'])),
        weak_labels='Unique GT class instance in sealed reference-noun manifest; automatic prompt alias mapping, not human gold')
    save(out/'coverage.json',coverage)
    if not selected:
        raise RuntimeError('No eligible images. Read coverage.json: missing_original_prompt_text requires ORIGINAL prompt_cache.json, not UPT text or guessed detections. Send review ZIP for other rejection causes.')
    # Retain only this tiny subset; preserve tensors exactly, with no re-encoding as model input.
    torch.save({r['filename']:legacy[r['filename']] for r in selected},out/'selected_legacy.pt')
    cfg.update(selected=selected,policy=POLICY,input_sha256=dict(checkpoint=A_SHA,dataset_json=DATA_SHA,
        legacy_prompt=LEGACY_SHA,noun_receipt=NOUN_SHA),test_used=False)
    save(out/'protocol.json',cfg)
    print(f'Metadata complete: {len(selected)}/200 selected from {len(candidates)}/10000 eligible.',flush=True)


def build(cfg):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Expected exact A wrapper.')
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(state['caption_word2idx'])))
    base.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True)
    if model_digest(base)!=BASE_SHA:raise ValueError('Wrong A tensor identity.')
    return base.requires_grad_(False).cuda().eval()


class Probe:
    def __init__(self,model):self.model=model;self.handles=[];self.weights={}
    def __enter__(self):
        from route_probe import attention
        for name,module in [('q',self.model.encoder.qformer.layers[-1].cross_attn),('p',self.model.encoder.prompt_to_visual_attn)]:
            def observe(module,args,kwargs,name=name):
                call=dict(kwargs)
                for field,arg in zip(('query','key','value'),args):call[field]=arg
                self.weights[name]=attention(module,**call).detach()
                return None
            self.handles.append(module.register_forward_pre_hook(observe,with_kwargs=True))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()


def worker(cfg):
    import h5py
    import torch
    from transformers import CLIPTextModel
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Select T4x2; two independent read-only workers, no DDP collectives.')
    torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True)
    out=Path(cfg['output']);rows=cfg['selected'][rank::2];begin=time.monotonic()
    for name,sha in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError('Source mismatch: '+name)
    legacy=torch.load(out/'selected_legacy.pt',map_location='cpu',weights_only=False)
    # Verify text/CLIP token binding against the ORIGINAL cached vectors, not guessed prompt text.
    text_model=CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV).float().cuda().eval()
    text_max=0.
    with torch.no_grad():
        for r in rows:
            entry=legacy[r['filename']];ids=torch.tensor([r['input_ids']],device='cuda')
            mask=entry['mask'].long()[None].cuda()
            actual=text_model(input_ids=ids,attention_mask=mask).last_hidden_state[0]
            expected=entry['tokens'].float().cuda()
            text_max=max(text_max,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3,
                msg='Original text/token embedding binding differs; stop, do not replace cached embeddings.')
    del text_model;torch.cuda.empty_cache()
    model=build(cfg);cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in rows],'Frozen attention TRAIN diagnostic')
    records=[];saved_maps={};parity=[];metadata={}
    try:
        for step,r in enumerate(rows):
            file_id,_=cache.index[r['coco_id']]
            if file_id not in metadata:
                with h5py.File(cache.paths[file_id],'r') as f:
                    meta=json.loads(f.attrs['metadata_json']);metadata[file_id]=meta
                    if (meta['dataset_json_sha256']!=DATA_SHA or meta['model_id']!='openai/clip-vit-base-patch16'
                        or meta['output']!='vision_model.last_hidden_state' or meta['preprocessing']!='bilinear'
                        or meta['image_size']!=[224,224] or meta['antialias'] is not True
                        or meta['compute_dtype']!='float32' or meta['autocast'] is not False
                        or meta['mean']!=[.48145466,.4578275,.40821073] or meta['std']!=[.26862954,.26130258,.27577711]):
                        raise ValueError('Different cache preprocessing: box coordinates require A square-resize tokens.')
            entry=legacy[r['filename']];v=torch.from_numpy(cache.read(r['coco_id']))[None].cuda()
            p=entry['tokens'].float()[None].cuda();mask=entry['mask'].long()[None].cuda()
            with torch.no_grad():
                if step==0:
                    expected,pad=model.build_memory(v,p,mask)
                    prefix=torch.tensor([[1]],device='cuda');before=model.decoder(prefix,expected,memory_key_padding_mask=pad)
                with Probe(model) as probe:
                    actual,apad=model.build_memory(v,p,mask)
                if step==0:
                    after=model.decoder(prefix,actual,memory_key_padding_mask=apad)
                    if not torch.equal(expected,actual) or not torch.equal(before,after) or not torch.equal(pad,apad):
                        raise ValueError('Passive observer changed same-shape A memory/logits. No tolerance relaxation.')
                    parity.append(dict(memory_max=float((expected-actual).abs().max()),logit_max=float((before-after).abs().max())))
                q=probe.weights['q'].mean(1)[0];a=probe.weights['p'].mean(1)[0]
                if q.shape!=(32,197) or a.shape!=(len(mask[0]),32):raise ValueError('Unexpected attention interface.')
                paths=a@q;cls=paths[:,0];patch=paths[:,1:];mass=patch.sum(-1)
                if not torch.isfinite(paths).all() or bool((mass<=1e-10).any()):raise ValueError('No finite spatial path.')
                conditioned=patch/mass[:,None]
                maps=torch.stack([conditioned[t['positions']].mean(0) for t in r['phrases']]).cpu().numpy()
                cls_mass=[float(cls[t['positions']].mean()) for t in r['phrases']]
            key=str(r['coco_id']);saved_maps[key]=maps
            # Preserve the inputs needed to distinguish absent Q-region capacity from wrong prompt selection.
            saved_maps['q_full_'+key]=q.cpu().numpy()
            saved_maps['p_to_q_'+key]=a.cpu().numpy()
            record=dict(coco_id=r['coco_id'],filename=r['filename'],prompt=r['prompt'],phrases=r['phrases'],pairs=r['pairs'],
                cls_mass=cls_mass,metrics=evaluate_image(maps,r['phrases'],r['pairs'],key))
            records.append(record)
            if step%10==0:print(f'GPU{rank}: {step+1}/{len(rows)} images observed; no training',flush=True)
        if model_digest(model)!=BASE_SHA or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen A changed.')
    finally:cache.close()
    np.savez_compressed(out/f'maps_rank{rank}.npz',**saved_maps)
    save(out/f'rows_rank{rank}.json',records)
    save(out/f'complete_rank{rank}.json',dict(status='FROZEN_FORWARD_COMPLETE',rank=rank,images=len(records),
        checkpoint_sha256=A_SHA,base_tensor_sha256=BASE_SHA,parity=parity,text_binding_max_abs=text_max,
        cache_metadata=metadata,runtime=runtime_manifest(),seconds=time.monotonic()-begin,optimizer_used=False))


def finish(cfg):
    out=Path(cfg['output']);rows=[]
    for rank in (0,1):
        complete=json.loads((out/f'complete_rank{rank}.json').read_text())
        part=json.loads((out/f'rows_rank{rank}.json').read_text())
        if complete['status']!='FROZEN_FORWARD_COMPLETE' or complete['images']!=len(part):raise ValueError('Incomplete rank.')
        rows.extend(part)
    expected=[r['coco_id'] for r in cfg['selected']]
    if len(rows)!=len(expected) or sorted(r['coco_id'] for r in rows)!=sorted(expected):raise ValueError('Incomplete/duplicate cohort.')
    result=aggregate(rows);save(out/'result.json',result)
    (out/'RESULT.md').write_text('# Frozen-A attention geometry diagnostic\n\n'+result['status']+'\n\n'+
        json.dumps(result,ensure_ascii=False,indent=2)+'\n\nTRAIN diagnostic only. No caption gain or novelty established.\n',encoding='utf-8')
    save(out/'receipt.json',dict(status='DIAGNOSTIC_COMPLETE',result=result['status'],images=len(rows),
        payload_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('receipt.json','selected_legacy.pt')},
        optimizer_used=False,test_used=False))
    print('COMPLETE:',result['status'],'; send review ZIP, no checkpoint was created.',flush=True)


def archive(cfg):
    out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.md','.log','.py','.npz'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','finish','archive']);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    globals()[args.mode](cfg)
