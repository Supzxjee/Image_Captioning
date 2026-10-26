"""User-run pixel/text counterfactual preparation. Training pilot is not implemented here."""
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from diagnostic_data import A_SHA,BASE_SHA,DATA_SHA,LEGACY_SHA,CLIP_REV,digest,save,prompt_text,text_index
from stale_data import *


def decode_segmentation(seg,h,w):
    from pycocotools import mask as api
    if isinstance(seg,list):rle=api.merge(api.frPyObjects(seg,h,w))
    elif isinstance(seg,dict) and isinstance(seg.get('counts'),list):rle=api.frPyObjects(seg,h,w)
    else:rle=seg
    value=api.decode(rle)
    return value.any(-1) if value.ndim==3 else value


def annotations(cfg,ids):
    import ijson
    from urllib.request import urlopen
    path=Path(cfg['annotation_archive'])
    if not path.is_file():
        path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.partial')
        print('Downloading official COCO annotation archive only (~253MB).',flush=True)
        with urlopen(ANNOTATION_URL,timeout=90) as source,part.open('wb') as target:
            while block:=source.read(4*1024**2):target.write(block)
        part.replace(path)
    if path.stat().st_size!=ANNOTATION_BYTES or digest(path)!=ANNOTATION_SHA:raise ValueError('Official annotation archive differs.')
    images={};byclass=defaultdict(list);cats=None
    with zipfile.ZipFile(path) as z:
        for split in ('train','val'):
            member=f'annotations/instances_{split}2014.json'
            with z.open(member) as f:current=list(ijson.items(f,'categories.item',use_float=True))
            if cats is None:cats={c['id']:c['name'] for c in current}
            elif cats!={c['id']:c['name'] for c in current}:raise ValueError('COCO category mismatch.')
            with z.open(member) as f:
                for r in ijson.items(f,'images.item',use_float=True):
                    if r['id'] in ids:images[r['id']]=r
            with z.open(member) as f:
                for a in ijson.items(f,'annotations.item',use_float=True):
                    if a['image_id'] in ids:byclass[(a['image_id'],cats[a['category_id']])].append(a)
    if set(images)!=set(ids):raise ValueError('Missing TRAIN10k annotation images.')
    return images,byclass


def prepare(cfg):
    import torch
    from captioning.tokenizer import CaptionTokenizer
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_consistency import ALIASES
    from hidden_data import locked_groups
    from audit import AMBIGUOUS
    from PIL import Image
    from transformers import CLIPTokenizerFast
    from diagnostic_data import phrase_positions
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for k,h in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA),('prompt_text_json',TEXT_SHA)]:
        print('Hashing input:',k,flush=True)
        if digest(cfg[k])!=h:raise ValueError('Pinned input differs: '+k)
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));groups=locked_groups(data)
    tok=CaptionTokenizer([s['raw'] for r in data['images'] if r['split'] in ('train','restval') for s in r['sentences'][:5]])
    if hashlib.sha256(json.dumps(tok.word2idx,sort_keys=True).encode()).hexdigest()!='427a33573c9302661282dd4553e87ccd82138f5903cea25823afc7eddb49bd3c':raise ValueError('Caption vocabulary/order differs from A.')
    texts=text_index(json.loads(Path(cfg['prompt_text_json']).read_text(encoding='utf-8')))
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],[r['filename'] for r in groups['train']+groups['val']])
    if len(legacy)!=11000:raise ValueError('Missing original prompt coverage.')
    lexicon=lexical_ids(tok.word2idx,ALIASES,AMBIGUOUS)
    clip_tokenizer=CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV)
    image_info,byclass=annotations(cfg,{r['coco_id'] for r in groups['train']})
    rejects=Counter();candidates=[]
    for r in groups['train']:
        image=data['images'][r['image_id']];text=prompt_text(legacy[r['filename']]) or texts.get(r['filename'])
        if not text:rejects['missing_original_text']+=1;continue
        encoded_prompt=clip_tokenizer(text,max_length=20,padding='max_length',truncation=True,return_offsets_mapping=True)
        window=visible_window(text,encoded_prompt)
        window_tokens=clip_tokenizer(window,max_length=20,padding='max_length',truncation=True)
        if window_tokens['input_ids']!=encoded_prompt['input_ids'] or window_tokens['attention_mask']!=encoded_prompt['attention_mask']:
            rejects['visible_window_not_exact_token_replay']+=1;continue
        visible={p['category'] for p in phrase_positions(text,ALIASES,encoded_prompt)}
        options=[];im=image_info[r['coco_id']]
        for label in sorted({c for a,b,c in mentions(text,ALIASES)}):
            if label not in visible:rejects['target_prompt_truncated']+=1;continue
            if label not in lexicon:rejects['no_safe_single_token_alias']+=1;continue
            reference=None
            for ci,s in enumerate(image['sentences'][:5]):
                target=reference_target(s['raw'],label,ALIASES,tok.word2idx,lexicon[label])
                if target is not None:reference=dict(caption_index=ci,caption=s['raw'],**target);break
            if reference is None:rejects['no_first_single_token_reference']+=1;continue
            anns=byclass[(r['coco_id'],label)]
            try:m=union_mask(anns,im['height'],im['width'],decode_segmentation)
            except (ValueError,TypeError,KeyError):m=None
            if m is None:rejects['invalid_crowd_missing_or_area_mask']+=1;continue
            neutral=neutralize(window,label,ALIASES)
            if not neutral or neutral==window:raise ValueError('Neutralization failed.')
            if len(clip_tokenizer(neutral,truncation=False)['input_ids'])>20:
                rejects['neutral_window_exceeds_budget']+=1;continue
            neutral_tokens=clip_tokenizer(neutral,max_length=20,padding='max_length',truncation=True,return_offsets_mapping=True)
            options.append(dict(r,target=label,reference=reference,alias_ids=lexicon[label],original_text=text,visible_original_text=window,neutral_text=neutral,
                original_tokenization=dict(encoded_prompt),neutral_tokenization=dict(neutral_tokens),
                image_size=[im['width'],im['height']],annotations=anns,mask_fraction=float(m.mean()),
                annotation_ids=[a['id'] for a in anns],filepath=image['filepath']))
        if options:candidates.append(min(options,key=lambda o:order(f"{o['coco_id']}:{o['target']}")))
        else:rejects['image_without_valid_target']+=1
    selected=sorted(candidates,key=lambda r:order(r['coco_id']))[:POLICY['max_aux_images']]
    save(out/'manifest.json',dict(policy=POLICY,rows=selected,eligible_images=len(candidates),rejections=dict(rejects),
        annotation_sha256=ANNOTATION_SHA,original_text_sha256=TEXT_SHA,test_used=False))
    if len(selected)<128:raise RuntimeError('Too few eligible images for the fixed preparation panel/score sample; no policy relaxation.')
    # Validate pixels exist and have annotation dimensions before GPU work.
    for r in selected:
        path=Path(cfg['images'])/r['filepath']/r['filename']
        with Image.open(path) as image:
            if list(image.size)!=r['image_size']:raise ValueError('Pixel/annotation image-size mismatch.')
    main=np.random.default_rng(42).permutation(len(groups['train']))[:400]
    main_rows=[groups['train'][int(i)] for i in main];eval_rows=groups['val'][:64]
    names={r['filename'] for r in selected+main_rows+eval_rows}
    torch.save({n:dict(tokens=legacy[n]['tokens'].clone(),mask=legacy[n]['mask'].clone()) for n in names},out/'legacy_subset.pt')
    import h5py
    from captioning.visual_cache import VisualCache
    cache=VisualCache([cfg['visual_cache']])
    try:
        with h5py.File(out/'benchmark_inputs.h5','w') as f:
            for split,benchrows in [('main',main_rows),('eval',eval_rows)]:
                g=f.create_group(split)
                v=g.create_dataset('visual',(len(benchrows),197,768),'f4');p=g.create_dataset('prompt',(len(benchrows),20,512),'f4')
                m=g.create_dataset('mask',(len(benchrows),20),'u1');caps=[];pads=[]
                for i,r in enumerate(benchrows):
                    v[i]=cache.read(r['coco_id']);p[i]=legacy[r['filename']]['tokens'].float().numpy();m[i]=legacy[r['filename']]['mask'].numpy()
                    encoded=[tok.encode(s['raw']) for s in data['images'][r['image_id']]['sentences'][:5]]
                    caps.append([e[0].tolist() for e in encoded]);pads.append([e[1].tolist() for e in encoded])
                g.create_dataset('captions',data=np.asarray(caps,dtype='i8'));g.create_dataset('padding',data=np.asarray(pads,dtype=bool))
            f.attrs['complete']=True
    finally:cache.close()
    cfg.update(rows=selected,main_rows=main_rows,eval_rows=eval_rows,word2idx=tok.word2idx,
        dataset_groups=groups,policy=POLICY,metadata_seconds=time.monotonic()-start,
        manifest_sha256=digest(out/'manifest.json'),legacy_subset_sha256=digest(out/'legacy_subset.pt'),benchmark_sha256=digest(out/'benchmark_inputs.h5'))
    save(out/'protocol.json',cfg)
    print(f'Manifest ready: {len(selected)}/{len(candidates)} eligible TRAIN images; all instances per target retained.',flush=True)


def cache(cfg):
    import torch
    import h5py
    from PIL import Image,ImageDraw
    from transformers import CLIPVisionModel,CLIPTextModel,CLIPTokenizer
    from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,runtime_manifest
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True)
    out=Path(cfg['output']);start=time.monotonic();rows=cfg['rows'][rank::2]
    if digest(out/'manifest.json')!=cfg['manifest_sha256'] or digest(out/'legacy_subset.pt')!=cfg['legacy_subset_sha256']:raise ValueError('Prepared inputs changed.')
    legacy=torch.load(out/'legacy_subset.pt',map_location='cpu',weights_only=False)
    vision=CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV).float().cuda().eval()
    textmodel=CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV).float().cuda().eval()
    tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16',revision=CLIP_REV)
    transform=Compose([Resize((224,224),interpolation=InterpolationMode.BILINEAR,antialias=True),ToTensor(),
        Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
    visual=VisualCache([cfg['visual_cache']]);checked=set();panel_ids={r['coco_id'] for r in cfg['rows'][:POLICY['panel_images']]}
    max_text_error=0.;max_image_error=0.;output=out/f'aux_rank{rank}.h5'
    try:
        with torch.no_grad(),h5py.File(output,'w') as f:
            arrays={n:f.create_dataset(n,shape=shape,dtype=dtype) for n,shape,dtype in [
                ('visual_original',(len(rows),197,768),'f4'),('visual_occluded',(len(rows),197,768),'f4'),
                ('prompt_original',(len(rows),20,512),'f4'),('prompt_neutral',(len(rows),20,512),'f4'),
                ('mask_original',(len(rows),20),'u1'),('mask_neutral',(len(rows),20),'u1')]}
            f.create_dataset('coco_ids',data=[r['coco_id'] for r in rows])
            for i,r in enumerate(rows):
                if cfg['metadata_seconds']+time.monotonic()-start>POLICY['max_preparation_seconds']:raise RuntimeError('Preparation exceeded30minute wall-clock budget; preserve partial output, no training.')
                fid,_=visual.index[r['coco_id']]
                if fid not in checked:
                    with h5py.File(visual.paths[fid],'r') as source:
                        meta=json.loads(source.attrs['metadata_json'])
                        if (meta['dataset_json_sha256']!=DATA_SHA or meta['image_size']!=[224,224] or meta['preprocessing']!='bilinear'
                            or meta['model_id']!='openai/clip-vit-base-patch16' or meta['output']!='vision_model.last_hidden_state'
                            or meta['mean']!=[.48145466,.4578275,.40821073] or meta['std']!=[.26862954,.26130258,.27577711]
                            or meta['antialias'] is not True or meta['compute_dtype']!='float32' or meta['autocast'] is not False):raise ValueError('Different original visual-cache preprocessing.')
                    checked.add(fid)
                original=visual.read(r['coco_id']);old=legacy[r['filename']]
                inputs=tokenizer([r['original_text'],r['neutral_text']],max_length=20,padding='max_length',truncation=True,return_tensors='pt').to('cuda')
                for index,key in [(0,'original_tokenization'),(1,'neutral_tokenization')]:
                    if inputs.input_ids[index].cpu().tolist()!=r[key]['input_ids'] or inputs.attention_mask[index].cpu().tolist()!=r[key]['attention_mask']:raise ValueError('Stored fast offsets/token IDs differ from encoding tokenizer.')
                if not torch.equal(inputs.attention_mask[0].cpu(),old['mask'].long()):raise ValueError('Original text-token mask differs.')
                encoded=textmodel(**inputs).last_hidden_state
                max_text_error=max(max_text_error,float((encoded[0].cpu()-old['tokens'].float()).abs().max()))
                torch.testing.assert_close(encoded[0].cpu(),old['tokens'].float(),rtol=.002,atol=.002)
                with Image.open(Path(cfg['images'])/r['filepath']/r['filename']) as im:image=im.convert('RGB')
                mask=union_mask(r['annotations'],image.height,image.width,decode_segmentation)
                if mask is None:raise ValueError('Selected union mask no longer valid.')
                pixels=np.asarray(image).copy();pixels[mask]=POLICY['mask_fill'];occluded=Image.fromarray(pixels)
                occluded_input=transform(occluded)
                if r['coco_id'] in panel_ids:
                    original_input=transform(image)
                    factual=vision(pixel_values=original_input[None].cuda()).last_hidden_state[0].cpu()
                    max_image_error=max(max_image_error,float((factual-torch.from_numpy(original)).abs().max()))
                    torch.testing.assert_close(factual,torch.from_numpy(original),rtol=.002,atol=.002)
                    # Save full-context images, not only crops; reviewer can inspect unintended occlusion.
                    panel=Image.new('RGB',(448,224));panel.paste(image.resize((224,224),Image.Resampling.BILINEAR),(0,0));panel.paste(occluded.resize((224,224),Image.Resampling.BILINEAR),(224,0))
                    (out/'panel').mkdir(exist_ok=True);panel.save(out/'panel'/f"{r['coco_id']}.png")
                    Image.fromarray(mask.astype('uint8')*255).save(out/'panel'/f"{r['coco_id']}_mask.png")
                    save(out/'panel'/f"{r['coco_id']}_pixels.json",dict(image_size=list(image.size),masked_pixels=int(mask.sum()),mask_fraction=float(mask.mean()),
                        original_CLIP_input_sha256=hashlib.sha256(original_input.numpy().tobytes()).hexdigest(),
                        occluded_CLIP_input_sha256=hashlib.sha256(occluded_input.numpy().tobytes()).hexdigest(),
                        display='224x224 bilinear RGB panels are the pre-normalization images used by CLIP; mask PNG retains original dimensions'))
                masked=vision(pixel_values=occluded_input[None].cuda()).last_hidden_state[0].cpu().numpy()
                for name,value in [('visual_original',original),('visual_occluded',masked),('prompt_original',old['tokens'].float().numpy()),
                    ('prompt_neutral',encoded[1].cpu().numpy()),('mask_original',old['mask'].numpy()),('mask_neutral',inputs.attention_mask[1].cpu().numpy())]:
                    if not np.isfinite(value).all():raise ValueError('Nonfinite cache value.')
                    arrays[name][i]=value
                if i%50==0:print(f'GPU{rank}: counterfactual cache {i+1}/{len(rows)}',flush=True)
            f.attrs['complete']=True
    finally:visual.close()
    save(out/f'cache_rank{rank}.json',dict(status='CACHE_COMPLETE',images=len(rows),file=output.name,sha256=digest(output),
        bytes=output.stat().st_size,seconds=time.monotonic()-start,original_text_max_error=max_text_error,
        original_image_panel_max_error=max_image_error,runtime=runtime_manifest()))


def archive(cfg):
    out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.py','.md','.html','.png'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip','; KEEP ENTIRE FOLDER:',out,flush=True)


def finish(cfg):
    import html
    out=Path(cfg['output']);ranks=[];scores=[];profiles=[]
    for rank in (0,1):
        receipt=json.loads((out/f'cache_rank{rank}.json').read_text());file=out/receipt['file']
        if receipt['status']!='CACHE_COMPLETE' or digest(file)!=receipt['sha256']:raise ValueError('Incomplete/changed auxiliary cache.')
        ranks.append(receipt);scores+=json.loads((out/f'four_scores_rank{rank}.json').read_text())
        profiles.append(json.loads((out/f'benchmark_rank{rank}.json').read_text()))
    if sum(r['images'] for r in ranks)!=len(cfg['rows']) or len(scores)!=128:raise ValueError('Cache/score coverage differs.')
    if {r['coco_id'] for r in scores}!={r['coco_id'] for r in cfg['rows'][:128]}:raise ValueError('Fixed score panel changed.')
    if profiles[0]['arms']!=profiles[1]['arms']:raise ValueError('Rank profiles differ.')
    score_summary={k:dict(mean=float(np.mean([r[k] for r in scores])),median=float(np.median([r[k] for r in scores])),
        positive=sum(r[k]>0 for r in scores)) for k in ('factual_prompt_boost','occluded_prompt_boost','visual_contrast_original','visual_contrast_neutral','stale_hinge')}
    by_id={r['coco_id']:r for r in cfg['rows']};strata={}
    for s in scores:
        r=by_id[s['coco_id']];area=r['mask_fraction'];size='small' if area<.05 else 'medium' if area<.2 else 'large'
        for key in ('class:'+r['target'],'mask_size:'+size):strata.setdefault(key,[]).append(s['occluded_prompt_boost'])
    prevalence={k:dict(images=len(v),mean=float(np.mean(v)),fraction_above_margin=float(np.mean(np.asarray(v)>.1))) for k,v in strata.items()}
    boost=np.asarray([s['occluded_prompt_boost'] for s in scores])
    prep=cfg['metadata_seconds']+max(r['seconds'] for r in ranks)
    costs_ok=prep<=1800 and all(p['within_2h'] and p['estimated_memory_within_T4_headroom'] for p in profiles[0]['arms'].values())
    cards=[]
    for r in cfg['rows'][:16]:
        cards.append(f"<article><h2>{r['coco_id']} · {html.escape(r['target'])}</h2><img src='panel/{r['coco_id']}.png'><p>Left: original. Right: union mask of {len(r['annotations'])} instances; area {r['mask_fraction']:.1%}.</p><p>P1 visible window: {html.escape(r['visible_original_text'])}</p><p>P0: {html.escape(r['neutral_text'])}</p><p>Reference (factual only): {html.escape(r['reference']['caption'])}</p></article>")
    (out/'panel.html').write_text('<!doctype html><meta charset="utf-8"><title>Four-condition preparation panel</title><style>body{font:17px Arial;max-width:1000px;margin:30px auto}article{padding:18px;border-bottom:1px solid #bbb}img{width:896px;max-width:100%}</style><h1>Fixed16 metadata-selected panel — review before any pilot</h1><p>Occlusion is not proof of absence. Check all target instances, unintended context loss and neutralized wording. No training pilot has run.</p>'+''.join(cards),encoding='utf-8')
    summary=dict(status='PREPARATION_COMPLETE_WAIT_PANEL_AND_ARGON',images=len(cfg['rows']),score_images=128,
        target_classes=dict(Counter(r['target'] for r in cfg['rows'])),
        mask_fraction=dict(min=min(r['mask_fraction'] for r in cfg['rows']),max=max(r['mask_fraction'] for r in cfg['rows']),mean=float(np.mean([r['mask_fraction'] for r in cfg['rows']]))),
        costs_within_proposed_budget=costs_ok,metadata_and_cache_seconds=prep,benchmark=profiles[0]['arms'],
        frozen_A_four_condition_scores=score_summary,stale_boost_prevalence=dict(images=len(scores),active=int((boost>.1).sum()),fraction_above_point1=float((boost>.1).mean()),
            quantiles=np.quantile(boost,[0,.25,.5,.75,1]).tolist(),strata=prevalence),cache_bytes=sum(r['bytes'] for r in ranks),
        forward_backward_profile_batches=50,optimizer_steps=0,full_pilot_run=False,saved_checkpoint=False,
        interpretation='No scientific GO from score summaries or cost gate. Manual mask/text panel and Argon/Astra critique remain; no C/T pilot automatically starts.',
        limits=['Mask artifacts/context loss, incomplete COCO annotations, indirect language cues and prompt abandonment remain risks.',
            'Profile extrapolation includes safety/scoring reserve but is not a measured1250-update ETA.',
            'Original image re-encoding parity checked on fixed16, original text binding on all selected images.',
            'Four-condition scores use the first128 hash-selected TRAIN examples, not held-out caption quality.'])
    save(out/'summary.json',summary)
    save(out/'receipt.json',dict(status='PREPARATION_COMPLETE',images=len(cfg['rows']),full_pilot_run=False,
        payload_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('receipt.json','legacy_subset.pt')},
        panel_sha256={p.name:digest(p) for p in sorted((out/'panel').iterdir()) if p.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','cache','finish','archive']);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    for name,sha in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError('Runtime source changed: '+name)
    globals()[a.mode](cfg)
