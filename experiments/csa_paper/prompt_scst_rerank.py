"""USER-RUN frozen CLIP; reuse fixed164 selector, no BLIP or reference-based selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from prompt_scst_data import PHASES,validate_candidates,DATA_SHA,DETECTION_SHA
from captioning.multiscore_reranking import rerank_multiscore


def validation_image_paths(root,records):
    """One directory scan instead of four mounted-file probes per image."""
    root=Path(root).resolve();wanted={r['filename'] for r in records};found={};pending=[root];scanned=0
    print('JPEG path scan START:',root,'wanted',len(wanted),flush=True)
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                scanned+=1
                if entry.name in wanted:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):raise ValueError('Expected regular JPEG:'+entry.path)
                    if entry.name in found:raise ValueError('Ambiguous JPEG filename:'+entry.name)
                    found[entry.name]=Path(entry.path)
                elif entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
                if scanned%25000==0:print('JPEG path scan:',scanned,'entries;',len(found),'/',len(wanted),'found',flush=True)
    missing=wanted-set(found)
    if missing:raise FileNotFoundError('Missing JPEGs under '+str(root)+': '+', '.join(sorted(missing)[:5]))
    print('JPEG paths COMPLETE:',len(found),flush=True)
    return found


def validation_detections(source,names):
    from pathlib import PurePosixPath
    from captioning.object_consistency import index_detections
    # The full file SHA is still checked; only VALID entries need object indexing.
    subset={k:v for k,v in source.items() if PurePosixPath(str(k).replace('\\','/')).name in names}
    indexed=index_detections(subset)
    if set(indexed)!=set(names):raise ValueError('Missing VALID detector records')
    return indexed


def checked_jpeg(item):
    from PIL import Image
    i,path,detected=item
    if any(not np.isfinite(s) or not .5<=s<=1 for s in detected.values()):raise ValueError('Invalid detector confidence')
    raw=path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:image.verify()
    return dict(coco_id=i,image_path=str(path),image_sha256=hashlib.sha256(raw).hexdigest(),detected=detected)


def prepare_images(val,paths,detected,workers=8):
    started=time.monotonic();inputs=[]
    items=((i,paths[val[i]['filename']],detected[val[i]['filename']]) for i in sorted(val))
    print('JPEG hash/verify START:',len(val),'images,',workers,'I/O threads',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(checked_jpeg,items):
            inputs.append(row)
            if len(inputs)%100==0 or len(inputs)==len(val):print('JPEG hash/verify:',len(inputs),'/',len(val),'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    return inputs


def fixed_rank(row,scores,detected):
    if len(scores)!=5 or not np.isfinite(scores).all() or any(not 0<=s<=2.50001 for s in scores):raise ValueError('Invalid CLIP scores')
    enriched=[dict(c,clipscore=float(s)) for c,s in zip(row['candidates'],scores)]
    selected,_=rerank_multiscore(dict(candidates=enriched),detected,.4,.3,hallucination_penalty=1.)
    return selected['original_rank']


def prepare(cfg):
    out=Path(cfg['output']);started=time.monotonic()
    for key,h in (('dataset_json',DATA_SHA),('detections',DETECTION_SHA)):
        print('Rerank preparation HASH START:',key,flush=True);began=time.monotonic()
        if sha(cfg[key])!=h:raise ValueError('Changed input:'+key)
        print('Rerank preparation HASH PASS:',key,round(time.monotonic()-began,1),'s',flush=True)
    print('Loading COCO metadata; VALID5000 only.',flush=True)
    val={int(Path(r['filename']).stem.split('_')[-1]):r for r in read(cfg['dataset_json'])['images'] if r['split']=='val'}
    refs=read(out/'validation_references.json')
    if len(val)!=5000 or set(val)!={r['coco_id'] for r in refs}:raise ValueError('Wrong VALID5000')
    if any([s['raw'] for s in val[r['coco_id']]['sentences']]!=r['references'] for r in refs):raise ValueError('VALID reference provenance differs')
    print('Loading detector JSON and indexing5000VALID entries.',flush=True)
    detected=validation_detections(read(cfg['detections']),{r['filename'] for r in val.values()})
    print('VALID detector index COMPLETE:',len(detected),flush=True)
    paths=validation_image_paths(cfg['coco_root'],list(val.values()))
    inputs=prepare_images(val,paths,detected)
    save(out/'rerank_inputs.json',inputs) # Images/detections only; no references or quality labels.
    save(out/'rerank_prepared.json',dict(input_sha256=sha(out/'rerank_inputs.json'),references_sha256=sha(out/'validation_references.json'),
        dataset_sha256=DATA_SHA,detector_sha256=DETECTION_SHA,images=5000,test_used=False,
        selector='.3minmax(raw decoder)+.4minmax(CLIPScore)+.3normalized detector consistency; earlier ties'))
    print('VALID JPEG/detector inputs checked before training;',round(time.monotonic()-started,1),'seconds.',flush=True)


def prefetch(cfg):
    from huggingface_hub import snapshot_download
    from transformers import CLIPProcessor
    locked=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip']
    root=Path(snapshot_download(locked['id'],revision=locked['revision'],allow_patterns=list(locked['files_sha256']),cache_dir=cfg['model_cache']))
    for n,h in locked['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('CLIP assets differ from frozen164:'+n)
    CLIPProcessor.from_pretrained(root,local_files_only=True,use_fast=False)
    save(Path(cfg['output'])/'clip_assets.json',dict(**locked,path=str(root)))
    print('Same CLIP model/processor bytes as164. No BLIP.',flush=True)


def candidate_sets(out,expected):
    result={}
    for phase in PHASES:
        d=Path(out)/'evaluations'/phase;rows=read(d/'candidates_rank0.json')+read(d/'candidates_rank1.json');validate_candidates(rows,expected)
        result[phase]={r['coco_id']:r for r in rows}
    return result


def worker(cfg):
    import torch
    from torch.nn import functional as F
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);prepared=read(out/'rerank_prepared.json')
    if sha(out/'rerank_inputs.json')!=prepared['input_sha256']:raise ValueError('Changed rerank inputs')
    inputs=read(out/'rerank_inputs.json');candidates=candidate_sets(out,{r['coco_id'] for r in inputs});asset=read(out/'clip_assets.json')
    locked=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip']
    if {k:v for k,v in asset.items() if k!='path'}!=locked:raise ValueError('Wrong CLIP asset receipt')
    for n,h in locked['files_sha256'].items():
        if sha(Path(asset['path'])/n)!=h:raise ValueError('Changed CLIP asset:'+n)
    model=CLIPModel.from_pretrained(asset['path'],local_files_only=True,use_safetensors=False,attn_implementation='eager').float().to(device).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(asset['path'],local_files_only=True,use_fast=False);started=time.monotonic();count=0
    path=out/f'rerank_scores_rank{rank}.jsonl'
    with path.open('x',encoding='utf-8') as f,torch.inference_mode():
        for row in inputs[rank::2]:
            if time.monotonic()-started>cfg['rerank_seconds']:raise TimeoutError('Rerank cap; saved captions/weights preserved')
            raw=Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Changed JPEG')
            with Image.open(io.BytesIO(raw)) as image:image=image.convert('RGB')
            pixels=processor(images=image,return_tensors='pt')['pixel_values'].to(device);image_feature=F.normalize(model.get_image_features(pixel_values=pixels).float(),dim=-1)
            points={}
            for phase in PHASES:
                beam=candidates[phase][row['coco_id']];text=[c['caption'] for c in beam['candidates']]
                tokens=processor.tokenizer(text,padding=True,truncation=False,return_tensors='pt')
                if tokens['input_ids'].shape[1]>77:raise ValueError('Would truncate candidate text')
                features=F.normalize(model.get_text_features(**tokens.to(device)).float(),dim=-1)
                scores=(2.5*(features*image_feature).sum(-1).clamp_min(0)).cpu().tolist()
                points[phase]=dict(clipscore=scores,selected_rank=fixed_rank(beam,scores,row['detected']))
            f.write(json.dumps(dict(coco_id=row['coco_id'],image_sha256=row['image_sha256'],phases=points),allow_nan=False)+'\n');f.flush();count+=1
            if count%100==0:print('CLIP rerank',rank,count,'/2500, ORIGINAL+END',flush=True)
            if count==25 and (time.monotonic()-started)/count*(2500-count)>cfg['rerank_seconds']-(time.monotonic()-started):raise TimeoutError('Measured CLIP projection exceeds cap')
    save(out/f'rerank_worker{rank}.json',dict(complete=True,images=count,candidate_pairs=count*10,scores_sha256=sha(path),seconds=time.monotonic()-started,training=False))


def score(cfg):
    from run_scst_pilot import score as score_phase
    out=Path(cfg['output']);prepared=read(out/'rerank_prepared.json');inputs=read(out/'rerank_inputs.json')
    if sha(out/'rerank_inputs.json')!=prepared['input_sha256'] or sha(out/'validation_references.json')!=prepared['references_sha256']:raise ValueError('Changed evaluation inputs')
    candidates=candidate_sets(out,{r['coco_id'] for r in inputs});scores={}
    for rank in (0,1):
        worker=read(out/f'rerank_worker{rank}.json');path=out/f'rerank_scores_rank{rank}.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if not worker['complete'] or worker['images']!=2500 or worker['candidate_pairs']!=25000 or sha(path)!=worker['scores_sha256']:raise ValueError('Incomplete rerank worker')
        if [r['coco_id'] for r in rows]!=[r['coco_id'] for r in inputs[rank::2]]:raise ValueError('Rerank IDs/order differ')
        scores.update({r['coco_id']:r for r in rows})
    metrics=read(out/'training_result.json')['metrics'];result={}
    for phase,step in PHASES.items():
        selected={};ranks={}
        for row in inputs:
            s=scores[row['coco_id']];p=s['phases'][phase];beam=candidates[phase][row['coco_id']]
            if s['image_sha256']!=row['image_sha256']:raise ValueError('Wrong image-score identity')
            k=fixed_rank(beam,p['clipscore'],row['detected'])
            if k!=p['selected_rank']:raise ValueError('Fixed selector replay differs')
            ranks[str(row['coco_id'])]=k;selected[row['coco_id']]=beam['candidates'][k]['caption']
        source=out/'evaluations'/phase;d=out/'reranked'/phase;d.mkdir(parents=True,exist_ok=True)
        for rank in (0,1):
            rows=read(source/f'val_rank{rank}.json')
            save(d/f'val_rank{rank}.json',[dict(coco_id=r['coco_id'],caption=selected[r['coco_id']]) for r in rows])
            shutil.copy2(source/f'worker{rank}.json',d/f'worker{rank}.json')
        for n in ('vocabulary.json','validation_references.json'):shutil.copy2(out/n,d/n)
        score_phase({**cfg,'output':str(d),'study':'full1','evaluation_step':step})
        save(d/'selected_ranks.json',ranks)
        result[phase]=dict(decoder_rank0=metrics[phase],fixed_rerank=read(d/'result.json')['metrics'],changed_captions=sum(k!=0 for k in ranks.values()))
    save(out/'result.json',dict(status='PROMPT_SCST_RERANK_VALID_COMPLETE',arm=cfg['arm'],source=read(out/'prepared.json')['source'],
        updates=14161,image_events=113287,validation_images=5000,metrics=result,test_used=False,oracle_used=False,
        selection='fixed END, both policies reported; no tuned weights or posthoc fallback',
        NLL_note='Same teacher-forced model NLL within a phase; reranking changes output selection only.',
        limitations=['One training seed and reused VALID.', 'Selected XE sources C10/P9 have unequal initial training epochs.',
            'CHAIR checks objects, not relation correctness; detector absence is not proven absence.']))
    print('SCST + fixed rerank COMPLETE',cfg['arm'],result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','prefetch','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
