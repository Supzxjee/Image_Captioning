"""User-run two-GPU inference benchmark, not a training notebook."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import time
import zipfile
from candidate_data import *


def verify(cfg):
    for name,sha in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError('Source differs: '+name)


def prepare(cfg):
    import h5py
    import numpy as np
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_consistency import index_detections
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for name,sha in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA),('detections',DET_SHA),('labels_archive',ARCHIVE_SHA)]:
        print('Hashing pinned input:',name,flush=True)
        if digest(cfg[name])!=sha:raise ValueError('Wrong input: '+name)
    with zipfile.ZipFile(cfg['labels_archive']) as z:
        if hashlib.sha256(z.read('receipt.json')).hexdigest()!=RECEIPT_SHA:raise ValueError('Wrong label receipt.')
        receipt=json.loads(z.read('receipt.json'))
        for n,sha in receipt['payload_sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=sha:raise ValueError('Wrong label payload.')
        groups=json.loads(z.read('groups.json'));targets=json.loads(z.read('targets.json'));classes=receipt['classes']
    dataset=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));selected=sample_rows(groups,dataset)
    cfg.update(sample=selected,train_ids=[r['image_id'] for r in groups['train']],test_used=False,
               input_sha256=dict(checkpoint=A_SHA,dataset_json=DATA_SHA,legacy_prompt=LEGACY_SHA,detections=DET_SHA,labels_archive=ARCHIVE_SHA))
    policy=json.loads(Path('lexical_policy.json').read_text());detector=index_detections(json.loads(Path(cfg['detections']).read_text()),min_confidence=.9)
    labels={}
    for r in selected:
        target=targets[str(r['image_id'])];refs=[s['raw'] for s in dataset['images'][r['image_id']]['sentences'][:5]]
        if target['coco_id']!=r['coco_id']:raise ValueError('Label identity differs.')
        support={c for c,y in zip(classes,target['y']) if y};uncertain=set();counts=Counter()
        for ref in refs:
            good,bad=mentions(ref,ALIASES,policy);counts.update(good);uncertain|=bad
        if r['filename'] not in detector:raise ValueError('Detector row missing.')
        labels[str(r['image_id'])]=dict(support=sorted(support),salient=sorted(c for c,n in counts.items() if n>=2 and c in support),
            conflict=bool(set(detector[r['filename']])-support),uncertain_reference_objects=sorted(uncertain-support))
    save(out/'labels_train_sample.json',labels)
    names=[str(Path('/images')/dataset['images'][r['image_id']]['filepath']/r['filename']) for r in selected]
    print('Loading legacy once; staging only 2000 TRAIN images.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False);legacy=align_prompt_cache(bundle['data'],names);length=legacy[names[0]]['tokens'].shape[0]
    cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in selected],'TRAIN2000')
    byfile=defaultdict(list)
    for i,r in enumerate(selected):
        file_id,pos=cache.index[r['coco_id']];byfile[file_id].append((pos,i))
    stage=Path(cfg['stage'])
    if stage.exists():raise FileExistsError('Preserve old stage; start fresh.')
    import shutil
    if shutil.disk_usage(stage.parent).free<2000*(197*768*2+length*512*4+length)+1024**3:raise RuntimeError('Not enough staging disk plus1GiB reserve.')
    try:
        with h5py.File(stage,'w') as f:
            arrays={k:f.create_dataset(k,shape=shape,dtype=dtype) for k,shape,dtype in [('visual',(2000,197,768),'f2'),('legacy',(2000,length,512),'f4'),('mask',(2000,length),'u1')]}
            f.create_dataset('coco_ids',data=np.asarray([r['coco_id'] for r in selected],dtype='i8'));done=0
            for file_id,positions in sorted(byfile.items()):
                with h5py.File(cache.paths[file_id],'r') as source:
                    meta=json.loads(source.attrs['metadata_json']);features=source['features']
                    if meta['dataset_json_sha256']!=DATA_SHA or meta['model_id']!='openai/clip-vit-base-patch16' or meta['output']!='vision_model.last_hidden_state' or meta['preprocessing']!='bilinear' or meta['image_size']!=[224,224] or meta['antialias'] is not True or meta['mean']!=[.48145466,.4578275,.40821073] or meta['std']!=[.26862954,.26130258,.27577711] or meta['compute_dtype']!='float32' or meta['autocast'] is not False or features.dtype!=np.dtype('float16'):raise ValueError('Different A cache metadata/storage.')
                    for pos,i in sorted(positions):
                        old=legacy[names[i]];v=features[pos];p=old['tokens'].float().numpy();m=old['mask'].long().numpy()
                        if p.shape!=(length,512) or m.shape!=(length,) or not np.isfinite(v).all() or not np.isfinite(p).all() or not np.isin(m,[0,1]).all():raise ValueError('Invalid inputs.')
                        arrays['visual'][i]=v;arrays['legacy'][i]=p;arrays['mask'][i]=m;done+=1
                        if done%250==0:print(f'Staged {done}/2000 TRAIN images',flush=True)
            f.attrs['complete']=True
        with h5py.File(stage,'r') as f:
            for i in (0,499,1999):
                np.testing.assert_array_equal(f['visual'][i].astype('f4'),cache.read(selected[i]['coco_id']))
                np.testing.assert_array_equal(f['legacy'][i],legacy[names[i]]['tokens'].float().numpy())
    finally:cache.close()
    cfg.update(stage_sha256=digest(stage),preparation_seconds=time.monotonic()-start)
    save(out/'protocol.json',cfg)


def build(cfg):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst_runtime import model_digest
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False);tok=CaptionTokenizer([])
    tok.word2idx=state['caption_word2idx'];tok.idx2word={v:k for k,v in tok.word2idx.items()}
    model=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tok.word2idx)))
    if any(not k.startswith('base.') for k in state['model_state_dict']):raise ValueError('Unexpected A architecture.')
    model.load_state_dict({k[5:]:v for k,v in state['model_state_dict'].items()},strict=True)
    if model_digest(model)!=BASE_SHA:raise ValueError('Wrong A tensors.')
    return model.requires_grad_(False).cuda().eval(),tok


def generate(model,memory,padding,tok,coco_ids,draw):
    import torch
    seq=torch.full((len(coco_ids),1),tok.bos_idx,dtype=torch.long,device=memory.device);active=torch.ones(len(seq),dtype=torch.bool,device=memory.device)
    generators=[torch.Generator(device=memory.device).manual_seed(draw_seed(cid,draw)) for cid in coco_ids]
    for _ in range(29):
        logits=model.decoder(seq,memory,memory_key_padding_mask=padding)[:,-1].float()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits.')
        if draw==0:tokens=logits.argmax(-1)
        else:
            ordered,indices=(logits/.8).sort(dim=-1,descending=True);prob=ordered.softmax(-1);exclude=prob.cumsum(-1)-prob>=.95
            prob=prob.masked_fill(exclude,0);prob=prob/prob.sum(-1,keepdim=True)
            sampled=torch.stack([torch.multinomial(row,1,generator=g).squeeze(0) for row,g in zip(prob,generators)])
            tokens=indices.gather(1,sampled[:,None]).squeeze(1)
        tokens=torch.where(active,tokens,tok.pad_idx);seq=torch.cat([seq,tokens[:,None]],1);active &= tokens.ne(tok.eos_idx)
        if not active.any():break
    return seq


def worker(cfg):
    import h5py
    import torch
    from captioning.scst_runtime import configure_strict_runtime,runtime_manifest,model_digest
    from captioning.scst import rollout,decode
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True);torch.manual_seed(42)
    out=Path(cfg['output']);model,tok=build(cfg);start=time.monotonic();result=[];timings={};checked=False
    save(out/f'runtime_rank{rank}.json',runtime_manifest())
    with h5py.File(cfg['stage'],'r') as stage,torch.inference_mode():
        if not stage.attrs['complete']:raise ValueError('Incomplete stage.')
        batches=[(i,min(i+16,end)) for begin,end in ((0,500),(500,2000)) for i in range(begin,end,16)]
        for start_index,end_index in batches:
            indices=[i for i in range(start_index,end_index) if i%2==rank]
            if time.monotonic()-start>cfg['worker_budget_seconds']:raise RuntimeError('Inference budget exhausted; preserve partial results.')
            rows=[cfg['sample'][i] for i in indices];cids=[r['coco_id'] for r in rows]
            if list(map(int,stage['coco_ids'][indices]))!=cids:raise ValueError('Stage identity mismatch.')
            v=torch.from_numpy(stage['visual'][indices].astype('float32')).cuda();p=torch.from_numpy(stage['legacy'][indices]).cuda();m=torch.from_numpy(stage['mask'][indices]).cuda()
            memory,padding=model.build_memory(v,p,m);draws=[]
            for draw in range(5):draws.append(generate(model,memory,padding,tok,cids,draw))
            if not checked:
                expected,_=rollout(model,memory,padding,tok,sample=False,max_len=30)
                if not torch.equal(draws[0],expected):raise ValueError('Greedy policy differs from A.')
                repeat=generate(model,memory,padding,tok,cids,1)
                if not torch.equal(repeat,draws[1]):raise ValueError('Sampling RNG not repeatable.')
                checked=True
            for j,row in enumerate(rows):
                candidates=[]
                for draw,seq in enumerate(draws):
                    ids=seq[j].cpu().tolist();eos=tok.eos_idx in ids[1:];end=ids.index(tok.eos_idx,1) if eos else len(ids)
                    actions=ids[1:end];valid=eos and bool(actions) and not set(actions)&{tok.pad_idx,tok.bos_idx,tok.word2idx[tok.UNK]}
                    candidates.append(dict(draw=draw,caption=decode(seq[j:j+1],tok)[0],token_ids=ids[:end+1] if eos else ids,valid=bool(valid),ended=eos))
                result.append(dict(**row,candidates=candidates))
            save(out/f'candidates_rank{rank}.json',result)
            if end_index==500:
                torch.cuda.synchronize();timings['first500_seconds']=time.monotonic()-start
                timings['projected_2000_seconds']=timings['first500_seconds']*4
                save(out/f'profile_rank{rank}.json',timings)
                if timings['projected_2000_seconds']*1.5>cfg['worker_budget_seconds']:raise RuntimeError('500-image projection exceeds budget; no training started.')
            if start_index%128==0:print(f'GPU{rank}: {len(result)}/1000 TRAIN images, greedy + 4 samples',flush=True)
    torch.cuda.synchronize()
    if model_digest(model)!=BASE_SHA:raise ValueError('Frozen A changed.')
    save(out/f'complete_rank{rank}.json',dict(status='COMPLETE',images=len(result),seconds=time.monotonic()-start,base_sha256=BASE_SHA,
        candidates_sha256=digest(out/f'candidates_rank{rank}.json'),greedy_parity=True,sampling_repeatable=True,trained=False,peak_allocated=torch.cuda.max_memory_allocated()))


def evaluate(cfg):
    import numpy as np
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    out=Path(cfg['output']);records=[]
    for rank in (0,1):
        done=json.loads((out/f'complete_rank{rank}.json').read_text())
        if done['status']!='COMPLETE' or done['images']!=1000 or done['base_sha256']!=BASE_SHA or digest(out/f'candidates_rank{rank}.json')!=done['candidates_sha256']:raise ValueError('Incomplete/different generation.')
        records+=json.loads((out/f'candidates_rank{rank}.json').read_text())
    if len(records)!=2000 or {r['image_id'] for r in records}!={r['image_id'] for r in cfg['sample']}:raise ValueError('Missing/duplicate sample images.')
    records.sort(key=lambda r:r['image_id']);byid={r['image_id']:r for r in records};data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
    ids=sorted(cfg['train_ids']);ptb=PTBTokenizer();refs=ptb.tokenize({i:[dict(caption=s['raw']) for s in data['images'][i]['sentences'][:5]] for i in ids})
    # Unselected TRAIN rows have blank hypotheses only to hold the exact10k IDF fixed.
    for draw in range(5):
        hypotheses=ptb.tokenize({i:[dict(caption=byid[i]['candidates'][draw]['caption'] if i in byid else '')] for i in ids})
        _,values=Cider().compute_score(refs,hypotheses)
        for i,value in zip(ids,values):
            if i in byid:byid[i]['candidates'][draw]['CIDEr']=float(value)
    labels=json.loads((out/'labels_train_sample.json').read_text());policy=json.loads(Path('lexical_policy.json').read_text());pairs=[];gain=[];classes=Counter()
    for row in records:
        lab=labels[str(row['image_id'])];args=(row['candidates'],lab['support'],lab['salient'],policy,lab['conflict'])
        pair=choose_pair(*args);best=oracle(*args);gain.append(best['CIDEr']-row['candidates'][0]['CIDEr'])
        if pair:
            pairs.append(dict(image_id=row['image_id'],coco_id=row['coco_id'],filename=row['filename'],**pair));classes.update(lab['salient'])
    save(out/'scored_candidates.json',records);save(out/'eligible_pairs.json',pairs)
    save(out/'numeric_summary.json',dict(status='CANDIDATE_AUDIT_COMPLETE',trained=False,source_A_sha256=A_SHA,sample_images=2000,
        valid_pair_images=len(pairs),valid_pair_fraction=len(pairs)/2000,oracle_CIDEr_gain=float(np.mean(gain)),
        numeric_screen_pass=bool(len(pairs)>=400 and np.mean(gain)>=.05),
        salient_classes_in_pairs=dict(classes),images_detector_conflict=sum(v['conflict'] for v in labels.values()),
        greedy_CIDEr=float(np.mean([r['candidates'][0]['CIDEr'] for r in records])),
        warning='TRAIN-only oracle headroom, not validation/model gain. No train10k expansion or training authorized by this result alone.',
        human_review_required=30,human_gate='>=27/30 agree with chosen direction; no systematic parser/compound errors; not yet performed',
        reference_IDF='Fixed locked TRAIN10k; official PTB/CIDEr sigma6; blank unselected hypotheses affect no selected score',
        next_gate='After numeric+human review, full10k bank must yield>=2000 pair images and>=15 salient classes before training. Novelty unresolved.'))
    review=sorted(pairs,key=lambda p:hashlib.sha256(('candidate-review-v1:'+str(p['coco_id'])).encode()).hexdigest())[:30];key={};cards=[]
    for pair in review:
        cid=pair['coco_id'];swap=draw_seed(cid,9)%2;xy=[pair['winner'],pair['loser']][::(-1 if swap else 1)];key[str(cid)]='Y' if swap else 'X'
        split='val2014' if 'val2014' in pair['filename'] else 'train2014'
        cards.append(f'<article><h2>COCO {cid}</h2><img loading="lazy" src="https://images.cocodataset.org/{split}/{pair["filename"]}"><p>X: {html.escape(xy[0]["caption"])}</p><p>Y: {html.escape(xy[1]["caption"])}</p><label>Câu nào mô tả ảnh tốt hơn? <select data-id="{cid}"><option value="">Chưa chấm</option><option>X</option><option>Y</option><option value="tie">Ngang nhau</option><option value="unclear">Chưa rõ</option></select></label></article>')
    save(out/'review_key.json',key)
    fingerprint=digest(out/'eligible_pairs.json')
    page='<!doctype html><meta charset="utf-8"><title>Kiểm tra caption TRAIN</title><style>body{font:17px Arial;max-width:900px;margin:30px auto;background:#f5f6f8}article{background:white;padding:22px;margin:20px 0}img{max-width:700px;max-height:430px}select,button{font:inherit;padding:8px}</style><h1>Kiểm tra cặp caption TRAIN</h1><p>Chọn câu đúng ảnh và đầy đủ hơn; không chọn chỉ vì văn phong. Nếu ngang nhau hoặc chưa rõ, ghi đúng lựa chọn. Ảnh tải qua Internet.</p>'+''.join(cards)+'''<button onclick="download()">Tải đánh giá JSON</button><script>function download(){const annotations={};document.querySelectorAll('select').forEach(s=>annotations[s.dataset.id]=s.value);const b=new Blob([JSON.stringify({schema:1,manifest_sha256:'FINGERPRINT',annotations},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='candidate_pair_review.json';a.click()}</script>'''.replace('FINGERPRINT',fingerprint)
    (out/'review_30.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate']);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());verify(cfg);globals()[a.mode](cfg)
