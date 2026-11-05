"""USER RUN ONLY: frozen CLIP and BLIP scoring; no captioner/training."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from rerank_data import MODELS,POLICIES,prepare,select
from real_bank_data import read,save,sha


def prefetch(cfg):
    from huggingface_hub import snapshot_download
    from transformers import CLIPProcessor,BlipProcessor
    out=Path(cfg['output']);manifest={}
    for name,m in MODELS.items():
        folder=Path(snapshot_download(m['id'],revision=m['revision'],allow_patterns=['*.json','*.txt','pytorch_model.bin'],cache_dir=cfg['model_cache']))
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
        if not {'config.json','preprocessor_config.json','pytorch_model.bin'}<=set(files):raise ValueError('Missing scoring model assets')
        manifest[name]=dict(**m,path=str(folder),files_sha256=files)
    rows=read(out/'inference_inputs.json');texts=[c['caption'] for r in rows for c in r['candidates']]
    maximum={}
    for name,cls in [('clip',CLIPProcessor),('blip',BlipProcessor)]:
        processor=cls.from_pretrained(manifest[name]['path'],local_files_only=True,use_fast=False)
        lengths=[len(t) for t in processor.tokenizer(texts,truncation=False)['input_ids']]
        config=read(Path(manifest[name]['path'])/'config.json')
        limit=config['text_config']['max_position_embeddings']
        if len(lengths)!=25000 or max(lengths)>limit:raise ValueError('Would truncate '+name+' candidates')
        maximum[name]=dict(max_tokens=max(lengths),limit=limit)
    save(out/'models.json',manifest);save(out/'token_coverage.json',maximum)
    print('Pinned scoring assets cached and hashed; all25000captions untruncated.',flush=True)


def worker(cfg):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor,BlipForImageTextRetrieval,BlipProcessor
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);receipt=read(out/'prepared.json');models=read(out/'models.json')
    if sha(out/'inference_inputs.json')!=receipt['inference_inputs_sha256']:raise ValueError('Changed inference payload')
    for name,m in models.items():
        if (m['id'],m['revision'])!=(MODELS[name]['id'],MODELS[name]['revision']):raise ValueError('Wrong model revision')
        for n,h in m['files_sha256'].items():
            if sha(Path(m['path'])/n)!=h:raise ValueError('Changed scoring model/processor asset')
    clip=CLIPModel.from_pretrained(models['clip']['path'],local_files_only=True,use_safetensors=False,attn_implementation='eager').float().to(device).eval()
    blip=BlipForImageTextRetrieval.from_pretrained(models['blip']['path'],local_files_only=True,use_safetensors=False,attn_implementation='eager').float().to(device).eval()
    cp=CLIPProcessor.from_pretrained(models['clip']['path'],local_files_only=True,use_fast=False)
    bp=BlipProcessor.from_pretrained(models['blip']['path'],local_files_only=True,use_fast=False)
    for model in (clip,blip):
        for p in model.parameters():p.requires_grad_(False)
    rows=read(out/'inference_inputs.json')[rank::2];started=time.monotonic();count=0
    path=out/f'scores_rank{rank}.jsonl'
    with path.open('x',encoding='utf-8') as log,torch.inference_mode():
        for row in rows:
            if time.monotonic()-started>cfg['worker_seconds']:raise TimeoutError('Scoring worker cap; preserve partial scores')
            raw=Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Raw image changed')
            with Image.open(io.BytesIO(raw)) as image:image=image.convert('RGB')
            texts=[c['caption'] for c in row['candidates']]
            pixels=cp(images=image,return_tensors='pt')['pixel_values'].to(device)
            text=cp.tokenizer(texts,padding=True,truncation=False,return_tensors='pt').to(device)
            image_features=F.normalize(clip.get_image_features(pixel_values=pixels).float(),dim=-1)
            text_features=F.normalize(clip.get_text_features(**text).float(),dim=-1)
            cosine=(image_features*text_features).sum(-1)
            clipscore=(2.5*cosine.clamp_min(0)).cpu().tolist()
            pairs=bp(images=[image]*5,text=texts,padding=True,truncation=False,return_tensors='pt').to(device)
            logits=blip(**pairs,use_itm_head=True).itm_score.float()
            if logits.shape!=(5,2) or not torch.isfinite(logits).all():raise ValueError('Invalid BLIP ITM logits')
            scored=dict(row,clipscore=clipscore,itm_logits=logits.cpu().tolist(),itm_probability=logits.softmax(-1)[:,1].cpu().tolist())
            selected=select(scored)
            log.write(json.dumps(dict(coco_id=row['coco_id'],image_sha256=row['image_sha256'],clipscore=clipscore,
                itm_logits=scored['itm_logits'],itm_probability=scored['itm_probability'],selected_ranks=selected),allow_nan=False)+'\n')
            log.flush();count+=1
            if count%50==0:print('RERANK GPU',rank,count,'/2500',flush=True)
            if count==25:
                remaining=(time.monotonic()-started)/count*(2500-count)
                if remaining>cfg['worker_seconds']-(time.monotonic()-started):raise TimeoutError('Measured scoring projection exceeds cap')
    save(out/f'worker{rank}.json',dict(complete=True,images=count,scored_pairs=count*5,scores_sha256=sha(path),
        seconds=time.monotonic()-started,torch_version=torch.__version__,models=MODELS,fp32=True,training=False))


def score(cfg,preflight=False):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts
    from candidate_data import baseline
    out=Path(cfg['output']);prepared=read(out/'prepared.json')
    if sha(out/'inference_inputs.json')!=prepared['inference_inputs_sha256']:raise ValueError('Changed selection inputs')
    rows=read(out/'inference_inputs.json');policies={p:[] for p in POLICIES};ranks={p:[] for p in POLICIES}
    if preflight:
        policies={'decoder_rank0':[dict(coco_id=r['coco_id'],caption=r['candidates'][0]['caption']) for r in rows]}
    else:
        scored={}
        for rank in (0,1):
            w=read(out/f'worker{rank}.json');path=out/f'scores_rank{rank}.jsonl'
            if not w['complete'] or w['images']!=2500 or w['scored_pairs']!=12500 or w['models']!=MODELS or sha(path)!=w['scores_sha256']:raise ValueError('Incomplete scoring worker')
            part=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if [r['coco_id'] for r in part]!=[r['coco_id'] for r in rows[rank::2]]:raise ValueError('Scored ID/order mismatch')
            scored.update({r['coco_id']:r for r in part})
        if len(scored)!=5000:raise ValueError('Wrong full scoring cohort')
        for row in rows:
            s=scored[row['coco_id']]
            if s['image_sha256']!=row['image_sha256']:raise ValueError('Wrong image identity')
            chosen=select(dict(row,clipscore=s['clipscore'],itm_logits=s['itm_logits']))
            if chosen!=s['selected_ranks']:raise ValueError('Selector replay differs')
            for p,k in chosen.items():
                ranks[p].append(k);policies[p].append(dict(coco_id=row['coco_id'],caption=row['candidates'][k]['caption']))
        save(out/'selected_ranks.json',ranks)
    refs=read(out/'validation_references.json');saved=baseline()
    if sha(out/'validation_references.json')!=prepared['references_sha256'] or refs!=saved['references']:raise ValueError('Reference identity differs')
    ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs})
    chair=Path(cfg['chair']);metrics={};flags={}
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    for p,selected in policies.items():
        hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in selected})
        c,per=Cider().compute_score(ref,hyp);b,_=Bleu(4).compute_score(ref,hyp);r,_=Rouge().compute_score(ref,hyp)
        ci=out/(p+'_chair_input.json');co=out/(p+'_chair_output.json')
        save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in selected])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=300)
        chair_output=read(co);m,counts=chair_counts(chair_output,selected)
        point=dict(CIDEr=float(c),Bleu_4=float(b[3]),ROUGE_L=float(r),**m,counts=counts,
            mean_words=sum(len(x['caption'].split()) for x in selected)/5000,short_le2=sum(len(x['caption'].split())<=2 for x in selected))
        if p=='decoder_rank0':
            old=saved['metrics']
            for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','mean_words','short_le2'):
                if abs(point[k]-old[k])>1e-7:raise ValueError('Baseline metric replay differs: '+k)
            if counts!=old['counts']:raise ValueError('Baseline CHAIR counts differ')
        save(out/(p+'_captions.json'),selected)
        save(out/(p+'_cider_per_image.json'),[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(ref,per)])
        flags[p]={r['image_id']:bool(r['mscoco_hallucinated_words']) for r in chair_output['sentences']};metrics[p]=point
    if preflight:
        save(out/'metric_preflight.json',dict(passed=True,images=5000,metrics=metrics['decoder_rank0']))
        print('Exact rank0 VALID5000 metric replay PASS before scoring models.',flush=True);return
    changes={}
    for p in POLICIES[1:]:
        base=flags['decoder_rank0'];other=flags[p]
        changes[p]=dict(changed_captions=sum(k!=0 for k in ranks[p]),rank_histogram={str(k):ranks[p].count(k) for k in range(5)},
            CHAIR_repairs=sum(base[i] and not other[i] for i in base),CHAIR_new_flags=sum(not base[i] and other[i] for i in base))
    result=dict(status='CSA_FIXED_RERANK_VALID_COMPLETE',images=5000,metrics=metrics,changes=changes,actual_reranking=True,
        test_used=False,training=False,oracle_used_for_selection=False,candidate_source='ducanh2403/2-vi-cai-kia-qua-nhieu/163',models=MODELS,
        selection='fixed decoder rank0 / .3decoder+.4CLIP+.3object / pure BLIP positive-negative margin; earlier ties',
        limitations=['VALID only; external BLIP COCO fine-tuning split not independently audited.',
            'Detector absence is not ground truth; new official CLIP processor differs historical cached-CLS pipeline.'])
    save(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def metric_preflight(cfg):score(cfg,preflight=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','prefetch','worker','metric_preflight','score']);p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
