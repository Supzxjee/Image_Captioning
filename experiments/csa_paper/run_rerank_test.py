"""USER RUN ONLY: locked CSA161 TEST decode and fixed CLIP-object selection."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha
from candidate_data import MODEL_SHA,content,image_batch
from rerank_test_data import prepare,load_features,read_beams,fixed_rank,checked_lock,POLICIES


def prefetch(cfg):
    from huggingface_hub import snapshot_download
    out=Path(cfg['output']);locked=checked_lock(out)['clip']
    root=Path(snapshot_download(locked['id'],revision=locked['revision'],allow_patterns=list(locked['files_sha256']),cache_dir=cfg['model_cache']))
    for n,h in locked['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('CLIP assets differ from VALID164: '+n)
    save(out/'clip_assets.json',dict(**locked,path=str(root)))
    print('CLIP model/processor match frozen VALID164 bytes; no BLIP download.',flush=True)


def decode(cfg):
    import torch
    from model import Config,CSACaptioner,beam_search
    from full_xe_decode import check_beam_rows
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);lock=checked_lock(out)
    if sha(lock['checkpoint'])!=lock['checkpoint_sha256'] or sha(Path(__file__).with_name('model.py'))!=MODEL_SHA:raise ValueError('CSA weights/core differ')
    ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)
    if ck['identity']['arm']!='SCST_CE' or ck['state']['updates']!=14161 or ck['state']['image_events']!=113287:raise ValueError('Not selected SCST+CE full1 END161')
    torch.manual_seed(42);base=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
    base.load_state_dict(ck['model'],strict=True);del ck;base.to(device).eval().requires_grad_(False)
    arrays,vocab=load_features(lock);inputs=read(out/'inference_inputs.json')
    if sha(out/'inference_inputs.json')!=lock['inference_sha256']:raise ValueError('Changed TEST inference IDs')
    if arrays['ids'].tolist()!=[r['coco_id'] for r in inputs]:raise ValueError('Feature/TEST ID order differs')
    indices=np.arange(5000)[rank::2];started=time.monotonic();count=0;path=out/f'beams_rank{rank}.jsonl'
    with path.open('x',encoding='utf-8') as log,torch.inference_mode():
        for offset in range(0,len(indices),8):
            if time.monotonic()-started>cfg['phase_seconds']:raise TimeoutError('TEST decode cap; preserve partial beams')
            group=indices[offset:offset+8];x,v=[torch.from_numpy(a).to(device) for a in image_batch(arrays,group)];vo,vr=base.encode(x,v)
            beams=beam_search(base,vo,vr,v,beams=5,maximum=20)
            for i,(seq,scores) in zip(group,beams):
                seq=seq.cpu().tolist();scores=scores.cpu().tolist();check_beam_rows([(seq,scores)],1,10000)
                candidates=[dict(ids=ids,caption=' '.join(vocab[t] for t in content(ids)),logprob=float(s)) for ids,s in zip(seq,scores)]
                log.write(json.dumps(dict(coco_id=int(arrays['ids'][i]),candidates=candidates),allow_nan=False)+'\n');count+=1
            log.flush()
            if offset%200==0:print('TEST DECODE GPU',rank,count,'/2500',flush=True)
            if offset==24 and (time.monotonic()-started)/count*(2500-count)>cfg['phase_seconds']-(time.monotonic()-started):raise TimeoutError('Measured TEST decode projection exceeds cap')
    save(out/f'decode_worker{rank}.json',dict(complete=True,images=count,beams_sha256=sha(path),checkpoint_sha256=lock['checkpoint_sha256'],
        optimizer_steps=0,beam=5,max_new_tokens=20,torch_version=torch.__version__,seconds=time.monotonic()-started))


def text_preflight(cfg):
    from transformers import CLIPProcessor
    out=Path(cfg['output']);lock=checked_lock(out);rows=read_beams(out,lock)
    model=read(out/'clip_assets.json');processor=CLIPProcessor.from_pretrained(model['path'],local_files_only=True,use_fast=False)
    lengths=[len(v) for v in processor.tokenizer([c['caption'] for r in rows for c in r['candidates']],truncation=False)['input_ids']]
    if len(lengths)!=25000 or max(lengths)>77:raise ValueError('Would truncate TEST candidates')
    save(out/'token_coverage.json',dict(images=5000,captions=25000,max_tokens=max(lengths),limit=77))


def worker(cfg):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);lock=checked_lock(out);asset=read(out/'clip_assets.json')
    for n,h in lock['clip']['files_sha256'].items():
        if sha(Path(asset['path'])/n)!=h:raise ValueError('Changed CLIP model/processor')
    base=CLIPModel.from_pretrained(asset['path'],local_files_only=True,use_safetensors=False,attn_implementation='eager').float().to(device).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(asset['path'],local_files_only=True,use_fast=False)
    if sha(out/'inference_inputs.json')!=lock['inference_sha256']:raise ValueError('Changed TEST inference inputs')
    inputs=read(out/'inference_inputs.json');beams=read_beams(out,lock);started=time.monotonic();count=0;path=out/f'scores_rank{rank}.jsonl'
    with path.open('x',encoding='utf-8') as log,torch.inference_mode():
        for row,beam in list(zip(inputs,beams))[rank::2]:
            if time.monotonic()-started>cfg['phase_seconds']:raise TimeoutError('TEST CLIP scoring cap; preserve partial scores')
            raw=Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Changed raw TEST image')
            with Image.open(io.BytesIO(raw)) as image:image=image.convert('RGB')
            pixels=processor(images=image,return_tensors='pt')['pixel_values'].to(device)
            text=processor.tokenizer([c['caption'] for c in beam['candidates']],padding=True,truncation=False,return_tensors='pt').to(device)
            image_features=F.normalize(base.get_image_features(pixel_values=pixels).float(),dim=-1)
            text_features=F.normalize(base.get_text_features(**text).float(),dim=-1)
            clipscore=(2.5*(image_features*text_features).sum(-1).clamp_min(0)).cpu().tolist()
            chosen=fixed_rank(beam,clipscore,row['detected'])
            log.write(json.dumps(dict(coco_id=row['coco_id'],image_sha256=row['image_sha256'],clipscore=clipscore,selected_rank=chosen),allow_nan=False)+'\n');log.flush();count+=1
            if count%100==0:print('TEST CLIP GPU',rank,count,'/2500',flush=True)
            if count==25 and (time.monotonic()-started)/count*(2500-count)>cfg['phase_seconds']-(time.monotonic()-started):raise TimeoutError('Measured TEST CLIP projection exceeds cap')
    save(out/f'worker{rank}.json',dict(complete=True,images=count,scored_pairs=count*5,scores_sha256=sha(path),
        seconds=time.monotonic()-started,torch_version=torch.__version__,training=False,clip_revision=lock['clip']['revision']))


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts
    out=Path(cfg['output']);lock=checked_lock(out);inputs=read(out/'inference_inputs.json');beams=read_beams(out,lock)
    if sha(out/'inference_inputs.json')!=lock['inference_sha256'] or sha(out/'test_references.json')!=lock['references_sha256']:raise ValueError('Changed locked inputs')
    scored={}
    for rank in (0,1):
        w=read(out/f'worker{rank}.json');path=out/f'scores_rank{rank}.jsonl'
        if not w['complete'] or w['images']!=2500 or w['scored_pairs']!=12500 or sha(path)!=w['scores_sha256']:raise ValueError('Incomplete TEST scoring worker')
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [r['coco_id'] for r in rows]!=[r['coco_id'] for r in inputs[rank::2]]:raise ValueError('Wrong TEST score IDs')
        scored.update({r['coco_id']:r for r in rows})
    if len(scored)!=5000:raise ValueError('Missing TEST scores')
    chosen=[]
    for row,beam in zip(inputs,beams):
        s=scored[row['coco_id']]
        if s['image_sha256']!=row['image_sha256']:raise ValueError('Wrong image-score identity')
        k=fixed_rank(beam,s['clipscore'],row['detected'])
        if k!=s['selected_rank']:raise ValueError('Fixed selection replay differs')
        chosen.append(k)
    save(out/'selected_ranks.json',chosen)
    refs=read(out/'test_references.json');ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs})
    chair=Path(cfg['chair']);metrics={};flags={}
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    for policy in POLICIES:
        selected=[dict(coco_id=r['coco_id'],caption=r['candidates'][0 if policy=='decoder_rank0' else k]['caption']) for r,k in zip(beams,chosen)]
        hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in selected})
        c,per=Cider().compute_score(ref,hyp);b,_=Bleu(4).compute_score(ref,hyp);r,_=Rouge().compute_score(ref,hyp)
        ci=out/(policy+'_chair_input.json');co=out/(policy+'_chair_output.json');save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in selected])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=300)
        ch=read(co);m,counts=chair_counts(ch,selected)
        metrics[policy]=dict(CIDEr=float(c),Bleu_4=float(b[3]),ROUGE_L=float(r),**m,counts=counts,
            mean_words=sum(len(r['caption'].split()) for r in selected)/5000,short_le2=sum(len(r['caption'].split())<=2 for r in selected))
        save(out/(policy+'_captions.json'),selected);save(out/(policy+'_cider_per_image.json'),[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(ref,per)])
        flags[policy]={r['image_id']:bool(r['mscoco_hallucinated_words']) for r in ch['sentences']}
    a,b=[flags[p] for p in POLICIES]
    changes=dict(changed_captions=sum(k!=0 for k in chosen),rank_histogram={str(k):chosen.count(k) for k in range(5)},
        CHAIR_repairs=sum(a[i] and not b[i] for i in a),CHAIR_new_flags=sum(not a[i] and b[i] for i in a))
    result=dict(status='CSA161_FIXED_RERANK_TEST5000_COMPLETE',images=5000,test_used=True,training=False,
        checkpoint_sha256=lock['checkpoint_sha256'],metrics=metrics,changes=changes,selector=lock['selector'],
        tuned_on_test=False,oracle_used_for_selection=False,prior_project_test_exposure=True,
        selection='both prelocked policies reported; no TEST-based fallback',NLL_note='same frozen captioner; reranking changes output only, no target-conditioned pass')
    save(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','prefetch','decode','text_preflight','worker','score']);p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
