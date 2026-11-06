"""USER-RUN frozen TEST5000, six prelocked conditions across two accounts."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha,validate_arrays
from prompt_test_data import cache_root,source_checkpoint,load_inputs,donor_plan,prompt_batch,image_batch
from prompt_scst_rerank import validation_image_paths,validation_detections,prepare_images,fixed_rank


def modes(arm):return ('normal','zero','shuffle') if arm=='PROMPT' else ('normal',)


def metric_preflight(cfg):
    import shutil
    from run_scst_pilot import metric_preflight as check
    root=Path(cfg['output']);scratch=root/'metric_smoke';scratch.mkdir(parents=True,exist_ok=False)
    shutil.copy2(root/'test_references.json',scratch/'validation_references.json')
    check({**cfg,'output':str(scratch)})


def prepare(cfg):
    from scst_pilot_data import DATA_SHA,DETECTION_SHA
    out=Path(cfg['output']);root,inv=cache_root(cfg['cache_root']);source_checkpoint(cfg['checkpoint_root'],cfg['arm'])
    for key,digest in (('dataset_json',DATA_SHA),('detections',DETECTION_SHA)):
        print('Hashing TEST input:',key,flush=True)
        if sha(cfg[key])!=digest:raise ValueError('Changed '+key)
    a,rows,texts,vectors,_=load_inputs(root);meta=read(root/'test_metadata.json');validate_arrays(a,meta,read(root/'vocabulary.json'))
    dataset={int(r['cocoid']):r for r in read(cfg['dataset_json'])['images'] if r['split']=='test'}
    if set(dataset)!=set(a['ids'].tolist()) or any(r['source_split']!='test' or r['group']!='test' for r in meta):raise ValueError('Wrong Karpathy TEST cohort')
    if any(dataset[r['coco_id']]['filename']!=r['filename'] or [s['raw'] for s in dataset[r['coco_id']]['sentences']]!=r['captions'] for r in meta):raise ValueError('Wrong TEST reference provenance')
    paths=validation_image_paths(cfg['coco_root'],list(dataset.values()));detected=validation_detections(read(cfg['detections']),{r['filename'] for r in dataset.values()})
    inputs=prepare_images(dataset,paths,detected)
    save(out/'inference_inputs.json',inputs);save(out/'test_references.json',[dict(coco_id=r['coco_id'],references=r['captions']) for r in meta])
    _,donors=donor_plan(rows);save(out/'donors.json',donors)
    save(out/'test_lock.json',dict(cache_root=str(root),cache_files=inv['cache_files'],arm=cfg['arm'],checkpoint_sha256=sha(source_checkpoint(cfg['checkpoint_root'],cfg['arm'])),
        source=read(Path(__file__).with_name('PROMPT_SCST_RESULT.json'))['arms'][cfg['arm']]['save_version'],
        inference_sha256=sha(out/'inference_inputs.json'),references_sha256=sha(out/'test_references.json'),donors_sha256=sha(out/'donors.json'),
        modes=modes(cfg['arm']),beam=5,maximum=20,training=False,prior_project_test_exposure=True))
    print('TEST inputs and donor plan LOCKED; no captions scored.',flush=True)


def decode(cfg):
    import torch
    from model import Config,CSACaptioner,beam_search
    from prompt_adapter_model import PromptCSA
    from full_xe_decode import check_beam_rows
    from cuda_runtime_guard import check_runtime
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=read(out/'test_lock.json');root,_=cache_root(cfg['cache_root']);a,rows,texts,vectors,vocab=load_inputs(root)
    plan,donors=donor_plan(rows)
    if donors!=read(out/'donors.json') or sha(out/'donors.json')!=lock['donors_sha256']:raise ValueError('Changed donor plan')
    path=source_checkpoint(cfg['checkpoint_root'],cfg['arm'])
    if sha(path)!=lock['checkpoint_sha256']:raise ValueError('Changed selected weights')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['identity']['schema']!='prompt_scratch_scst_full1_v1' or ck['identity']['arm']!=cfg['arm'] or ck['state']['updates']!=14161 or ck['state']['image_events']!=113287 or ck['state']['validated']!=['original','end']:raise ValueError('Wrong SCST endpoint')
    for n in ('model.py','prompt_adapter_model.py'):
        if sha(Path(__file__).with_name(n))!=ck['identity']['sources']['sha256']['experiments/csa_paper/'+n]:raise ValueError('Changed model implementation:'+n)
    torch.manual_seed(42);model=PromptCSA(CSACaptioner(Config(vocab=10000),ck['model']['base.nouns'],ck['model']['base.verbs']))
    model.load_state_dict(ck['model'],strict=True);del ck;model.to(device).eval().requires_grad_(False)
    indices=np.arange(5000)[rank::2];started=time.monotonic()
    for mode in modes(cfg['arm']):
        output=out/f'{mode}_beams_rank{rank}.jsonl';count=0;began=time.monotonic()
        with output.open('x',encoding='utf-8') as f,torch.inference_mode():
            for offset in range(0,len(indices),8):
                if time.monotonic()-started>cfg['decode_seconds']:raise TimeoutError('Decode cap; preserve beams, no restart')
                ix=indices[offset:offset+8];x,v=[torch.from_numpy(b).to(device) for b in image_batch(a,ix)]
                p={k:torch.from_numpy(b).to(device) for k,b in prompt_batch([rows[i] for i in ix],vectors,texts,mode,plan).items()}
                memory=model.encode(x,v,p,enabled=cfg['arm']=='PROMPT')
                for i,(seq,scores) in zip(ix,beam_search(model.base,*memory,v,beams=5,maximum=20)):
                    ids=seq.cpu().tolist();scores=scores.cpu().tolist();check_beam_rows([(ids,scores)],1,10000);beams=[]
                    for token,score in zip(ids,scores):
                        content=token[1:];content=content[:content.index(2)] if 2 in content else content
                        beams.append(dict(ids=token,caption=' '.join(vocab[t] for t in content if t>1),logprob=float(score)))
                    f.write(json.dumps(dict(coco_id=int(a['ids'][i]),candidates=beams),allow_nan=False)+'\n');count+=1
                f.flush()
                if offset%200==0:print('TEST decode',cfg['arm'],mode,rank,count,'/2500',flush=True)
        save(out/f'{mode}_decode_rank{rank}.json',dict(complete=True,images=count,sha256=sha(output),seconds=time.monotonic()-began,mode=mode,checkpoint_sha256=lock['checkpoint_sha256'],optimizer_steps=0))


def read_beams(out,mode):
    rows=[]
    for rank in (0,1):
        p=out/f'{mode}_beams_rank{rank}.jsonl';r=read(out/f'{mode}_decode_rank{rank}.json')
        if not r['complete'] or r['images']!=2500 or sha(p)!=r['sha256']:raise ValueError('Incomplete TEST decode')
        rows.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines())
    from prompt_scst_data import validate_candidates
    validate_candidates(rows,{r['coco_id'] for r in read(out/'inference_inputs.json')})
    return {r['coco_id']:r for r in rows}


def rerank(cfg):
    import hashlib,io
    import torch
    from torch.nn import functional as F
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(cfg['output']);asset=read(out/'clip_assets.json');spec=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip']
    if {k:v for k,v in asset.items() if k!='path'}!=spec:raise ValueError('Wrong CLIP receipt')
    for n,h in spec['files_sha256'].items():
        if sha(Path(asset['path'])/n)!=h:raise ValueError('Changed CLIP file:'+n)
    lock=read(out/'test_lock.json')
    if sha(out/'inference_inputs.json')!=lock['inference_sha256']:raise ValueError('Changed inference payload')
    model=CLIPModel.from_pretrained(asset['path'],local_files_only=True,use_safetensors=False,attn_implementation='eager').float().to(device).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(asset['path'],local_files_only=True,use_fast=False);beams=read_beams(out,'normal');inputs=read(out/'inference_inputs.json')[rank::2];started=time.monotonic();count=0
    path=out/f'rerank_rank{rank}.jsonl'
    with path.open('x',encoding='utf-8') as f,torch.inference_mode():
        for row in inputs:
            if time.monotonic()-started>cfg['rerank_seconds']:raise TimeoutError('Rerank cap; preserve captions')
            raw=Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('JPEG changed')
            with Image.open(io.BytesIO(raw)) as image:image=image.convert('RGB')
            pixels=processor(images=image,return_tensors='pt')['pixel_values'].to(device);text=processor.tokenizer([c['caption'] for c in beams[row['coco_id']]['candidates']],padding=True,truncation=False,return_tensors='pt').to(device)
            if text['input_ids'].shape[1]>77:raise ValueError('Overlong CLIP text')
            scores=(2.5*(F.normalize(model.get_image_features(pixel_values=pixels).float(),dim=-1)*F.normalize(model.get_text_features(**text).float(),dim=-1)).sum(-1).clamp_min(0)).cpu().tolist()
            f.write(json.dumps(dict(coco_id=row['coco_id'],clipscore=scores,selected_rank=fixed_rank(beams[row['coco_id']],scores,row['detected'])),allow_nan=False)+'\n');f.flush();count+=1
            if count%100==0:print('TEST rerank',rank,count,'/2500',flush=True)
    save(out/f'rerank_worker{rank}.json',dict(complete=True,images=count,sha256=sha(path),seconds=time.monotonic()-started))


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts,check_rows
    out=Path(cfg['output']);lock=read(out/'test_lock.json');refs=read(out/'test_references.json');inputs=read(out/'inference_inputs.json');expected={r['coco_id'] for r in refs}
    if len(expected)!=5000 or sha(out/'test_references.json')!=lock['references_sha256'] or sha(out/'inference_inputs.json')!=lock['inference_sha256']:raise ValueError('Changed TEST identities')
    normal=read_beams(out,'normal');selected={}
    for rank in (0,1):
        path=out/f'rerank_rank{rank}.jsonl';done=read(out/f'rerank_worker{rank}.json');rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if not done['complete'] or done['images']!=2500 or sha(path)!=done['sha256'] or [r['coco_id'] for r in rows]!=[r['coco_id'] for r in inputs[rank::2]]:raise ValueError('Incomplete rerank scores')
        selected.update({r['coco_id']:r for r in rows})
    for row in inputs:
        s=selected[row['coco_id']]
        if fixed_rank(normal[row['coco_id']],s['clipscore'],row['detected'])!=s['selected_rank']:raise ValueError('Selector replay differs')
    policies={m:[dict(coco_id=i,caption=r['candidates'][0]['caption']) for i,r in read_beams(out,m).items()] for m in modes(cfg['arm'])}
    policies['fixed_rerank']=[dict(coco_id=i,caption=r['candidates'][selected[i]['selected_rank']]['caption']) for i,r in normal.items()]
    ptb=PTBTokenizer();reference=ptb.tokenize({r['coco_id']:[dict(caption=t) for t in r['references']] for r in refs});metrics={};chair=Path(cfg['chair'])
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    for policy,rows in policies.items():
        check_rows(rows,expected);hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows});c,per=Cider().compute_score(reference,hyp);b,_=Bleu(4).compute_score(reference,hyp);rouge,_=Rouge().compute_score(reference,hyp)
        ci=out/f'{policy}_chair_input.json';co=out/f'{policy}_chair_output.json';save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=300)
        ch,counts=chair_counts(read(co),rows);metrics[policy]=dict(CIDEr=float(c),Bleu_4=float(b[3]),ROUGE_L=float(rouge),**ch,counts=counts,mean_words=sum(len(r['caption'].split()) for r in rows)/5000)
        save(out/f'{policy}_captions.json',rows);save(out/f'{policy}_per_image.json',[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(reference,per)])
    save(out/'result.json',dict(status='PROMPT_TEST5000_COMPLETE',arm=cfg['arm'],images=5000,checkpoint_sha256=lock['checkpoint_sha256'],metrics=metrics,training=False,test_tuning=False,prior_project_test_exposure=True,
        limitations=['One seed, C10/P9 unequal initial XE epochs.','Zero/shuffle are inference sensitivity controls, not matched training ablation.','CHAIR measures objects, not relation truth.']))
    print('TEST COMPLETE',cfg['arm'],metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','metric_preflight','decode','rerank','score'));p.add_argument('--config',required=True);a=p.parse_args();globals()[a.mode](read(a.config))
