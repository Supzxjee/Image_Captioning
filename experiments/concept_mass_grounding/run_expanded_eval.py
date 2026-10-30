"""User-run evaluation only: copy pinned caches, infer three fixed models, score."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from expanded_data import *

def prepare(cfg):
 import h5py
 import torch
 from captioning.visual_cache import VisualCache
 from captioning.semantic_prompts import align_prompt_cache
 from recover_stage import normalized,row_sha
 start=time.monotonic();out=Path(cfg['output'])
 if out.exists():raise FileExistsError('Preserve old output; choose a fresh path')
 for key,want in [('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA),('metadata',META_SHA)]:
  print('Hashing pinned input:',key,flush=True)
  if sha(cfg[key])!=want:raise ValueError('Changed input '+key)
 for name,info in CHECKPOINTS.items():
  if sha(cfg['checkpoints'][name])!=info['sha256']:raise ValueError('Wrong checkpoint '+name)
 if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA or sha(HERE/'EXPANDED_EVAL_MANIFEST.json')!=MANIFEST_SHA:raise ValueError('Changed metric cache/manifest')
 old=pinned_metadata(cfg['metadata']);manifest=json.loads((HERE/'EXPANDED_EVAL_MANIFEST.json').read_bytes())
 primary,sentinels=rows_and_sentinels(manifest,old)
 corpus=json.loads(Path(cfg['dataset_json']).read_bytes())['images'];rows=checked_rows(primary+sentinels,corpus);del corpus
 meta=dict(rows=rows,word2idx=old['word2idx'],primary_indices=list(range(4000)),sentinel_indices=list(range(4000,4016)))
 out.mkdir(parents=True);save(out/'protocol.json',cfg);save(out/'metadata.json',meta)
 if shutil.disk_usage(out).free<3*1024**3:raise RuntimeError('Need3GiB free for prepared cache and outputs')
 bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
 prompts=align_prompt_cache(bundle['data'],[r['filename'] for r in rows]);del bundle
 cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in rows],'all4000+16parity sentinels')
 expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],antialias=True,compute_dtype='float32',autocast=False)
 grouped=defaultdict(list)
 for i,r in enumerate(rows):
  fi,position=cache.index[r['coco_id']];grouped[fi].append((position,i))
 try:
  with h5py.File(out/'inputs.h5','x') as target:
   target.attrs['complete']=False
   for key,shape,dtype in [('visual',(197,768),'f2'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:target.create_dataset(key,(len(rows),*shape),dtype=dtype)
   target.create_dataset('coco_ids',data=[r['coco_id'] for r in rows]);journal=target.create_dataset('row_sha256',(len(rows),),dtype='S64');done=0
   for fi,positions in sorted(grouped.items()):
    with h5py.File(cache.paths[fi],'r') as source:
     info=json.loads(source.attrs['metadata_json'])
     if str(source['features'].dtype)!='float16' or any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Visual source metadata changed')
     for position,i in sorted(positions):
      p=prompts[rows[i]['filename']];values=normalized(dict(visual=source['features'][position],prompt=p['tokens'].float().numpy(),mask=p['mask'].numpy()))
      for k,v in values.items():target[k][i]=v
      journal[i]=row_sha(values).encode();done+=1
      if done==1 or done%250==0:target.flush();print(f'Staged exact caches {done}/{len(rows)}; {time.monotonic()-start:.1f}s',flush=True)
      if time.monotonic()-start>EVAL_POLICY['prepare_seconds']:raise TimeoutError('Preparation cap')
   target.attrs['complete']=True
 finally:cache.close()
 save(out/'preparation.json',dict(status='PREPARATION_COMPLETE',images=4000,sentinels=16,manifest_sha256=MANIFEST_SHA,checkpoint_sha256={n:i['sha256'] for n,i in CHECKPOINTS.items()},payload_sha256={n:sha(out/n) for n in ['metadata.json','inputs.h5']},seconds=time.monotonic()-start,neural_forward=False))

def worker(cfg):
 import torch
 from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
 from concept_data import Inputs
 from run_concept import build,tokenizer,predict
 rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);data=None;start=time.monotonic()
 if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Require exactly two GPUs')
 configure_strict_runtime(True);torch.cuda.set_device(rank);torch.set_num_threads(2);device=torch.device('cuda',rank)
 try:
  prep=json.loads((out/'preparation.json').read_bytes())
  if prep['status']!='PREPARATION_COMPLETE':raise ValueError('Incomplete preparation')
  for n,h in prep['payload_sha256'].items():
   if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Changed prepared payload')
  data=Inputs(out);tok=tokenizer(data.meta);save(out/f'runtime_rank{rank}.json',runtime_manifest())
  sentinel=json.loads((HERE/'EXPANDED_EVAL_SENTINELS.json').read_bytes());reports={}
  for name in EVAL_POLICY['models']:
   if sha(cfg['checkpoints'][name])!=CHECKPOINTS[name]['sha256']:raise ValueError('Changed checkpoint '+name)
   torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   model=build(dict(cfg,checkpoint=cfg['checkpoints']['A']),device)
   if name!='A':
    state=torch.load(cfg['checkpoints'][name],map_location='cpu',weights_only=False)
    if state['arm']!=name or state['updates']!=500 or state['caption_word2idx']!=data.meta['word2idx']:raise ValueError('Wrong adapted checkpoint')
    weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Wrong checkpoint keys')
    model.base.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True);del state,weights
   model.requires_grad_(False).eval();before=model_digest(model)
   if before!=CHECKPOINTS[name]['digest']:raise ValueError('Native wrapper tensor identity differs '+name)
   sent,ts=predict(model,data,data.meta['sentinel_indices'][rank::2],tok,device)
   expected={r['image_id']:r['caption'] for r in sentinel['predictions'][name]}
   if any(r['caption']!=expected[r['image_id']] for r in sent):raise ValueError('Historical caption parity failed '+name)
   records,timing=predict(model,data,data.meta['primary_indices'][rank::2],tok,device)
   validate_predictions(records,[data.meta['rows'][i] for i in data.meta['primary_indices'][rank::2]])
   if model_digest(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Inference changed model/gradients')
   save(out/f'{name}_rank{rank}.json',records);reports[name]=dict(model_digest=before,sentinel_mismatches=0,timing=timing)
   del model;torch.cuda.empty_cache()
   if time.monotonic()-start>EVAL_POLICY['gpu_seconds']:raise TimeoutError('Inference phase cap')
  save(out/f'worker_rank{rank}.json',dict(status='EVALUATION_INFERENCE_COMPLETE',rank=rank,images=2000,models=reports,training=False,collectives=False,seconds=time.monotonic()-start))
 except BaseException as e:save(out/f'failure_rank{rank}.json',dict(status='INCOMPLETE',error=repr(e)));raise
 finally:
  if data is not None:data.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate']);p.add_argument('--config',required=True);args=p.parse_args();cfg=json.loads(Path(args.config).read_bytes())
 if cfg['policy']!=EVAL_POLICY:raise ValueError('Changed locked evaluation policy')
 for name,h in cfg['source_assets'].items():
  if sha(HERE.parents[1]/name)!=h:raise ValueError('Changed source '+name)
 if args.mode=='prepare':prepare(cfg)
 else:
  if json.loads((Path(cfg['output'])/'protocol.json').read_bytes())!=cfg:raise ValueError('Changed config after prepare')
  if args.mode=='worker':worker(cfg)
  else:
   from expanded_metrics import evaluate
   evaluate(cfg)

if __name__=='__main__':main()
