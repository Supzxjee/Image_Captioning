"""Validation/test orchestration: shared beams, fixed baseline, frozen UPT evidence."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def command(args, cwd=None):
    print('Running:', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def merge_records(records, images):
    lookup={r['image_id']:r for r in records}
    if len(lookup)!=len(records) or set(lookup)!={r['image_id'] for r in images}:
        raise ValueError('Missing or duplicate shard IDs.')
    ordered=[lookup[r['image_id']] for r in images]
    for original,record in zip(images,ordered):
        if any(original[k]!=record[k] for k in ('image_id','coco_id','filename')):
            raise ValueError('Shard identity mismatch.')
    return ordered


def load_reused_candidates(source, source_protocol, cfg):
    """Reuse completed beams/CLIP only after validating their source identity."""
    old=json.loads(Path(source_protocol).read_text(encoding='utf-8'))
    bundle=json.loads(Path(source).read_text(encoding='utf-8'))
    meta=bundle.get('metadata',{})
    if meta.get('protocol_sha256')!=sha(source_protocol): raise ValueError('Reused candidate protocol hash differs.')
    for key in ('checkpoint','prompt_cache_path','dataset_json_path'):
        if old['input_sha256'][key]!=cfg['input_sha256'][key]: raise ValueError(f'Reused {key} differs.')
    if old['images']!=cfg['images'] or old['visual_shards']!=cfg['visual_shards']:
        raise ValueError('Reused image IDs/order or visual shards differ.')
    expected={'split':cfg.get('split','val'),'count':len(cfg['images']),'beam_size':5,'candidate_count':5,
        'epoch':10,'visual_adapter':'qformer','num_visual_queries':32,'qformer_layers':2,
        'prompt_conditioned_qformer':False,'clipscore_model':'openai/clip-vit-base-patch16',
        'clipscore_revision':cfg['clip_revision'],'checkpoint_sha256':cfg['input_sha256']['checkpoint']}
    if any(meta.get(k)!=v for k,v in expected.items()): raise ValueError('Reused candidate model/split/CLIP revision differs.')
    records=merge_records(bundle.get('data',[]),cfg['images'])
    for record in records:
        candidates=record.get('candidates',[])
        if not 1<=len(candidates)<=5 or len({c['caption'] for c in candidates})!=len(candidates):
            raise ValueError('Invalid reused beam candidates.')
        if any(not math.isfinite(c[k]) for c in candidates for k in ('logprob','clipscore')):
            raise ValueError('Nonfinite reused candidate score.')
    return {**bundle,'data':records}


def load_test_selection(path):
    if sha(path)!='4d812d8b4bcc02f4d24823b5eaf50d72a8fe41d5d6d735c036934b09bdf6ac38': raise ValueError('Validation selection file hash differs.')
    locked=json.loads(Path(path).read_text(encoding='utf-8'))
    if (locked['split']!='val' or locked['test_run'] or not locked['selection_on_validation_only']
        or locked['fixed_base']!={'decoder':.3,'clip':.4,'occ':.3} or locked['upt_weight']!=.1):
        raise ValueError('Expected validation-frozen lambda=.1 and original fixed baseline.')
    return locked


def run_workers(cfg, phase, repo, worker_script='hoi_caption_worker.py'):
    config_path=Path(cfg['output'])/f'{phase}_worker_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg: raise ValueError('Worker protocol changed.')
    save(config_path,cfg)
    processes=[]; handles=[]; positions=[0]*cfg['workers']
    try:
        for rank in range(cfg['workers']):
            log=Path(cfg['output'])/f'{phase}_gpu{rank}.log'
            stream=log.open('a',encoding='utf-8');handles.append(stream)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE')
            if cfg.get('strict_reproducibility', False):
                env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42')
            processes.append(subprocess.Popen([sys.executable,'-u',str(repo/worker_script),
                '--config',str(config_path),'--rank',str(rank),'--phase',phase],cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT))
        last=time.monotonic()
        while True:
            for rank in range(cfg['workers']):
                path=Path(cfg['output'])/f'{phase}_gpu{rank}.log'
                with path.open(encoding='utf-8',errors='replace') as reader:
                    reader.seek(positions[rank]); text=reader.read();positions[rank]=reader.tell()
                if text: print(f'[GPU {rank}] {text}',end='',flush=True)
                status=processes[rank].poll()
                if status not in (None,0): raise RuntimeError(f'{phase} GPU {rank} failed ({status}); see {path}')
            if all(p.poll()==0 for p in processes): break
            if time.monotonic()-last>=30:
                print(f'{phase}: workers active; journal files preserve completed images.',flush=True);last=time.monotonic()
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
    records=[]
    for rank in range(cfg['workers']):
        records.extend(json.loads(line) for line in (Path(cfg['output'])/f'{phase}_{rank}.jsonl').read_text(encoding='utf-8').splitlines())
    return merge_records(records,cfg['images'])


def score_and_select(bundle, gt, detections, output, chair, locked=None):
    from captioning.hoi_caption_reranking import rerank_with_hoi, extract_claims
    from captioning.object_consistency import index_detections,extract_caption_objects
    from captioning.metrics import score_files
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    source=index_detections(json.loads(Path(detections).read_text(encoding='utf-8')))
    records=bundle['data']
    if any(r['filename'] not in source for r in records): raise ValueError('OCC detections missing images.')
    # One tokenizer invocation for references and all beam candidates; no reference
    # text enters claim extraction, UPT inference or per-candidate evidence.
    truth=json.loads(Path(gt).read_text());refs={r['image_id']:[] for r in records}
    for item in truth['annotations']: refs[item['image_id']].append({'caption':item['caption']})
    tokenized=PTBTokenizer().tokenize(refs)
    beam_text={r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in records}
    beam_tokens=PTBTokenizer().tokenize(beam_text)
    selections={};trial=[]
    if locked is None:
        grid=[0.,.05,.1,.2]
        for weight in grid:
            selected=[rerank_with_hoi(r,source[r['filename']],weight)[0] for r in records]
            pred={r['image_id']:[beam_tokens[r['image_id']][c['original_rank']]] for r,c in zip(records,selected)}
            cider,_=Cider().compute_score(tokenized,pred)
            trial.append({'upt_weight':weight,'CIDEr':float(cider),
                'changed_vs_rank0':sum(c['original_rank']!=0 for c in selected)})
            selections[weight]=selected
            print('VALIDATION GRID:',json.dumps(trial[-1]),flush=True)
        best=max(trial,key=lambda r:(r['CIDEr'],-r['upt_weight']))
        save(output/'locked_validation_selection.json',{'split':'val','selection_metric':'CIDEr; tie -> smaller UPT weight',
            'grid_fixed_before_run':grid,'fixed_base':{'decoder':.3,'clip':.4,'occ':.3},
            'upt_weight':best['upt_weight'],'trials':trial,'selection_on_validation_only':True,
            'candidate_sha256':sha(output/f"{bundle['metadata']['split']}_hoi_candidates.json"),'ground_truth_sha256':sha(gt),
            'test_run':False})
    else:
        weight=locked['upt_weight']
        selections={w:[rerank_with_hoi(r,source[r['filename']],w)[0] for r in records] for w in (0.,weight)}
        best={'upt_weight':weight,'selected_on':'validation','retuned_on_test':False}
        save(output/'applied_validation_selection.json',locked)
    chosen={'rank0':[r['candidates'][0] for r in records],'fixed_rerank':selections[0.], 'upt_rerank':selections[best['upt_weight']]}
    metrics={};chair_outputs={}
    chair_cache=chair/'chair.pkl'
    # Cached evaluator must cover EVERY actual COCO ID, not the eval row IDs.
    check="import sys,pickle,json;sys.path.insert(0,sys.argv[1]);from chair import CHAIR;obj=pickle.load(open(sys.argv[2],'rb'));ids=json.load(open(sys.argv[3]));assert all(r['coco_id'] in obj.imid_to_objects for r in ids['data']),'CHAIR cache lacks COCO IDs';print('CHAIR full-ID coverage PASS')"
    command([sys.executable,'-c',check,chair,chair_cache,output/f"{bundle['metadata']['split']}_hoi_candidates.json"])
    for name,selected in chosen.items():
        path=output/f'{name}_predictions.json'
        save(path,[{'image_id':r['image_id'],'caption':c['caption']} for r,c in zip(records,selected)])
        metric=score_files(path,gt,output/f'{name}_metrics.json')
        if name!='rank0' and locked is None:
            selected_weight=0. if name=='fixed_rerank' else best['upt_weight']
            grid_score=next(t['CIDEr'] for t in trial if t['upt_weight']==selected_weight)
            if abs(metric['CIDEr']-grid_score)>1e-10: raise ValueError('Grid/final CIDEr differs.')
        chair_input=output/f'{name}_chair_input.json';chair_output=output/f'{name}_chair_output.json'
        save(chair_input,[{'image_id':r['coco_id'],'caption':c['caption']} for r,c in zip(records,selected)])
        command([sys.executable,'-u',chair/'chair.py','--cap_file',chair_input,'--cache',chair_cache,'--save_path',chair_output],cwd=chair)
        chair_data=json.loads(chair_output.read_text(encoding='utf-8'));chair_outputs[name]=chair_data
        if {r['image_id'] for r in chair_data['sentences']}!={r['coco_id'] for r in records}: raise ValueError('CHAIR output IDs differ.')
        metrics[name]={'caption':metric,'chair':chair_data['overall_metrics'],
            'mean_words':sum(len(c['caption'].split()) for c in selected)/len(selected),
            'captions_with_parsed_hoi':sum(bool(extract_claims(c['caption'])) for c in selected),
            'mean_explicit_object_classes':sum(len(extract_caption_objects(c['caption'])) for c in selected)/len(selected),
            'changed_vs_rank0':sum(c['caption']!=r['candidates'][0]['caption'] for r,c in zip(records,selected))}
    transitions={}
    for baseline in ('rank0','fixed_rerank'):
        left={r['image_id']:r for r in chair_outputs[baseline]['sentences']}
        right={r['image_id']:r for r in chair_outputs['upt_rerank']['sentences']}
        transitions[baseline]={'hallucination_removed':sum(left[i]['metrics']['CHAIRs']==1 and right[i]['metrics']['CHAIRs']==0 for i in left),
            'hallucination_introduced':sum(left[i]['metrics']['CHAIRs']==0 and right[i]['metrics']['CHAIRs']==1 for i in left),
            'caption_changed':sum(left[i]['caption']!=right[i]['caption'] for i in left)}
    result={'metrics':metrics,'selection':best,'trials':trial,'transitions_upt_vs_baseline':transitions,
        'retuned_on_test':False}
    if locked is None:
        result.update(historical_rank0_val_CIDEr=1.1740115042,
            historical_rank0_reproduced_within_1e_5=abs(metrics['rank0']['caption']['CIDEr']-1.1740115042)<1e-5)
    return result


def run(a):
    locked=load_test_selection(a.locked_selection) if a.split=='test' else None
    import torch
    repo=Path(__file__).resolve().parent;output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    paths={key:str(Path(getattr(a,key)).resolve()) for key in ('checkpoint','prompt_cache_path','dataset_json_path','base_path','visual_cache','detections','upt_directory','pocket_directory','upt_checkpoint','chair_directory')}
    for key in ('checkpoint','prompt_cache_path','dataset_json_path','detections','upt_checkpoint'):
        if not Path(paths[key]).is_file(): raise FileNotFoundError(f'{key}: {paths[key]}')
    if torch.cuda.device_count()<a.workers: raise RuntimeError(f'Need {a.workers} CUDA GPUs; found {torch.cuda.device_count()}.')
    payload=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False)
    profile={k:payload.get(k) for k in ('epoch','visual_adapter','num_visual_queries','qformer_layers')}
    if profile!={'epoch':10,'visual_adapter':'qformer','num_visual_queries':32,'qformer_layers':2}: raise ValueError(f'Wrong baseline checkpoint: {profile}')
    if any(payload.get(k,False) for k in ('prompt_conditioned_qformer','object_semantic_alignment','cascade_semantic_alignment','grounded_relation_alignment')) or float(payload.get('itc_weight',0))!=0:
        raise ValueError('Expected original Q-Former baseline without auxiliary branches.')
    if float(payload.get('alignment_weight',0))!=0 or int(payload.get('max_train_batches',0))!=0:
        raise ValueError('Expected full original baseline training without a region loss.')
    del payload
    dataset=json.loads(Path(paths['dataset_json_path']).read_text(encoding='utf-8'))
    images=[];annotations=[];annotation_id=0
    for eval_id,item in enumerate(dataset['images']):
        if item['split']!=a.split: continue
        if a.limit and len(images)>=a.limit: break
        coco_id=int(Path(item['filename']).stem.rsplit('_',1)[1])
        images.append({'image_id':eval_id,'coco_id':coco_id,'filename':item['filename'],'filepath':item['filepath']})
        for sent in item['sentences'][:5]:
            annotations.append({'id':annotation_id,'image_id':eval_id,'caption':sent['raw']});annotation_id+=1
        if not (Path(paths['base_path'])/item['filepath']/item['filename']).is_file(): raise FileNotFoundError(item['filename'])
    if not images or (not a.limit and len(images)!=5000): raise ValueError(f'Expected Karpathy {a.split}5000.')
    from captioning.upt_reference import UPT_REVISION,POCKET_REVISION
    refs={paths['upt_directory']:UPT_REVISION,paths['pocket_directory']:POCKET_REVISION,
        str(Path(paths['upt_directory'])/'detr'):'820d17abc697e8dd5f49d4cd1fdbde412545bac5',
        str(Path(paths['upt_directory'])/'hicodet'):'9dd59c1f6ee2ddc011567765d0bc644858391fea',
        paths['chair_directory']:'4087a26211aa2339b9a76307cb8f0321ef691d0a'}
    for directory,revision in refs.items():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()!=revision: raise ValueError('Reference revision changed.')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=directory,text=True).strip(): raise ValueError('Reference source modified.')
    input_hashes={k:sha(paths[k]) for k in ('checkpoint','prompt_cache_path','dataset_json_path','detections','upt_checkpoint')}
    if input_hashes['upt_checkpoint']!='12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c': raise ValueError('UPT checkpoint hash differs.')
    if locked is not None:
        identity=json.loads((repo/'audit/hoi_validation_input_identity.json').read_text(encoding='utf-8'))
        if input_hashes!=identity['input_sha256']: raise ValueError('Test model/data/detection identity differs from validation.')
    from captioning.hoi_caption_reranking import METHOD
    cfg={**paths,'output':str(output.resolve()),'workers':a.workers,'images':images,'require_gpu':True,
        'input_sha256':input_hashes,'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'reference_revisions':refs,'method':METHOD,'split':a.split,'beam_size':5,'candidate_count':5,
        'base_weights':[.3,.4,.3],'upt_grid':[0.,.05,.1,.2] if locked is None else None,
        'locked_selection_sha256':sha(a.locked_selection) if locked is not None else None,
        'locked_upt_weight':locked['upt_weight'] if locked is not None else None,
        'reuse_scored_sha256':sha(a.reuse_scored_candidates) if a.reuse_scored_candidates else None,
        'reuse_protocol_sha256':sha(a.reuse_protocol) if a.reuse_protocol else None,
        'clip_revision':'57c216476eefef5ab752ec549e440a49ae4ae5f3',
        'visual_shards':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(Path(paths['visual_cache']).glob('*.h5'))],
        'ground_truth_use':'Metrics only; no test selection.' if locked else 'Metrics and validation weight selection only; never HOI extraction.',
        'evidence':'Max raw logit per mentioned class over actual detected person-object pairs; signed tanh contrast against strongest different trained object class for SAME predicate/image. Missing or no competitor neutral. Conservative lexical parser; no unique instance binding.',
        'test_run':a.split=='test','training_performed':False}
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=cfg: raise ValueError('Run inputs changed; choose a fresh output directory.')
    save(protocol,cfg)
    gt=output/f'{a.split}_gt.json';save(gt,{'info':{},'licenses':[],'type':'captions','images':[{'id':r['image_id']} for r in images],'annotations':annotations})
    metadata={'split':a.split,'count':len(images),'beam_size':5,'candidate_count':5,'visual_adapter':'qformer','epoch':10,
        'num_visual_queries':32,'qformer_layers':2,'prompt_conditioned_qformer':False,'protocol_sha256':sha(protocol),'checkpoint_sha256':input_hashes['checkpoint']}
    scored=output/f'{a.split}_beam5_clipscore_candidates.json'
    if a.reuse_scored_candidates:
        load_reused_candidates(a.reuse_scored_candidates,a.reuse_protocol,cfg)
        if Path(a.reuse_scored_candidates).resolve()!=scored.resolve(): shutil.copyfile(a.reuse_scored_candidates,scored)
        print(f'Reused verified beam5+CLIP candidates for {len(images)} images; starting UPT.',flush=True)
    else:
        beam_records=run_workers(cfg,'beams',repo)
        raw=output/f'{a.split}_beam5_candidates.json';save(raw,{'metadata':metadata,'data':beam_records})
        command([sys.executable,'-u',repo/'score_clip_candidates.py','--candidates',raw,'--visual-cache',paths['visual_cache'],'--output',scored,'--device','cuda','--revision',cfg['clip_revision']])
    upt_cfg={**cfg,'clip_candidates':str(scored),'clip_candidates_sha256':sha(scored)}
    upt_records=run_workers(upt_cfg,'upt',repo)
    for original,record in zip(json.loads(scored.read_text())['data'],upt_records):
        if [{k:v for k,v in c.items() if k!='hoi_evidence'} for c in record['candidates']]!=original['candidates']: raise ValueError('UPT changed beam candidates or base scores.')
    bundle={'metadata':{**json.loads(scored.read_text())['metadata'],'hoi_method':METHOD,
        'validation_protocol_sha256' if a.split=='val' else 'test_protocol_sha256':sha(protocol)},'data':upt_records}
    save(output/f'{a.split}_hoi_candidates.json',bundle)
    result=score_and_select(bundle,gt,paths['detections'],output,Path(paths['chair_directory']),locked=locked)
    result.update(count=len(images),split=a.split,status='VALIDATION_COMPLETE' if a.split=='val' else 'TEST_COMPLETE',protocol_sha256=sha(protocol),
        caption_candidate_integrity_verified=True,training_performed=False,test_run=a.split=='test',
        upt_coverage={'images_without_pairs':sum(r.get('hoi_pair_count')==0 for r in upt_records),
            'images_with_available_parsed_claim':sum(any(c['hoi_evidence']['available_claims'] for c in r['candidates']) for r in upt_records),
            'images_with_nonzero_relation_feature':sum(any(c['hoi_evidence']['feature']!=0 for c in r['candidates']) for r in upt_records)},
        production_promotion=False,limitations=['Validation-tuned results; independent test pending.' if locked is None else 'Independent test with validation-frozen weight; no test tuning.','Heuristic lexical parsing with partial coverage.','Existential person/object-class support does not identify caption instance or validate attributes.','Missing detections/unsupported ontology are neutral, not evidence of absence.','HICO and COCO pretraining overlap not exhaustively audited.'])
    save(output/'run_summary.json',result)
    archive=output.parent/(output.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in {'.json','.jsonl','.log'}: z.write(path,path.name)
    print(f'FULL {a.split.upper()} COMPLETE',flush=True);print(json.dumps(result,indent=2),flush=True);print('Review ZIP:',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','prompt-cache-path','dataset-json-path','base-path','visual-cache','detections','upt-directory','pocket-directory','upt-checkpoint','chair-directory','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--locked-selection',default='')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--reuse-scored-candidates',default='');p.add_argument('--reuse-protocol',default='')
    args=p.parse_args()
    if args.workers<1 or args.limit<0:p.error('Invalid worker count or limit.')
    if bool(args.reuse_scored_candidates)!=bool(args.reuse_protocol):p.error('Reuse requires both candidates and their source protocol.')
    if (args.split=='test')!=bool(args.locked_selection):p.error('Test requires locked selection; validation must not receive it.')
    run(args)
