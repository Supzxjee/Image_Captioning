"""Manual-Kaggle learned ranker pilot; train/tune from train, assess on val512."""
import argparse
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from run_hoi_caption_validation import sha,save,command,run_workers
from captioning.learned_caption_ranker import CaptionRanker,MODES,ranking_loss,select_rank

INPUTS=('scores','fixed_scores','image_global','text_global','text_tokens','text_mask','patches','candidate_mask')


def cohort(dataset,seed=42):
    rows=[]
    for i,r in enumerate(dataset['images']):
        rows.append({'image_id':i,'coco_id':int(Path(r['filename']).stem.rsplit('_',1)[1]),
                     'filename':r['filename'],'filepath':r['filepath'],'split':r['split']})
    train=sorted([r for r in rows if r['split'] in ('train','restval')],key=lambda r:r['image_id'])
    val=sorted([r for r in rows if r['split']=='val'],key=lambda r:r['image_id'])
    rng=random.Random(seed);rng.shuffle(train);rng.shuffle(val)
    if len(train)<1280 or len(val)<512:raise ValueError('Incomplete dataset.')
    groups={'fit':train[:1024],'tune':train[1024:1280],'validation':val[:512]}
    sets=[{r['coco_id'] for r in v} for v in groups.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    return {k:sorted(v,key=lambda r:r['image_id']) for k,v in groups.items()}


def ground_truth(images,dataset):
    return {'info':{},'licenses':[],'type':'captions','images':[{'id':r['image_id']} for r in images],
        'annotations':[{'id':j*5+k,'image_id':r['image_id'],'caption':s['raw']}
          for j,r in enumerate(images) for k,s in enumerate(dataset['images'][r['image_id']]['sentences'][:5])]}


def cider_targets(records,gt):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    refs={r['image_id']:[] for r in records}
    for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
    assert all(len(v)==5 for v in refs.values())
    truth=PTBTokenizer().tokenize(refs)
    texts=PTBTokenizer().tokenize({r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in records})
    result=np.zeros((len(records),5),dtype=np.float64)
    for k in range(5):
        selected={r['image_id']:[texts[r['image_id']][min(k,len(r['candidates'])-1)]] for r in records}
        _,scores=Cider().compute_score(truth,selected);result[:,k]=scores
    return result


class Features(Dataset):
    def __init__(self,records,targets=None):self.records,self.targets=records,targets
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];data=torch.load(r['feature_file'],map_location='cpu',weights_only=True)
        assert data['image_id']==r['image_id'] and data['coco_id']==r['coco_id']
        result={k:data[k] for k in INPUTS}
        if self.targets is not None:result['targets']=torch.tensor(self.targets[i],dtype=torch.float32)
        return result


def device_batch(b):return {k:v.cuda(non_blocking=True) for k,v in b.items()}


def choices(model,records,mode=None):
    result=[];model.eval()
    with torch.no_grad():
        for b in DataLoader(Features(records),batch_size=8,shuffle=False,num_workers=0):
            b=device_batch(b)
            scores=b['fixed_scores'] if mode=='fixed' else model(b)[0]
            result.extend(select_rank(scores,b['candidate_mask']).cpu().tolist())
    return result


def fit(mode,train,tune,train_targets,tune_targets,output):
    torch.manual_seed(42);np.random.seed(42)
    model=CaptionRanker(mode).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(Features(train,train_targets),batch_size=8,shuffle=True,num_workers=0,generator=generator)
    path=output/f'{mode}_ranker.pt';history=[]
    def assess(epoch):
        pick=choices(model,tune)
        return {'epoch':epoch,'tune_CIDEr':float(tune_targets[np.arange(len(tune)),pick].mean())}
    best=assess(0)
    def checkpoint():
        torch.save({'mode':mode,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'selection':best,'protocol_sha256':sha(output/'protocol.json')},path)
    checkpoint();history.append(best.copy())
    for epoch in range(1,11):
        model.train();total=0.
        for b in loader:
            b=device_batch(b);optimizer.zero_grad(set_to_none=True)
            scores,residual=model(b)
            loss=ranking_loss(scores,b['targets'],b['candidate_mask'],residual)
            if not torch.isfinite(loss):raise ValueError('Nonfinite ranker loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())
        record=assess(epoch);record['training_loss']=total/len(loader);history.append(record)
        if record['tune_CIDEr']>best['tune_CIDEr']+1e-12:best=record.copy();checkpoint()
        print(mode,json.dumps(record),flush=True)
    saved=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'],strict=True)
    save(output/f'{mode}_history.json',history)
    return model,{'selected_epoch':saved['selection']['epoch'],'tune_CIDEr':saved['selection']['tune_CIDEr'],
                  'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(path)}


def prepare_features(a):
    if torch.cuda.device_count()<2:raise RuntimeError('Need T4 x2.')
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    repo=Path(__file__).resolve().parent;output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    paths={k:str(Path(getattr(a,k)).resolve()) for k in ('checkpoint','prompt_cache_path','dataset_json_path','base_path','visual_cache','detections','chair_directory')}
    hashes={k:sha(paths[k]) for k in ('checkpoint','prompt_cache_path','dataset_json_path','detections')}
    identity=json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    assert all(hashes[k]==identity[k] for k in hashes),'Original baseline input hash differs.'
    chair_revision='4087a26211aa2339b9a76307cb8f0321ef691d0a'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.chair_directory,text=True).strip()==chair_revision
    dataset=json.loads(Path(a.dataset_json_path).read_text());groups=cohort(dataset)
    images=[r for group in groups.values() for r in group]
    assert len({r['image_id'] for r in images})==1792 and all(r['split']!='test' for r in images)
    cfg={**paths,'output':str(output.resolve()),'images':images,'workers':2,'require_gpu':True,
         'variant':'learned_caption_ranker_pilot_v1','seed':42,'input_sha256':hashes,
         'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         'clip_revision':'57c216476eefef5ab752ec549e440a49ae4ae5f3','groups':groups,
         'chair_revision':chair_revision,
         'feature_schema':['decoder_minmax','clip_minmax','OCC_scaled','clipped_avg_logprob_div10','length_div30','rank_div4'],
         'architecture':{'hidden':128,'attention_heads':4,'caption_tokens':77,'visual_patches':196},
         'visual_shards':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(Path(a.visual_cache).glob('*.h5'))],
         'training':'1024 train/restval images;256 disjoint train/restval for epoch selection;512 reused validation for final pilot only',
         'modes':list(MODES),'epochs':10,'include_epoch0_fixed_baseline':True,'residual_scale':.2,
         'learning_rate':.001,'loss':'CIDEr-weighted pairwise softplus + .01 residual-square; CIDEr preferences not verified negatives',
         'test_used':False,'qformer_or_CLIP_training':False,'manual_execution_only':True}
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=cfg:raise ValueError('Changed protocol;use fresh output directory.')
    save(protocol,cfg)
    beam=run_workers(cfg,'beams',repo,worker_script='learned_ranker_worker.py')
    save(output/'beam_candidates.json',{'metadata':{'count':len(beam)},'data':beam})
    records=run_workers(cfg,'features',repo,worker_script='learned_ranker_worker.py')
    for r,original in zip(records,beam):
        assert [{k:v for k,v in c.items() if k!='clipscore'} for c in r['candidates']]==original['candidates']
        assert sha(r['feature_file'])==r['feature_sha256']
    save(output/'feature_candidates.json',{'metadata':{'count':len(records)},'data':records})
    return cfg,dataset,groups,records


def run(a):
    cfg,dataset,groups,records=prepare_features(a)
    output=Path(a.output)
    lookup={r['image_id']:r for r in records}
    subsets={name:[lookup[r['image_id']] for r in group] for name,group in groups.items()}
    train,tune=subsets['fit'],subsets['tune']
    train_targets=cider_targets(train,ground_truth(groups['fit'],dataset))
    tune_targets=cider_targets(tune,ground_truth(groups['tune'],dataset))
    models={};selections={}
    for mode in MODES:models[mode],selections[mode]=fit(mode,train,tune,train_targets,tune_targets,output)
    # Validation references enter metrics only AFTER all model epochs are frozen.
    val=subsets['validation'];gt=output/'validation_gt.json';save(gt,ground_truth(groups['validation'],dataset))
    from captioning.metrics import score_files
    measured={};chosen={};chair=Path(a.chair_directory)
    check="import sys,pickle,json;sys.path.insert(0,sys.argv[1]);from chair import CHAIR;x=pickle.load(open(sys.argv[2],'rb'));r=json.load(open(sys.argv[3]));assert all(i['coco_id'] in x.imid_to_objects for i in r)"
    save(output/'validation_images.json',groups['validation'])
    command([sys.executable,'-c',check,chair,chair/'chair.pkl',output/'validation_images.json'])
    for name in ('fixed',*MODES):
        pick=choices(models['scores'] if name=='fixed' else models[name],val,mode='fixed' if name=='fixed' else None)
        chosen[name]=pick
        selected=[{'image_id':r['image_id'],'caption':r['candidates'][k]['caption']} for r,k in zip(val,pick)]
        pred=output/f'{name}_predictions.json';save(pred,selected)
        metric=score_files(pred,gt,output/f'{name}_metrics.json')
        ci=output/f'{name}_chair_input.json';co=output/f'{name}_chair_output.json'
        save(ci,[{'image_id':r['coco_id'],'caption':r['candidates'][k]['caption']} for r,k in zip(val,pick)])
        command([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair)
        evaluated=json.loads(co.read_text());assert {r['image_id'] for r in evaluated['sentences']}=={r['coco_id'] for r in val}
        measured[name]={'caption':metric,'chair':evaluated['overall_metrics'],
                        'changed_vs_fixed':sum(i!=j for i,j in zip(pick,chosen['fixed']))}
    regional=measured['regional'];fixed=measured['fixed']
    go=(regional['caption']['CIDEr']>=fixed['caption']['CIDEr']+.005
        and regional['chair']['CHAIRs']<=fixed['chair']['CHAIRs']
        and regional['chair']['CHAIRi']<=fixed['chair']['CHAIRi']
        and regional['chair']['Recall']>=fixed['chair']['Recall']-.01
        and all(regional['caption']['CIDEr']>measured[m]['caption']['CIDEr'] for m in ('scores','global')))
    result={'status':'PILOT_COMPLETE','fit_images':1024,'tune_images':256,'validation_images':512,
            'protocol_sha256':sha(protocol),'selections':selections,'metrics':measured,'expansion_gate_passed':go,
            'gate':'regional CIDEr >= fixed+.005;CHAIRs/i <= fixed;Recall >= fixed-.01;regional CIDEr > scores and global',
            'test_used':False,'full_training_started':False,'production_promotion':False,
            'limits':['One seed, reused small validation;no new test result.','CIDEr preferences are weak supervision,not verified wrong relations.',
                      'Contextual cached patch attention is not unique-instance grounding or a crop CLIP feature.']}
    save(output/'run_summary.json',result)
    with zipfile.ZipFile(str(output)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.pt'):
                z.write(p,p.name)
    print(json.dumps(result,indent=2),flush=True);print('RETURN FILE:',str(output)+'_review.zip',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('checkpoint','prompt-cache-path','dataset-json-path','base-path','visual-cache','detections','chair-directory','output'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
