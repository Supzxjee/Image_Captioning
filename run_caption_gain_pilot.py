"""Manual baseline-relative gain pilot; reuse frozen V1 features when present."""
import argparse
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from captioning.caption_gain_ranker import CaptionGainRanker,gain_loss,propose,calibrate
from captioning.learned_caption_ranker import MODES
from captioning.fixed_idf_targets import targets as fixed_targets
from run_learned_caption_ranker_pilot import Features,device_batch,cohort,ground_truth,cider_targets,prepare_features
from run_hoi_caption_validation import sha,save,command


def split_groups(original):
    ordered=sorted(original['tune'],key=lambda r:r['image_id'])
    random.Random(20261015).shuffle(ordered)
    result={'fit':original['fit'],'tune':ordered[:128],'calibration':ordered[128:],'validation':original['validation']}
    assert {k:len(v) for k,v in result.items()}=={'fit':1024,'tune':128,'calibration':128,'validation':512}
    assert len({r['coco_id'] for v in result.values() for r in v})==1792
    return {k:sorted(v,key=lambda r:r['image_id']) for k,v in result.items()}


def source_cache(a,dataset):
    source=Path(a.source)
    if source.is_dir() and not a.regenerate_features:
        cfg=json.loads((source/'protocol.json').read_text())
        assert cfg['variant']=='learned_caption_ranker_pilot_v1'
        assert cfg['groups']==cohort(dataset),'Source cohort differs from frozen V1.'
        rows=json.loads((source/'feature_candidates.json').read_text())['data']
        assert len(rows)==1792
        adjusted=[]
        for r in rows:
            file=source/'private_features'/f"{r['image_id']}.pt"
            if not file.is_file():raise FileNotFoundError(f'{file}: full output cache required, not compact review ZIP.')
            if sha(file)!=r['feature_sha256']:raise ValueError('Feature bytes differ.')
            adjusted.append({**r,'feature_file':str(file)})
        print('Reusing all1792 hash-verified feature files;no beam/CLIP recomputation.',flush=True)
        return cfg,adjusted,source,False
    print('Previous feature output not mounted. Regenerating frozen beams/CLIP only;no V1 heads trained.',flush=True)
    kwargs=vars(a).copy();kwargs['output']=str(Path(a.output)/'private_source_cache')
    cfg,_,_,records=prepare_features(SimpleNamespace(**kwargs))
    return cfg,records,Path(kwargs['output']),True


def predict(model,records):
    model.eval();gains=[];baselines=[]
    with torch.no_grad():
        for b in DataLoader(Features(records),batch_size=8,shuffle=False,num_workers=0):
            g,base=model(device_batch(b));gains.append(g.cpu());baselines.append(base.cpu())
    return torch.cat(gains),torch.cat(baselines)


def snapshot(path,records,gains,base,correction):
    raw=propose(gains,base);guard=propose(gains,base,correction)
    save(path,[{'image_id':r['image_id'],'baseline_rank':int(b),'raw_rank':int(p),'guarded_rank':int(q),
                'predicted_gains':g[:len(r['candidates'])].tolist()} for r,g,b,p,q in zip(records,gains,base,raw,guard)])


def fit(mode,train,tune,labels,tune_labels,out):
    torch.manual_seed(42);model=CaptionGainRanker(mode).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(Features(train,labels),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    def assess(epoch):
        g,b=predict(model,tune);r=propose(g,b).numpy()
        return {'epoch':epoch,'tune_train_IDF_CIDEr':float(tune_labels[np.arange(len(tune)),r].mean())}
    best=assess(0);history=[best.copy()];file=out/f'{mode}_gain.pt'
    def checkpoint():torch.save({'mode':mode,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                'selection':best,'protocol_sha256':sha(out/'protocol.json')},file)
    checkpoint()
    for epoch in range(1,11):
        model.train();total=0.
        for b in loader:
            b=device_batch(b);opt.zero_grad(set_to_none=True);g,base=model(b)
            loss=gain_loss(g,base,b['targets'],b['candidate_mask'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite gain loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())
        record=assess(epoch);record['loss']=total/len(loader);history.append(record)
        if record['tune_train_IDF_CIDEr']>best['tune_train_IDF_CIDEr']+1e-12:best=record.copy();checkpoint()
        print(mode,json.dumps(record),flush=True)
    payload=torch.load(file,map_location='cpu',weights_only=True);model.load_state_dict(payload['state_dict'],strict=True)
    save(out/f'{mode}_history.json',history)
    return model,{'epoch':payload['selection']['epoch'],'tune_train_IDF_CIDEr':payload['selection']['tune_train_IDF_CIDEr'],'checkpoint_sha256':sha(file)}


def run(a):
    from captioning.scst_runtime import configure_strict_runtime, runtime_manifest
    configure_strict_runtime(getattr(a,'strict_reproducibility',False))
    out=Path(a.output)
    if (out/'protocol.json').exists() or (out/'run_summary.json').exists():
        raise FileExistsError('Use a fresh gain-pilot output; completed or started runs are not overwritten.')
    if not torch.cuda.is_available():raise RuntimeError('GPU required.')
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parent
    save(out/'runtime.json',runtime_manifest())
    dataset=json.loads(Path(a.dataset_json_path).read_text())
    cfg,records,source,regenerated=source_cache(a,dataset)
    identity=json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    assert all(v==identity[k] for k,v in cfg['input_sha256'].items())
    assert sha(a.dataset_json_path)==cfg['input_sha256']['dataset_json_path']
    lookup={r['image_id']:r for r in records};groups=split_groups(cfg['groups'])
    assert set(lookup)=={r['image_id'] for v in groups.values() for r in v}
    assert all(r['split'] in ('train','restval') for k,v in groups.items() if k!='validation' for r in v)
    chair=Path(a.chair_directory);chair_revision='4087a26211aa2339b9a76307cb8f0321ef691d0a'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=chair,text=True).strip()==chair_revision
    protocol={'variant':'caption_gain_guard_v2','project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'source_protocol_sha256':sha(source/'protocol.json'),'groups':groups,'source_directory':str(source),
        'source_input_sha256':cfg['input_sha256'],'feature_cache_regenerated':regenerated,
        'epoch_selection':'train tune128 only;epoch0..10;max raw-policy train-IDF CIDEr;tie earlier',
        'supervision_IDF':'Frozen from fit1024 references only;normalizer1024 shared by fit/tune/calibration;final metrics use standard validation-IDF scorer',
        'calibration':'disjoint128 train images;90% one-sided order statistic of selected-proposal error;clamp correction>=0',
        'loss':'SmoothL1 beta=.1 on CIDEr gain relative to fixed;2x weight for negative true gains;no semantic-negative labels',
        'nominal_coverage':.9,'selective_precision_guaranteed':False,'modes':list(MODES),
        'primary_policy':'regional_guarded','chair_revision':chair_revision,'test_used':False,
        'validation_scope':'same inspected512 images;development only,not independent confirmation',
        'seed':42,'epochs':10,'lr':.001,'weight_decay':.0001,'batch_size':8,
        'strict_reproducibility':getattr(a,'strict_reproducibility',False),
        'frozen_generator':'original QFormer epoch10; not SCST count0.5',
        'cache_runtime_scope':'Existing cache retains its generation runtime; strict flag controls heads and newly regenerated features only.',
        'gate':'guarded regional CIDEr>=fixed+.005,CHAIRs/i<=fixed,Recall>=fixed-.01,CIDEr>guarded scores/global'}
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Changed protocol;choose fresh output.')
    save(pp,protocol);save(out/'feature_candidates.json',{'data':records,'metadata':{'count':1792}})
    subsets={k:[lookup[r['image_id']] for r in v] for k,v in groups.items()}
    # Never build validation targets for fitting or calibration.
    targets={};fit_idf=None
    for name in ('fit','tune','calibration'):
        targets[name],fit_idf=fixed_targets(subsets[name],ground_truth(groups[name],dataset),fit_idf)
        save(out/f'{name}_CIDEr_targets.json',{'IDF_source':'fit1024 only','image_ids':[r['image_id'] for r in subsets[name]],'scores':targets[name].tolist()})
    save(out/'locked_fit_IDF.json',fit_idf)
    models={};selections={};policies={}
    for mode in MODES:
        model,selections[mode]=fit(mode,subsets['fit'],subsets['tune'],targets['fit'],targets['tune'],out)
        g,b=predict(model,subsets['calibration'])
        policies[mode]=calibrate(g.numpy(),b.numpy(),targets['calibration'])
        snapshot(out/f'{mode}_calibration_scores.json',subsets['calibration'],g,b,policies[mode]['correction'])
        tg,tb=predict(model,subsets['tune']);snapshot(out/f'{mode}_tune_scores.json',subsets['tune'],tg,tb,0.)
        models[mode]=model
    save(out/'locked_gain_policies.json',policies)
    # All epochs and corrections locked before evaluation references are scored.
    val=subsets['validation'];selections_by_policy={}
    for mode,model in models.items():
        g,b=predict(model,val)
        if 'fixed' in selections_by_policy:assert selections_by_policy['fixed']==b.tolist()
        selections_by_policy['fixed']=b.tolist()
        raw=propose(g,b);guard=propose(g,b,policies[mode]['correction'])
        assert ((guard==b)|(guard==raw)).all()
        selections_by_policy[mode+'_raw']=raw.tolist();selections_by_policy[mode+'_guarded']=guard.tolist()
        snapshot(out/f'{mode}_validation_scores.json',val,g,b,policies[mode]['correction'])
    gt=out/'validation_gt.json';save(gt,ground_truth(groups['validation'],dataset));save(out/'validation_images.json',groups['validation'])
    from captioning.metrics import score_files
    check="import sys,pickle,json;sys.path.insert(0,sys.argv[1]);from chair import CHAIR;x=pickle.load(open(sys.argv[2],'rb'));r=json.load(open(sys.argv[3]));assert all(i['coco_id'] in x.imid_to_objects for i in r)"
    command([sys.executable,'-c',check,chair,chair/'chair.pkl',out/'validation_images.json'])
    metrics={}
    for name,pick in selections_by_policy.items():
        pred=out/f'{name}_predictions.json';save(pred,[{'image_id':r['image_id'],'caption':r['candidates'][k]['caption']} for r,k in zip(val,pick)])
        caption=score_files(pred,gt,out/f'{name}_metrics.json')
        ci=out/f'{name}_chair_input.json';co=out/f'{name}_chair_output.json'
        save(ci,[{'image_id':r['coco_id'],'caption':r['candidates'][k]['caption']} for r,k in zip(val,pick)])
        command([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair)
        data=json.loads(co.read_text());assert {r['image_id'] for r in data['sentences']}=={r['coco_id'] for r in val}
        metrics[name]={'caption':caption,'chair':data['overall_metrics'],'changed_vs_fixed':sum(i!=j for i,j in zip(pick,selections_by_policy['fixed']))}
    r=metrics['regional_guarded'];f=metrics['fixed']
    gate=(r['caption']['CIDEr']>=f['caption']['CIDEr']+.005 and r['chair']['CHAIRs']<=f['chair']['CHAIRs']
        and r['chair']['CHAIRi']<=f['chair']['CHAIRi'] and r['chair']['Recall']>=f['chair']['Recall']-.01
        and all(r['caption']['CIDEr']>metrics[k+'_guarded']['caption']['CIDEr'] for k in ('scores','global')))
    summary={'status':'GAIN_PILOT_COMPLETE','fit_images':1024,'tune_images':128,'calibration_images':128,'validation_images':512,
        'protocol_sha256':sha(pp),'locked_policies_sha256':sha(out/'locked_gain_policies.json'),
        'fit_IDF_sha256':sha(out/'locked_fit_IDF.json'),'selections':selections,'locked_policies':policies,'metrics':metrics,
        'primary_policy':'regional_guarded','expansion_gate_passed':gate,'test_used':False,'production_promotion':False,
        'limitations':['Repeated inspected validation512;not independent confirmation.','Calibration images are train/restval;distribution may differ from validation.',
            'Error quantile is not a90% precision guarantee among changed captions.','CIDEr gain is weak preference supervision,not visual truth.']}
    save(out/'run_summary.json',summary)
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.pt'):z.write(p,p.name)
    print(json.dumps(summary,indent=2),flush=True);print('RETURN FILE:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--regenerate-features',action='store_true')
    p.add_argument('--strict-reproducibility',action='store_true')
    for n in ('checkpoint','prompt-cache-path','dataset-json-path','base-path','visual-cache','detections','chair-directory','output'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
