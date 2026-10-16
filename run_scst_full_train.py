"""Manual full-train SCST comparison, tune checkpoint then locked validation remainder."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import zipfile
from run_scst_comparison import cohorts, train_pair, evaluate
from run_hoi_caption_validation import save, sha


def full_groups(dataset):
    train_count=sum(r['split'] in ('train','restval') for r in dataset['images'])
    tune=cohorts(dataset,train_count,1000)
    tune_ids={r['image_id'] for r in tune['val']}
    remainder=[{'image_id':i,'coco_id':int(Path(r['filename']).stem.rsplit('_',1)[1]),
                'filename':r['filename'],'filepath':r['filepath']}
               for i,r in enumerate(dataset['images']) if r['split']=='val' and i not in tune_ids]
    if len(tune['val'])!=1000 or len(remainder)!=4000: raise ValueError('Expected validation1000+4000.')
    combined=tune['train']+tune['val']+remainder
    if len({r['coco_id'] for r in combined})!=len(combined): raise ValueError('Overlapping image cohorts.')
    return {**tune,'validation_remainder':remainder}


def select_checkpoint(metrics, candidates):
    """Baseline is an eligible fallback; tie keeps earlier checkpoint."""
    baseline=metrics['original_fixed']
    best='original'
    eligible=[]
    for name in candidates:
        value=metrics[name+'_fixed']
        safe=(all(value['chair'][k]<=baseline['chair'][k] for k in ('CHAIRs','CHAIRi')) and
              value['chair']['Recall']>=baseline['chair']['Recall']-.01)
        if safe:
            eligible.append(name)
            if value['caption']['CIDEr']>metrics[best+'_fixed']['caption']['CIDEr']+1e-12:best=name
    return best,eligible


def run(a):
    repo=Path(__file__).resolve().parent;out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Use a fresh full-train OUTPUT directory.')
    paths={k:str(Path(getattr(a,k)).resolve()) for k in ('checkpoint','prompt_cache_path','visual_cache',
           'dataset_json_path','base_path','detections','chair_directory')}
    for k in ('checkpoint','prompt_cache_path','dataset_json_path','detections'):
        if not Path(paths[k]).is_file():raise FileNotFoundError(paths[k])
    for k in ('visual_cache','base_path','chair_directory'):
        if not Path(paths[k]).is_dir():raise FileNotFoundError(paths[k])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=paths['chair_directory'],text=True).strip()!='4087a26211aa2339b9a76307cb8f0321ef691d0a':
        raise ValueError('CHAIR revision mismatch.')
    identity=json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    hashes={k:sha(paths[k]) for k in ('checkpoint','prompt_cache_path','dataset_json_path','detections')}
    if any(identity[k]!=v for k,v in hashes.items()):raise ValueError('Pinned baseline input mismatch.')
    dataset=json.loads(Path(paths['dataset_json_path']).read_text(encoding='utf-8'))
    groups=full_groups(dataset)
    if len(groups['train'])!=113287:raise ValueError('Expected pinned COCO113287 training/restval images.')
    from captioning.visual_cache import VisualCache
    cache=VisualCache([paths['visual_cache']],id_key='coco_id')
    try:cache.require_ids([r['coco_id'] for rows in groups.values() for r in rows],'full train and validation')
    finally:cache.close()
    from captioning.object_reward import make_targets, _AnnotationReader
    targets=make_targets({'train':groups['train'],'val':groups['val']+groups['validation_remainder']},
                         dataset,Path(paths['chair_directory'])/'chair.pkl',
                         json.loads(Path(paths['detections']).read_text(encoding='utf-8')),'aliases_v2','unique_count_v2')
    with (Path(paths['chair_directory'])/'chair.pkl').open('rb') as stream:annotations=_AnnotationReader(stream).load()
    if not all(r['coco_id'] in annotations.imid_to_objects for r in groups['val']+groups['validation_remainder']):
        raise ValueError('CHAIR validation coverage incomplete.')
    del annotations
    updates=math.ceil(len(groups['train'])/8);half=math.ceil(updates/2)
    cfg={**paths,'output':str(out),'groups':groups,'input_sha256':hashes,
         'variant':'scst_ce_count_w05_full_train_v1','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         'batch_size':8,'lr':5e-6,'epochs':1,'seed':42,'object_reward_comparison':True,
         'scst_weight':.05,'ce_weight':1.,'object_reward_weight':.5,'object_reward_parser':'aliases_v2',
         'object_reward_penalty':'unique_count_v2','strict_reproducibility':True,'train_same_gpu':True,
         'reward_reference_cache_size':256,'snapshot_steps':[half],
         'clip_revision':'57c216476eefef5ab752ec549e440a49ae4ae5f3',
         'object_reward_metadata':targets['metadata'],'test_used':False,
         'selection':'Fixed CIDEr maximum among original/mid/end on tune1000, with CHAIRs/i <= original and recall >= original-.01; ties earlier; baseline fallback',
         'evaluation':'Lock both choices before evaluating validation remainder4000; no test; fixed reranker unchanged',
         'validation_scope':'Reused development validation, not never-seen confirmation; report tune1000 and remainder4000 separately; no averaging different-IDF CIDEr',
         'manual_execution_only':True}
    out.mkdir(parents=True)
    save(out/'train_object_targets.json',targets)
    cfg.update(train_object_targets=str(out/'train_object_targets.json'),
               train_object_targets_sha256=sha(out/'train_object_targets.json'))
    protocol=out/'protocol.json';save(protocol,cfg)
    del dataset,targets
    try:
        train_pair(cfg,protocol,repo)
        checkpoints={'original':paths['checkpoint']}
        for arm in ('scst_ce','scst_ce_object'):
            checkpoints[arm+'_mid']=str(out/arm/f'checkpoint_step_{half}.pth')
            checkpoints[arm+'_end']=str(out/arm/'checkpoint.pth')
        tune_dir=out/'tune1000';tune_dir.mkdir()
        tune_cfg={**cfg,'output':str(tune_dir),'evaluation_checkpoints':checkpoints}
        save(tune_dir/'evaluation_protocol.json',tune_cfg)
        tune_metrics=evaluate(tune_cfg,repo);save(tune_dir/'metrics.json',tune_metrics)
        picks={};eligibility={}
        for arm in ('scst_ce','scst_ce_object'):
            picks[arm],eligibility[arm]=select_checkpoint(tune_metrics,[arm+'_mid',arm+'_end'])
        selected={'original':paths['checkpoint'],**{arm:checkpoints[name] for arm,name in picks.items()}}
        locked={'selected_names':picks,'eligible_names':eligibility,'checkpoint_paths':selected,
                'checkpoint_sha256':{k:sha(v) for k,v in selected.items()},'tune_metrics_sha256':sha(tune_dir/'metrics.json'),
                'rule':cfg['selection'],'test_used':False,'locked_before_remainder_evaluation':True}
        save(out/'locked_selection.json',locked)
        final_dir=out/'validation_remainder4000';final_dir.mkdir()
        final_cfg={**cfg,'output':str(final_dir),'groups':{'train':groups['train'],'val':groups['validation_remainder']},
                   'evaluation_checkpoints':selected}
        save(final_dir/'evaluation_protocol.json',final_cfg)
        metrics=evaluate(final_cfg,repo);save(final_dir/'metrics.json',metrics)
        treatment,control,baseline=(metrics[k+'_fixed'] for k in ('scst_ce_object','scst_ce','original'))
        gate=(treatment['caption']['CIDEr']>max(control['caption']['CIDEr'],baseline['caption']['CIDEr']) and
              all(treatment['chair'][k]<=min(control['chair'][k],baseline['chair'][k]) for k in ('CHAIRs','CHAIRi')) and
              treatment['chair']['Recall']>=max(control['chair']['Recall'],baseline['chair']['Recall'])-.01)
        summary={'status':'FULL_TRAIN_COMPARISON_COMPLETE','train_images':len(groups['train']),
                 'updates_per_arm':updates,'midpoint_update':half,'tune_images':1000,'evaluation_images':4000,
                 'locked_selection':locked,'tune_metrics':tune_metrics,'remainder_metrics':metrics,
                 'fixed_beats_both_with_hallucination_control':bool(gate),'test_used':False,
                 'protocol_sha256':sha(protocol),'production_promotion':False,
                 'limits':['One seed and reused development validation; no guaranteed improvement.',
                           'Checkpoint selection uses fixed policy; raw metrics secondary.',
                           'Remainder4000 was not used for checkpoint selection in this run; may have been seen historically.',
                           'Training CIDEr IDF now uses full train; different from20k pilot.',
                           'Annotation object support is not physical absence proof or relation grounding.']}
        save(out/'run_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    except Exception as error:
        save(out/'failure.json',{'status':'INCOMPLETE','error':repr(error)})
        raise
    finally:
        with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as bundle:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix in ('.json','.jsonl','.log'):bundle.write(p,p.relative_to(out))
        print('RETURN FILE:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','prompt-cache-path','visual-cache','dataset-json-path','base-path','detections','chair-directory','output'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
