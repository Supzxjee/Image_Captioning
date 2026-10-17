"""Manual locked test comparison: original QFormer versus selected full-train object0.5."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile
import torch
from run_scst_comparison import evaluate
from run_hoi_caption_validation import save,sha


SELECTED_LEGACY_SHA='5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'


def test_images(dataset):
    rows=[{'image_id':i,'coco_id':int(Path(r['filename']).stem.rsplit('_',1)[1]),
           'filename':r['filename'],'filepath':r['filepath']}
          for i,r in enumerate(dataset['images']) if r['split']=='test']
    if len(rows)!=5000 or len({r['image_id'] for r in rows})!=5000 or len({r['coco_id'] for r in rows})!=5000:
        raise ValueError('Expected5000 unique Karpathy test images.')
    return rows


def validate_selected(payload):
    expected={'visual_adapter':'qformer','num_visual_queries':32,'qformer_layers':2,
              'optimization':'scst_ce_object','scst_weight':.05,'ce_weight':1.,'object_reward_weight':.5,
              'object_reward_parser':'aliases_v2','object_reward_penalty':'unique_count_v2',
              'source_checkpoint_sha256':'155106910d49715f8065a444baca3321bb5e668a1f684ea3c78f860de8a39012'}
    for k,v in expected.items():
        if payload.get(k)!=v:raise ValueError(f'Wrong selected checkpoint metadata:{k}')
    return expected


def run(a):
    from captioning.scst_runtime import configure_strict_runtime,runtime_manifest
    configure_strict_runtime(True)
    repo=Path(__file__).resolve().parent;out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Fresh OUTPUT required;do not overwrite test results.')
    paths={k:str(Path(getattr(a,k)).resolve()) for k in ('checkpoint','selected_checkpoint','prompt_cache_path',
           'visual_cache','dataset_json_path','base_path','detections','chair_directory')}
    identity=json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    hashes={}
    for k in ('checkpoint','selected_checkpoint','prompt_cache_path','dataset_json_path','detections'):
        if not Path(paths[k]).is_file():raise FileNotFoundError(paths[k])
        print('Checking identity:',k,flush=True);hashes[k]=sha(paths[k])
        expected=SELECTED_LEGACY_SHA if k=='selected_checkpoint' else identity[k]
        if hashes[k]!=expected:raise ValueError(f'Input SHA mismatch:{k}')
    for k in ('visual_cache','base_path','chair_directory'):
        if not Path(paths[k]).is_dir():raise FileNotFoundError(paths[k])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=paths['chair_directory'],text=True).strip()!='4087a26211aa2339b9a76307cb8f0321ef691d0a':
        raise ValueError('CHAIR revision mismatch.')
    metadata=validate_selected(torch.load(paths['selected_checkpoint'],map_location='cpu',weights_only=True))
    dataset=json.loads(Path(paths['dataset_json_path']).read_text(encoding='utf-8'));images=test_images(dataset)
    from captioning.visual_cache import VisualCache
    cache=VisualCache([paths['visual_cache']],id_key='coco_id')
    try:cache.require_ids([r['coco_id'] for r in images],'locked test5000')
    finally:cache.close()
    from captioning.object_reward import _AnnotationReader,CACHE_SHA256
    chairfile=Path(paths['chair_directory'])/'chair.pkl'
    if sha(chairfile)!=CACHE_SHA256:raise ValueError('CHAIR annotation cache mismatch.')
    with chairfile.open('rb') as stream:annotations=_AnnotationReader(stream).load()
    if not all(r['coco_id'] in annotations.imid_to_objects for r in images):raise ValueError('IncompleteCHAIR testcoverage.')
    from captioning.object_consistency import index_detections
    detections=index_detections(json.loads(Path(paths['detections']).read_text(encoding='utf-8')))
    if not all(r['filename'] in detections for r in images):raise ValueError('Incomplete detector testcoverage.')
    del dataset,annotations,detections
    cfg={**paths,'output':str(out),'groups':{'test':images},'evaluation_split':'test',
         'evaluation_checkpoints':{'original':paths['checkpoint'],'scst_object':paths['selected_checkpoint']},
         'strict_reproducibility':True,'input_sha256':hashes,'variant':'scst_object_w05_locked_test5000',
         'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         'clip_revision':'57c216476eefef5ab752ec549e440a49ae4ae5f3','selected_metadata':metadata,
         'selected_source_zip_checkpoint_sha256':'6225193761e2519994e9922376db9a83b7b0e05b0cbe2cef251cdf166ce7804c',
         'selection_source':'Fulltrainend checkpoint selected on validation1000 before remainder4000;legacy conversion exact109tensors+metadata',
         'locked_before_test':True,'beam_size':5,'candidate_count':5,
         'fixed_weights':{'decoder':.3,'clip':.4,'object_consistency':.3},
         'primary_policy':'fixed','test_used':True,'training':False,'test_based_selection':False,
         'test_scope':'Previously evaluated test split in project history;not a never-seen independent test.'}
    out.mkdir(parents=True);save(out/'protocol.json',cfg);save(out/'runtime.json',runtime_manifest())
    try:
        metrics=evaluate(cfg,repo)
        deltas={policy:{'CIDEr':metrics['scst_object_'+policy]['caption']['CIDEr']-metrics['original_'+policy]['caption']['CIDEr'],
                        **{k:metrics['scst_object_'+policy]['chair'][k]-metrics['original_'+policy]['chair'][k]
                           for k in ('CHAIRs','CHAIRi','Recall')}} for policy in ('rank0','fixed')}
        summary={'status':'LOCKED_TEST_COMPLETE','test_images':5000,'metrics':metrics,'deltas':deltas,
                 'protocol_sha256':sha(out/'protocol.json'),'test_used':True,'training':False,
                 'checkpoint_changed_after_test':False,'primary_policy':'fixed','production_promotion':False,
                 'limits':['One training seed;test split has prior project exposure.',
                           'No hyperparameter/checkpoint selection on these test results.',
                           'No claim of new relation-alignment contribution from SCST alone.']}
        save(out/'run_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    except Exception as error:
        save(out/'failure.json',{'status':'INCOMPLETE','error':repr(error)});raise
    finally:
        with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as bundle:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix in ('.json','.jsonl','.log'):bundle.write(p,p.relative_to(out))
        print('RETURN FILE:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','selected-checkpoint','prompt-cache-path','visual-cache','dataset-json-path','base-path','detections','chair-directory','output'):
        p.add_argument('--'+n,required=True)
    run(p.parse_args())
