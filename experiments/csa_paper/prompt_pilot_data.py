"""Data-only contracts for paired prompt pilots. No Torch/model/API execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
from full_cache_data import stage_full,collate
from full_xe_data import validation_references
from scst_pilot_data import schedule,event_schedule,DATA_SHA,MODEL_SHA
from prompt_text_cache import validate_links
from prompt_adapter_data import tensor_batch

CHECKPOINT_SHA='58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b'
CACHE_RECEIPT_SHA='3c3416d1d6d21ba693e4b5c9833d89b0479e89a712196453340fb9ad79d21a42'
MANIFEST_SHA='24c1c75fda9be90f6e1409aedc8d2bbecb04cb78e5570443a3f813b402b66486'


def cache_root(root):
    matches=[p for p in Path(root).rglob('cache_receipt.json') if sha(p)==CACHE_RECEIPT_SHA]
    if len(matches)!=1:raise ValueError('Attach one complete exact169 cache receipt; no latest substitute')
    root=matches[0].parent;r=read(matches[0])
    if r['status']!='PROMPT_TEXT_CACHE_COMPLETE' or set(r['payload_sha256'])!={'prompt_embeddings.npy','prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json'}:
        raise ValueError('Incomplete/wrong169 receipt')
    for n,h in r['payload_sha256'].items():
        if Path(n).name!=n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed169 payload:'+n)
    if sha(root/'cache_manifest.json')!=MANIFEST_SHA:raise ValueError('Wrong169 manifest')
    a=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if a.shape!=(318,512) or a.dtype!=np.float32 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=2e-6,rtol=0):
        raise ValueError('Invalid169 embeddings')
    texts,validation=validate_links(root)
    return root,a,texts,validation


def checkpoint(root):
    root=Path(root)
    choices=[root] if root.is_file() else list(root.rglob('csa_ce161_best.bin'))
    matches=[p for p in choices if sha(p)==CHECKPOINT_SHA]
    if len(matches)!=1:raise ValueError('Need exact161 csa_ce161_best.bin from shared input Dataset')
    return matches[0]


def records(root):
    with (Path(root)/'prompt_links.jsonl').open(encoding='utf-8') as stream:
        return {r['coco_id']:r for r in map(json.loads,stream)}


def checked_join(data,links,cohort):
    expected={int(i):g for g in ('train','val') for i in cohort[g]}
    if len(expected)!=15000 or set(expected)!=set(links):raise ValueError('Unexpected prompt ID coverage')
    index={int(i):j for j,i in enumerate(data['ids'])}
    for i,g in expected.items():
        if i not in index:raise ValueError('Prompt image absent ROI')
        j=index[i];r=links[i]
        if int(data['group'][j])!=(g=='val') or r['group']!=g or r['roi_count']!=int(data['offsets'][j+1]-data['offsets'][j]):
            raise ValueError('Prompt group/ROI index space differs from staged visual cache')
    val={int(i) for i in data['ids'][data['group']==1]}
    if set(cohort['val'])!=val or cohort['test']:raise ValueError('VALID split differs/TEST leakage')


def batch(data,events,links,embeddings,texts):
    x,v,seq=collate(data,events)
    ids=data['ids'][data['caption_image'][np.asarray(events,dtype=np.int64)]]
    rows=[links[int(i)] for i in ids]
    return (x,v,seq),tensor_batch(rows,embeddings,texts),ids


def prepare(cfg):
    if cfg['arm'] not in ('CONTROL','PROMPT') or cfg['updates']!=500:raise ValueError('Only approved paired500 study')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    ck=checkpoint(cfg['input_root']);root,emb,texts,validation=cache_root(cfg['input_root'])
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO JSON')
    dataset=read(cfg['dataset_json']);refs={}
    for r in dataset['images']:
        if r['split'] in ('train','restval'):
            i=int(Path(r['filename']).stem.split('_')[-1]);refs[str(i)]=[s['raw'] for s in r['sentences'][:5]]
    if len(refs)!=113287 or any(len(v)!=5 for v in refs.values()):raise ValueError('Wrong TRAIN reward reference coverage')
    del dataset
    data,vocab,staged=stage_full(cfg['full_root'],cfg['stage'])
    links=records(root);cohort=read(root/'cohort.json');checked_join(data,links,cohort)
    pool,ids=schedule(list(map(int,refs)),updates=500)
    if pool!=cohort['train']:raise ValueError('TRAIN pool differs from mapping167')
    events=event_schedule(data,ids)
    if events.shape!=(500,8) or len(np.unique(ids))!=4000 or not np.array_equal(data['ids'][data['caption_image'][events]],ids):
        raise ValueError('Schedule join/exposure differs')
    save(out/'training_references.json',refs);save(out/'validation_references.json',validation_references(cfg['full_root']))
    save(out/'vocabulary.json',vocab);np.save(out/'events.npy',events);np.save(out/'coco_ids.npy',ids)
    names=('training_references.json','validation_references.json','vocabulary.json','events.npy','coco_ids.npy')
    save(out/'pilot_stage.json',dict(status='PROMPT_PILOT_STAGE_COMPLETE',checkpoint_sha256=CHECKPOINT_SHA,checkpoint_path=str(ck),
        cache_receipt_sha256=CACHE_RECEIPT_SHA,cache_root=str(root),model_sha256=MODEL_SHA,dataset_json_sha256=DATA_SHA,
        pool_ids=pool,unique_exposed=4000,updates=500,image_events=4000,global_batch=8,test_used=False,
        files_sha256={n:sha(out/n) for n in names},source_stage=staged['files_sha256'],validation=validation))
    print('Prompt pilot staged: exact161/cache169;500updates/global8/4000uniqueTRAIN;VALID5000/noTEST.',flush=True)
