"""Full175 prompt loading and image-sharing joins; NumPy/metadata only."""
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
from full_prompt_cache_data import checked_mapping,validate_vectors
from full_cache_data import collate
from sharing_data import shared_collate
from prompt_adapter_data import tensor_batch


def checked_cache(mount,verify=True):
    inventory=read(Path(__file__).with_name('FULL_PROMPT_FRESH_V175_RESULT.json'))['transport_inventory']
    files=inventory['cache_files'];matches=[p for p in Path(mount).rglob('cache_receipt.json') if sha(p)==files['cache_receipt.json']]
    if len(matches)!=1:raise ValueError('Need one exact175 cache receipt')
    root=matches[0].parent
    if verify:
        for n,h in files.items():
            if Path(n).name!=n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed175 payload:'+n)
    texts=read(root/'prompt_texts.json');emb=np.load(root/'prompt_embeddings.npy',allow_pickle=False);validate_vectors(texts,emb)
    if len(texts)!=410:raise ValueError('Expected410textvectors')
    manifest=read(root/'cache_manifest.json')
    if manifest['cache_mode']!='fresh_all' or manifest['test_used'] or manifest['training']:raise ValueError('Wrong full cache mode')
    if verify:
        # This reads the full mapping with the original receipt, independent of169.
        shadow=read(root/'mapping_receipt.json')
        for n,h in shadow['payload_sha256'].items():
            if files.get(n)!=h:raise ValueError('Mapping/cache receipt differs')
    with (root/'prompt_links.jsonl').open(encoding='utf-8') as stream:
        links={}
        for line in stream:
            r=json.loads(line)
            if r['coco_id'] in links:raise ValueError('Duplicate prompt ID')
            links[r['coco_id']]=r
    return root,links,emb,texts,inventory


def checked_join(data,links,cohort):
    expected={int(i):g for g in ('train','val') for i in cohort[g]}
    if len(expected)!=118287 or len(cohort['train'])!=113287 or len(cohort['val'])!=5000 or cohort['test'] or set(expected)!=set(links):raise ValueError('Wrong full prompt coverage')
    index={int(i):j for j,i in enumerate(data['ids'])}
    if set(index)!=set(expected):raise ValueError('Visual/prompt image IDs differ')
    for i,g in expected.items():
        j=index[i];r=links[i]
        if int(data['group'][j])!=(g=='val') or r['group']!=g or r['roi_count']!=int(data['offsets'][j+1]-data['offsets'][j]):raise ValueError('Prompt group/ROI space differs')
    if {int(i) for i in data['ids'][data['group']==1]}!=set(cohort['val']):raise ValueError('VALID differs')


def batch(data,indices,links,emb,texts,shared=False):
    indices=np.asarray(indices,np.int64);images=data['caption_image'][indices]
    if shared:
        x,v,seq,inverse=shared_collate(data,indices)
        _,first=np.unique(images,return_index=True);images=images[first]
    else:x,v,seq=collate(data,indices);inverse=None
    prompts=tensor_batch([links[int(data['ids'][i])] for i in images],emb,texts)
    return x,v,seq,prompts,inverse


def check_stage(stage,output,prompt_root):
    receipt=read(Path(stage)/'stage_receipt.json')
    data={k:np.load(Path(stage)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in receipt['shapes']}
    root,links,emb,texts,inventory=checked_cache(prompt_root)
    checked_join(data,links,read(root/'cohort.json'))
    from full_prompt_cache_data import validate_records
    validation=validate_records(iter(links.values()),read(root/'cohort.json'),texts,read(root/'summary.json'),{})
    save(Path(output)/'prompt_stage.json',dict(status='FULL_PROMPT175_JOIN_COMPLETE',cache_files=inventory['cache_files'],validation=validation,model_executed=False,test_used=False))
    print('Full175 joined to exact ROI134:118287images/410texts; no checkpoint loaded.',flush=True)
