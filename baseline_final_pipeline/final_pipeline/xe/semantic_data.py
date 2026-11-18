"""Pure NumPy fixed-bank/prompt175 to patch-grid preparation; no models."""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha
from full_data import load_stage,TRAIN_IMAGES,VALID_IMAGES

HERE=Path(__file__).resolve().parent
ARRAYS=dict(object_text_id=((8,),np.int16),object_box=((8,4),np.float32),object_weights=((8,196),np.float32),object_valid=((8,),bool),
    relation_text_id=((3,),np.int16),subject_weights=((3,196),np.float32),target_weights=((3,196),np.float32),relation_geometry=((3,8),np.float32),relation_valid=((3,),bool))

def box_weights(value):
    b=np.asarray(value,np.float64)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b<0).any() or (b>1).any() or (b[2:]<=b[:2]).any():raise ValueError('Invalid normalized xyxy box')
    x0,y0,x1,y1=b*14;axis=np.arange(14,dtype=np.float64)
    wx=np.maximum(0,np.minimum(axis+1,x1)-np.maximum(axis,x0));wy=np.maximum(0,np.minimum(axis+1,y1)-np.maximum(axis,y0));weights=(wy[:,None]*wx[None,:]).reshape(196)
    if weights.sum()<=0:raise ValueError('Box has no patch support')
    return (weights/weights.sum()).astype(np.float32)

def record_arrays(row,texts):
    expected={'coco_id','filename','group','roi_count','source_mode','objects','relations','counts'}
    if set(row)!=expected or len(row['objects'])>8 or len(row['relations'])>3:raise ValueError('Unexpected prompt fields/caps')
    out={k:np.zeros(shape,dtype=dtype) for k,(shape,dtype) in ARRAYS.items()};lookup={t:i for i,t in enumerate(texts)};objects={}
    for slot,obj in enumerate(row['objects']):
        instance=obj['instance_id']
        if not instance or instance in objects or obj['text'] not in lookup:raise ValueError('Invalid/duplicate object instance/text')
        if not 0<=obj['roi']<row['roi_count']:raise ValueError('Invalid original ROI provenance')
        # Pool the SAME matched YOLO rectangle used by legacy ROI features, not the ROI index.
        w=box_weights(obj['roi_bbox']);box_weights(obj['bbox'])
        objects[instance]=w;out['object_weights'][slot]=w;out['object_box'][slot]=obj['bbox'];out['object_text_id'][slot]=lookup[obj['text']];out['object_valid'][slot]=True
    for slot,rel in enumerate(row['relations']):
        s,o=rel['subject_instance'],rel['object_instance']
        if s==o or s not in objects or o not in objects or rel['text'] not in lookup:raise ValueError('Missing/collapsed relation endpoints/text')
        g=np.asarray(rel['geometry'],np.float32)
        if g.shape!=(8,) or not np.isfinite(g).all():raise ValueError('Invalid relation geometry')
        out['subject_weights'][slot]=objects[s];out['target_weights'][slot]=objects[o];out['relation_text_id'][slot]=lookup[rel['text']];out['relation_geometry'][slot]=g;out['relation_valid'][slot]=True
    return out

def checked_root(mount,receipt_name,expected):
    found=[p.parent for p in Path(mount).rglob(receipt_name) if sha(p)==expected[receipt_name]]
    if len(found)!=1:raise ValueError('Need one exact semantic cache root: '+receipt_name)
    root=found[0]
    for n,h in expected.items():
        if Path(n).name!=n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed semantic payload '+n)
    return root

def fixed_banks(mount):
    package=read(HERE/'SAVE_BANK_V130_PACKAGE.json');root=checked_root(mount,'cache_receipt.json',package['files']);bank=read(HERE/'bank_words_corrected.json');manifest=read(root/'cache_manifest.json')
    if sha(HERE/'bank_words_corrected.json')!='cec76a0229f30361235a4b64e9712ac6bb2e79fc62a9241079a7af3fa95d13ac' or bank['source']!='TRAIN only, train+restval' or manifest['source_bank_sha256']!=sha(HERE/'bank_words_corrected.json'):raise ValueError('Bank supervision/provenance changed')
    arrays={}
    with np.load(root/'semantic_banks.npz',allow_pickle=False) as f:
        for kind in ('noun','verb'):
            words=[x['word'] for x in bank[kind]];v=f[kind+'_vectors']
            if f[kind+'_words'].tolist()!=words or v.shape!=(len(words),512) or v.dtype!=np.float32 or not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Bad bank vector order/space')
            arrays[kind]=v.copy()
    return arrays,package,manifest

def checked_inputs(cfg):
    inv=read(HERE/'FULL_PROMPT_FRESH_V175_RESULT.json')['transport_inventory'];root=checked_root(cfg['prompt_root'],'cache_receipt.json',inv['cache_files']);cohort=read(root/'cohort.json');texts=read(root/'prompt_texts.json');vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False);prompt_manifest=read(root/'cache_manifest.json')
    if len(texts)!=410 or texts!=sorted(set(texts)) or vectors.shape!=(410,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Wrong frozen175 text vectors')
    banks,bank_package,bank_manifest=fixed_banks(cfg['bank_root'])
    if (prompt_manifest['model'],prompt_manifest['revision'])!=(bank_manifest['model'],bank_manifest['revision']):raise ValueError('Bank/prompt CLIP text spaces differ')
    if cohort['test'] or len(cohort['train'])!=TRAIN_IMAGES or len(cohort['val'])!=VALID_IMAGES or len(set(cohort['train']+cohort['val']))!=TRAIN_IMAGES+VALID_IMAGES:raise ValueError('Prompt source scope differs')
    return inv,root,cohort,texts,vectors,prompt_manifest,banks,bank_package


def stage_semantics(cfg,deadline):
    started=time.monotonic();stage=Path(cfg['stage']);out=Path(cfg['output']);data,_,meta=load_stage(stage,verify=False);count=TRAIN_IMAGES+VALID_IMAGES
    if len(data['ids'])!=count or meta.get('semantic'):raise ValueError('Expected new full patch stage')
    inv,root,cohort,texts,vectors,prompt_manifest,banks,bank_package=checked_inputs(cfg)
    if set(cohort['train'])!=set(map(int,data['ids'][:TRAIN_IMAGES])) or set(cohort['val'])!=set(map(int,data['ids'][TRAIN_IMAGES:])):raise ValueError('Prompt/full patch split differs')
    arrays={k:np.lib.format.open_memmap(stage/('sem_'+k+'.npy'),mode='w+',dtype=dtype,shape=(count,)+shape) for k,(shape,dtype) in ARRAYS.items()}
    for value in arrays.values():value[:]=0
    index={int(i):j for j,i in enumerate(data['ids'])};seen=set();totals={g:Counter() for g in ('train','val')}
    try:
        with (root/'prompt_links.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic()>deadline:raise TimeoutError('Semantic stage cap; not complete training input')
                row=json.loads(line);i=row['coco_id'];j=index.get(i)
                if type(i)!=int or i in seen or j is None or row['group']!=('train' if j<TRAIN_IMAGES else 'val'):raise ValueError('Foreign/repeated/leaked prompt image')
                encoded=record_arrays(row,texts)
                for key,a in encoded.items():arrays[key][j]=a
                seen.add(i);totals[row['group']].update(images=1,objects=len(row['objects']),relations=len(row['relations']),with_object=bool(row['objects']),with_relation=bool(row['relations']))
                if len(seen)%10000==0:print('Semantic patch inputs',len(seen),'/',count,flush=True)
        if seen!=set(index):raise ValueError('Missing semantic coverage')
        for value in arrays.values():value.flush()
    finally:
        for value in arrays.values():value._mmap.close()
        for value in data.values():value._mmap.close()
    for kind,v in banks.items():np.save(stage/('sem_'+kind+'_bank.npy'),v)
    np.save(stage/'sem_text_vectors.npy',vectors)
    names=['sem_'+n+'.npy' for n in ARRAYS]+['sem_noun_bank.npy','sem_verb_bank.npy','sem_text_vectors.npy']
    seals={n:sha(stage/n) for n in names}
    if time.monotonic()>deadline:raise TimeoutError('Semantic sealing cap')
    meta['files_sha256'].update(seals);meta['semantic']=dict(method='bank_prompt_patch_v1',prompt_cache=inv['cache_files'],bank_cache=bank_package['files'],text_model=prompt_manifest['model'],text_revision=prompt_manifest['revision'],geometry='14x14 fractional area on matched ROI box; Resize224x224, no center crop',counts={g:dict(c) for g,c in totals.items()},seconds=time.monotonic()-started,TRAIN_VALID_only=True,test_used=False,files_sha256=seals)
    save(stage/'manifest.json',meta);save(out/'stage_summary.json',meta);save(out/'semantic_stage.json',meta['semantic'])
    print('Semantic stage COMPLETE; no CLIP/UPT/model execution.',flush=True)

class SemanticContext:
    def __init__(self,stage):
        self.arrays={k:np.load(Path(stage)/('sem_'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ARRAYS}
        self.text=np.load(Path(stage)/'sem_text_vectors.npy',mmap_mode='r',allow_pickle=False)
    def images(self,indices,device):
        import torch
        p={k:np.array(v[indices],copy=True) for k,v in self.arrays.items() if k not in ('object_text_id','relation_text_id')}
        for role in ('object','relation'):
            p[role+'_text']=self.text[self.arrays[role+'_text_id'][indices]]*p[role+'_valid'][...,None]
        return {k:torch.from_numpy(np.ascontiguousarray(v)).to(device) for k,v in p.items()}
    def events(self,data,events,device):return self.images(data['caption_image'][events],device)
