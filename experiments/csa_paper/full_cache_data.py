"""Full-cache validation, disk staging and NumPy batching. No neural imports."""
from pathlib import Path
import time
import numpy as np
from real_bank_data import HERE,read,save,sha,validate_arrays
from package_full_roi_dataset import check_shard_metadata

def stage_full(root,stage):
    started=time.monotonic();root=Path(root);stage=Path(stage)
    if not root.is_dir():raise FileNotFoundError('Add Input csa-full-roi-karpathy-v134: '+str(root))
    package=read(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json')
    if not (root/'receipt.json').is_file():
        matches=[p.parent for p in root.rglob('receipt.json') if sha(p)==package['files']['receipt.json']]
        if len(matches)!=1:raise ValueError('Expected unique fullROI134 root')
        root=matches[0]
    for number,(n,h) in enumerate(package['files'].items(),1):
        if not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Changed fullROI payload: '+n)
        if number%25==0:print('FullROI SHA',number,'/247',flush=True)
    m=read(root/'manifest.json');vocab=read(root/'vocabulary.json');receipt=read(root/'receipt.json')
    if m['status']!='FULL_ROI_PREPARATION_COMPLETE' or receipt['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Full cache receipt differs')
    metadata_check=check_shard_metadata(root,m,len(vocab))
    if stage.exists():raise ValueError('Preserve existing stage; use a fresh temporary directory')
    stage.mkdir(parents=True)
    shapes=dict(features=(833473,768),offsets=(118288,),ids=(118287,),sequences=(591757,54),caption_image=(591757,),group=(118287,))
    types=dict(features=np.float16,offsets=np.int64,ids=np.int64,sequences=np.int32,caption_image=np.int32,group=np.uint8)
    arrays={k:np.lib.format.open_memmap(stage/(k+'.npy'),mode='w+',dtype=types[k],shape=v) for k,v in shapes.items()}
    image_start=feature_start=caption_start=0;train_events=0;seen_val=False
    for number,entry in enumerate(m['shards'],1):
        names=entry['files'];meta=read(root/next(n for n in names if n.endswith('.json')))['images']
        with np.load(root/next(n for n in names if n.endswith('.npz')),allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        validate_arrays(raw,meta,vocab)
        count=len(meta);regions=len(raw['features'])
        arrays['features'][feature_start:feature_start+regions]=raw['features']
        arrays['offsets'][image_start:image_start+count+1]=raw['offsets']+feature_start
        arrays['ids'][image_start:image_start+count]=raw['ids']
        is_val=entry['group']=='val'
        if seen_val and not is_val:raise ValueError('Expected TRAIN before val in staged order')
        seen_val|=is_val;arrays['group'][image_start:image_start+count]=int(is_val)
        for j,item in enumerate(meta):
            n=len(item['sequences']);arrays['sequences'][caption_start:caption_start+n]=item['sequences']
            arrays['caption_image'][caption_start:caption_start+n]=image_start+j;caption_start+=n
            if not is_val:train_events+=n
        image_start+=count;feature_start+=regions
        print('Validated/staged fullROI shard',number,'/119',flush=True)
    if (image_start,feature_start,caption_start,train_events)!=(118287,833473,591757,566747):raise ValueError('Staged coverage differs')
    for value in arrays.values():value.flush()
    del arrays
    summary=dict(status='FULL_CACHE_STAGE_COMPLETE',metadata=metadata_check,train_caption_events=train_events,val_caption_events=caption_start-train_events,
        shapes=shapes,files_sha256={k+'.npy':sha(stage/(k+'.npy')) for k in shapes},seconds=time.monotonic()-started,source_package=package)
    save(stage/'stage_receipt.json',summary)
    return {k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in shapes},vocab,summary

def collate(data,event_indices):
    event_indices=np.asarray(event_indices,dtype=np.int64);images=data['caption_image'][event_indices]
    lengths=data['offsets'][images+1]-data['offsets'][images]
    features=np.zeros((len(images),int(lengths.max()),768),np.float32);valid=np.zeros(features.shape[:2],bool)
    seq=np.asarray(data['sequences'][event_indices],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
    for j,i in enumerate(images):
        lo,hi=data['offsets'][int(i):int(i)+2];features[j,:int(hi-lo)]=data['features'][lo:hi];valid[j,:int(hi-lo)]=True
    return features,valid,seq

def stress_events(data,train_events,size=32):
    image_indices=data['caption_image'][:train_events]
    counts=np.diff(data['offsets'])[image_indices]
    lengths=(data['sequences'][:train_events,1:]!=0).sum(1)
    candidates=[int(counts.argmax()),int(lengths.argmax())]+list(range(size))
    return list(dict.fromkeys(candidates))[:min(size,train_events)]

def target_tokens(data,indices):return int((data['sequences'][indices,1:]!=0).sum())
