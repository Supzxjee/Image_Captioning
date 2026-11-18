"""Full metadata, schedules and lazy raw H5 reader; no neural imports."""
from collections import OrderedDict,Counter
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata
import numpy as np
from real_bank_data import read,save,sha
from data import vocabulary
from patch_data import DATA_SHA,COUNTS,check_features

TRAIN_IMAGES=113287;TRAIN_EVENTS=566747;VALID_IMAGES=5000;VALID_EVENTS=25010
STEPS=(TRAIN_EVENTS+31)//32;EPOCHS=10

def encode_tokens(tokens,lookup):
    if not isinstance(tokens,list) or not all(isinstance(t,str) and t for t in tokens):raise ValueError('Invalid caption tokens')
    x=[lookup.get(unicodedata.normalize('NFKC',t).lower(),3) for t in tokens[:52]]
    return [1]+x+[2]+[0]*(52-len(x))

def learning_rate(step):
    if type(step)!=int or not 1<=step<=EPOCHS*STEPS:raise ValueError('Invalid update')
    if step<=STEPS:return 8e-6+(1e-4-8e-6)*(step-1)/(STEPS-1)
    ratio=(step-STEPS)/(9*STEPS)
    return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*ratio))

def order_for_epoch(data,epoch):
    if type(epoch)!=int or not 1<=epoch<=10:raise ValueError('Invalid epoch')
    image=np.asarray(data['caption_image'][:TRAIN_EVENTS])
    if (np.diff(image)<0).any() or len(np.unique(image))!=TRAIN_IMAGES:raise ValueError('TRAIN caption mapping differs')
    if image.min()!=0 or image.max()!=TRAIN_IMAGES-1:raise ValueError('Heldout/foreign TRAIN image index')
    return np.random.default_rng(42+epoch).permutation(TRAIN_EVENTS)

def validate_state(s):
    epoch=s['epoch'];cursor=s['batch_cursor'];phase=s['phase']
    if type(epoch)!=int or type(cursor)!=int or not 1<=epoch<=11 or not 0<=cursor<=STEPS or phase not in ('train','validate'):raise ValueError('Invalid cursor')
    if epoch==11 and (cursor or phase!='train'):raise ValueError('Invalid completed cursor')
    if phase=='validate' and cursor!=STEPS:raise ValueError('Premature validation')
    if s['updates']!=(epoch-1)*STEPS+cursor or s['caption_events']!=(epoch-1)*TRAIN_EVENTS+min(cursor*32,TRAIN_EVENTS):raise ValueError('Exposure mismatch')
    if [r['epoch'] for r in s['history']]!=list(range(1,epoch)):raise ValueError('Incomplete history')

def prepare(cfg):
    import h5py
    started=time.monotonic();root=Path(cfg['stage']);root.mkdir(parents=True,exist_ok=False);out=Path(cfg['output'])
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO JSON')
    ds=read(cfg['dataset_json']);all_rows=ds['images']
    if dict(Counter(r['split'] for r in all_rows))!=COUNTS:raise ValueError('Wrong Karpathy splits')
    mapping={int(r['cocoid']):r for r in all_rows}
    if len(mapping)!=len(all_rows):raise ValueError('Duplicate COCO ID')
    train=[r for r in all_rows if r['split'] in ('train','restval')];val=[r for r in all_rows if r['split']=='val'];rows=train+val
    wanted={int(r['cocoid']):j for j,r in enumerate(rows)}
    index=np.full((len(rows),2),-1,np.int32);seen=set();profiles=[];sources=[]
    for number,path in enumerate(sorted(Path(cfg['visual_cache']).rglob('*.h5'))):
        with h5py.File(path,'r') as f:
            text=lambda x:x.decode() if isinstance(x,bytes) else str(x)
            if not f.attrs.get('complete',False) or text(f.attrs.get('id_key',''))!='coco_id':raise ValueError('Incomplete/ambiguous H5')
            meta=json.loads(text(f.attrs['metadata_json']));gold=read(Path(__file__).with_name('SMOKE_SOURCE.json'))['feature_profile']
            profile={k:meta.get(k) for k in gold}
            if profile!=gold:raise ValueError('Cache extraction profile differs from smoke192')
            ids=f['imgids'][:];names=f['filename'][:];features=f['features']
            if features.shape!=(len(ids),197,768) or features.dtype!=np.dtype('float16') or len(names)!=len(ids):raise ValueError('Raw H5 shape differs')
            for i,(coco,name) in enumerate(zip(ids,names)):
                coco=int(coco)
                if coco in seen or coco not in mapping or text(name)!=mapping[coco]['filename']:raise ValueError('H5 ID/filename mismatch')
                seen.add(coco)
                if coco in wanted:index[wanted[coco]]=(number,i)
            sources.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,shape=list(features.shape),dtype=str(features.dtype),chunks=features.chunks,compression=features.compression))
        print('Full H5 metadata:',number+1,'mapped',int((index[:,0]>=0).sum()),'/',len(rows),flush=True)
    if seen!=set(mapping) or (index<0).any() or len(train)!=TRAIN_IMAGES or len(val)!=VALID_IMAGES:raise ValueError('Missing/foreign raw cache coverage')
    words=vocabulary(ds);lookup={w:i for i,w in enumerate(words)}
    sequences=[];images=[];refs=[]
    for j,r in enumerate(rows):
        for sentence in r['sentences']:sequences.append(encode_tokens(sentence['tokens'],lookup));images.append(j)
        if j>=TRAIN_IMAGES:refs.append(dict(coco_id=int(r['cocoid']),references=[s['raw'] for s in r['sentences']]))
        if (j+1)%20000==0:print('Caption metadata:',j+1,'/',len(rows),flush=True)
    arrays=dict(sequences=np.asarray(sequences,np.int32),caption_image=np.asarray(images,np.int32),ids=np.asarray([int(r['cocoid']) for r in rows],np.int64),locations=index)
    if arrays['sequences'].shape!=(TRAIN_EVENTS+VALID_EVENTS,54) or int((arrays['caption_image']<TRAIN_IMAGES).sum())!=TRAIN_EVENTS:raise ValueError('Wrong full caption count')
    for n,v in arrays.items():np.save(root/(n+'.npy'),v)
    save(root/'vocabulary.json',words);save(root/'validation_references.json',refs)
    if sha(root/'vocabulary.json')!=read(Path(__file__).with_name('SMOKE_SOURCE.json'))['vocabulary_sha256']:raise ValueError('TRAIN vocabulary differs from smoke')
    manifest=dict(status='FULL_PATCH_METADATA_COMPLETE',schema='raw_h5_patch_stream_v1',dataset_sha256=DATA_SHA,sources=sources,feature_profile=gold,
        train_images=TRAIN_IMAGES,train_caption_events=TRAIN_EVENTS,valid_images=VALID_IMAGES,valid_caption_events=VALID_EVENTS,
        files_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()},test_features_used=False,feature_arrays_staged=False,seconds=time.monotonic()-started)
    save(root/'manifest.json',manifest);save(out/'stage_summary.json',manifest)

def load_stage(root,verify=True):
    root=Path(root);m=read(root/'manifest.json')
    if m['status']!='FULL_PATCH_METADATA_COMPLETE' or m['train_caption_events']!=TRAIN_EVENTS or m['test_features_used']:raise ValueError('Wrong stage scope')
    if verify:
        for n,h in m['files_sha256'].items():
            if Path(n).name!=n or sha(root/n)!=h:raise ValueError('Changed stage file '+n)
    data={k:np.load(root/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('sequences','caption_image','ids','locations')}
    return data,read(root/'vocabulary.json'),m

class FeatureReader:
    def __init__(self,root,capacity=256):
        self.data,_,self.meta=load_stage(root,verify=False);self.handles={};self.cache=OrderedDict();self.capacity=capacity
        self.golden=read(Path(__file__).with_name('SMOKE_SOURCE.json'))['raw_selected_row_sha256']
        self.local=None;self.local_rows=None
        if self.meta.get('feature_storage')=='local_fp16_memmap_v2':
            spec=self.meta['local_features']
            if spec['file']!='local_patches.npy' or spec['rows']!='local_rows.npy':raise ValueError('Unexpected local payload names')
            self.local=np.load(Path(root)/spec['file'],mmap_mode='r',allow_pickle=False)
            self.local_rows=np.load(Path(root)/spec['rows'],mmap_mode='r',allow_pickle=False)
            if self.local.shape!=(len(self.data['ids']),196,768) or self.local.dtype!=np.float16 or self.local_rows.shape!=(len(self.data['ids']),):raise ValueError('Invalid local patch mapping/shape')
    def get(self,image):
        image=int(image)
        if image in self.cache:self.cache.move_to_end(image);return self.cache[image]
        if self.local is not None:
            row=int(self.local_rows[image])
            if not 0<=row<len(self.local):raise ValueError('Invalid local row')
            value=self.local[row]
            if not np.isfinite(value).all():raise ValueError('Nonfinite local feature')
            self.cache[image]=value
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            return value
        import h5py
        file,row=map(int,self.data['locations'][image]);source=self.meta['sources'][file]
        if file not in self.handles:
            path=Path(source['path'])
            if path.stat().st_size!=source['bytes']:raise ValueError('Raw file size changed')
            self.handles[file]=h5py.File(path,'r',rdcc_nbytes=2*1024**2)
        raw=self.handles[file]['features'][row];coco=str(int(self.data['ids'][image]))
        if coco in self.golden and hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()!=self.golden[coco]:raise ValueError('Smoke raw row replay differs '+coco)
        value=check_features(raw[None])[0];self.cache[image]=value
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return value
    def close(self):
        self.cache.clear()
        for f in self.handles.values():f.close()
        self.handles.clear()
        if self.local is not None:self.local._mmap.close();self.local=None
        if self.local_rows is not None:self.local_rows._mmap.close();self.local_rows=None

class CaptionBatches:
    def __init__(self,root,events,rank=None,cursor=0,batch_size=32):
        self.root=str(root);self.events=np.asarray(events,np.int64);self.rank=rank;self.cursor=cursor;self.batch_size=batch_size;self.reader=None
    def __len__(self):return (len(self.events)+self.batch_size-1)//self.batch_size-self.cursor
    def __getstate__(self):return dict(self.__dict__,reader=None)
    def __getitem__(self,item):
        if self.reader is None:self.reader=FeatureReader(self.root)
        start=(item+self.cursor)*self.batch_size;ev=self.events[start:start+self.batch_size]
        if self.rank is not None:ev=ev[self.rank*16:(self.rank+1)*16]
        if not len(ev):raise ValueError('Empty distributed caption batch')
        d=self.reader.data;image=d['caption_image'][ev];x=np.stack([self.reader.get(i) for i in image]).astype(np.float32)
        seq=np.asarray(d['sequences'][ev],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
        return x,seq,ev.copy()

class ImageBatches:
    def __init__(self,root,images,batch_size=8):self.root=str(root);self.images=np.asarray(images,np.int64);self.batch_size=batch_size;self.reader=None
    def __len__(self):return (len(self.images)+self.batch_size-1)//self.batch_size
    def __getstate__(self):return dict(self.__dict__,reader=None)
    def __getitem__(self,item):
        if self.reader is None:self.reader=FeatureReader(self.root)
        indices=self.images[item*self.batch_size:(item+1)*self.batch_size]
        return np.stack([self.reader.get(i) for i in indices]).astype(np.float32),indices.copy()

def loader(dataset,workers=2):
    from torch.utils.data import DataLoader
    return DataLoader(dataset,batch_size=None,num_workers=workers,pin_memory=True,multiprocessing_context='spawn' if workers else None,prefetch_factor=2 if workers else None)
