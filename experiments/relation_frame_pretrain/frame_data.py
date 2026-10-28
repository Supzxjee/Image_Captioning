"""Fixed V2 metadata, token spans and fractional grid weights; no neural imports."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

CHECKPOINT_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
META_SHA='62f43f771aec9feba31a2fbbb6fe7db2878f259979faac378b74a754822c7b67'
FILES={
 'inputs.h5':'daf32d52859452eb2beb8c35e2aad6ed6418cf52c68e4ba646be633151a7a1e9',
 'relation_frames_manifest.bin':'6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7',
 'images.json':'531ac7ef4d794652a738af572f157f06c44f9b4a57ecb9926ebfa4fb29aa8923',
 'detector_boxes.json':'57bf20b276f937644cf01751e619add484b44e08711811f829633c6be125c677',
 'receipt.json':'9bc63a85ce59084d029739d55baec4d2ca359c3afeb78ff320a22b1f310f5915'}
RELATIONS=('on','under','above','below','on top of','beneath','in','inside','within','outside','contains','touching','next to','near','far from','far away from')
CONTACT={'touching','next to','near','far from','far away from'}
CLASSES=('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def lexical_targets(pair,vocab):
    """Target indices exclude BOS; three equally weighted lexical groups, no articles."""
    s=pair['positive']['subject'];o=pair['positive']['object'];r=pair['relation']
    options=[f'the {s} is {r} the {o}',f'the {s} {r} the {o}']
    q=pair['statement']
    if q not in options or r not in RELATIONS or s==o:raise ValueError('Invalid frozen template')
    words=q.split();ss=list(range(1,1+len(s.split())))
    start=1+len(s.split())+(1 if q==options[0] else 0)
    rr=list(range(start,start+len(r.split())))
    oo=list(range(start+len(r.split())+1,len(words)))
    if [words[i] for i in ss]!=s.split() or [words[i] for i in rr]!=r.split() or [words[i] for i in oo]!=o.split():raise ValueError('Span mismatch')
    if any(w not in vocab or vocab[w]<4 for w in words):raise ValueError('Frozen cohort OOV; stop without replacing words')
    ids=[vocab['<bos>']]+[vocab[w] for w in words]+[vocab['<eos>']]
    weights=np.zeros(len(ids)-1,dtype=np.float32)
    for positions in (ss,rr,oo):weights[positions]=1/(3*len(positions))
    if not np.isclose(weights.sum(),1):raise ValueError('Loss weights do not normalize')
    return ids,weights


def areas(box):
    b=np.asarray(box,dtype=np.float64)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1) or np.any(b[2:]<=b[:2]):raise ValueError('Invalid normalized box')
    lo=np.arange(14)/14;hi=lo+1/14
    x=np.maximum(0,np.minimum(hi,b[2])-np.maximum(lo,b[0]))*14
    y=np.maximum(0,np.minimum(hi,b[3])-np.maximum(lo,b[1]))*14
    a=np.outer(y,x).reshape(196).astype('float32')
    if not a.sum()>0:raise ValueError('Empty ROI')
    return a


def geometry(s,o):
    sw,sh=s[2:]-s[:2];ow,oh=o[2:]-o[:2]
    sc,oc=(s[:2]+s[2:])/2,(o[:2]+o[2:])/2
    inter=np.maximum(np.minimum(s[2:],o[2:])-np.maximum(s[:2],o[:2]),0).prod()
    return np.asarray([(oc[0]-sc[0])/sw,(oc[1]-sc[1])/sh,np.log(ow/sw),np.log(oh/sh),
        inter/(sw*sh+ow*oh-inter),inter/(sw*sh),inter/(ow*oh),np.linalg.norm(oc-sc)],dtype='float32')


def metadata(root,meta_path,verify=True):
    root=Path(root)
    for n,h in FILES.items():
        if not (root/n).is_file() or (verify and sha(root/n)!=h):raise ValueError('Missing/changed v105 input: '+n)
    if sha(meta_path)!=META_SHA:raise ValueError('Changed A vocabulary metadata')
    vocab=json.loads(gzip.decompress(Path(meta_path).read_bytes()))['word2idx']
    if len(vocab)!=10000 or [vocab[k] for k in ('<pad>','<bos>','<eos>','<unk>')]!=[0,1,2,3]:raise ValueError('A vocabulary differs')
    m=json.loads(gzip.decompress((root/'relation_frames_manifest.bin').read_bytes()))
    images=json.loads((root/'images.json').read_bytes());bank=json.loads((root/'detector_boxes.json').read_bytes())
    index={row['coco_id']:i for i,row in enumerate(images)}
    if len(index)!=612 or len(m['train'])!=256 or len(m['heldout'])!=64:raise ValueError('Frozen cohort counts differ')
    seen={};packed={}
    for split in ('train','heldout'):
        seen[split]=set();packed[split]=[]
        for p in m[split]:
            ids,w=lexical_targets(p,vocab);rows=[]
            for role,label in (('positive',1),('negative',0)):
                r=p[role];iid=r['image_id'];image=images[index[iid]]
                if r['label']!=label or r['statement']!=p['statement'] or r['relation']!=p['relation']:raise ValueError('Assertion mismatch')
                if any(r[k]!=p['positive'][k] for k in ('subject','object')):raise ValueError('Cross-image concepts differ')
                seen[split].add(iid);boxes=[];classes=[]
                for name in ('subject','object'):
                    pred=r['predicted'][name]
                    found=[b for b in bank[str(iid)] if b['original_index']==pred['index']]
                    if len(found)!=1 or found[0]['label']!=r[name] or found[0]['bbox_xyxy']!=pred['bbox_xyxy']:raise ValueError('Predicted referent binding changed')
                    b=np.asarray(found[0]['bbox_xyxy'],dtype='float64')/np.array([image['width'],image['height']]*2)
                    boxes.append(b);classes.append(CLASSES.index(found[0]['label']))
                s,o=boxes;u=np.r_[np.minimum(s[:2],o[:2]),np.maximum(s[2:],o[2:])]
                rows.append(dict(image_index=index[iid],areas=np.stack([areas(s),areas(o),areas(u)]),geometry=geometry(s,o),classes=classes))
            packed[split].append(dict(pair_id=p['pair_id'],relation=RELATIONS.index(p['relation']),
                stratum='contact_proximity' if p['relation'] in CONTACT else 'projective_containment',
                ids=ids,weights=w,rows=rows))
    if seen['train']&seen['heldout'] or len(seen['heldout'])!=128:raise ValueError('Image leakage')
    return vocab,images,packed


def schedule(step):
    epoch,offset=divmod(step,16)
    return np.random.default_rng(42+epoch).permutation(256)[offset*16:(offset+1)*16]


class Cache:
    def __init__(self,root,meta_path):
        import h5py
        self.vocab,self.images,self.pairs=metadata(root,meta_path,verify=False)
        with h5py.File(Path(root)/'inputs.h5','r') as f:
            if not bool(f.attrs['complete']) or f['visual'].shape!=(612,197,768) or str(f['visual'].dtype)!='float16':raise ValueError('Visual schema differs')
            if list(f['coco_ids'][:])!=[r['coco_id'] for r in self.images]:raise ValueError('HDF5 image order differs')
            self.visual=f['visual'][:,1:,:]
        if not np.isfinite(self.visual).all():raise ValueError('Nonfinite visual inputs')

    def batch(self,split,indices):
        pairs=[self.pairs[split][int(i)] for i in indices];n=len(pairs);length=max(len(p['ids'])-1 for p in pairs)
        ids=np.zeros((n,length),dtype='int64');targets=ids.copy();weights=np.zeros((n,length),dtype='float32')
        rows=[row for p in pairs for row in p['rows']]
        for i,p in enumerate(pairs):
            l=len(p['ids'])-1;ids[i,:l]=p['ids'][:-1];targets[i,:l]=p['ids'][1:];weights[i,:l]=p['weights']
        return dict(visual=self.visual[[r['image_index'] for r in rows]].astype('float32'),
            areas=np.stack([r['areas'] for r in rows]),geometry=np.stack([r['geometry'] for r in rows]),
            classes=np.asarray([r['classes'] for r in rows],dtype='int64'),ids=ids,targets=targets,weights=weights,
            relation=np.asarray([p['relation'] for p in pairs],dtype='int64'))
