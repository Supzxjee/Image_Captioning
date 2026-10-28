"""Shared I/J policy, predicted-box geometry and targets; no neural imports."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from experiments.relation_frame_pretrain.frame_data import areas, geometry, sha, save, FILES, META_SHA, CHECKPOINT_SHA

DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
DETECTION_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
LEGACY_SHA='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
DIMENSIONS_SHA='534c0501884863464bb10c3bb2f504bd69ba9a290c4345ebf3ee179ad3c60206'
CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
BASE_SHA='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
POLICY=dict(version='shared-pair-readout-ij-v1',seed=42,updates=500,caption_batch=8,relation_pairs=4,
    maximum_objects=12,hidden=512,attention=128,temperature=1.,lr_new=1e-4,lr_decoder=5e-6,
    weight_decay=.01,betas=[.9,.999],eps=1e-8,gradient_clip=1.,kl_weight=.1,relation_weight=.2,
    object_reward_weight=.5,max_len=30,prepare_seconds=1800,gpu_seconds=3600,evaluate_seconds=900,
    endpoint_loss=False,decoder_dropout=False,checkpoint_selection='endpoint500 only')


def vocabulary(path):
    if sha(path)!=META_SHA:raise ValueError('Changed A metadata')
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def bank_row(objects,width,height):
    if width<=0 or height<=0:raise ValueError('Invalid dimensions')
    selected=sorted(enumerate(objects),key=lambda p:(-float(p[1]['conf']),p[0]))[:12]
    rows=[]
    for index,o in selected:
        b=np.asarray(o['bbox'],dtype='float64');conf=float(o['conf'])
        if b.shape!=(4,) or not np.isfinite(b).all() or not 0<=conf<=1:raise ValueError('Malformed prediction')
        b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[width,height])
        if np.any(b[2:]<=b[:2]):raise ValueError('Degenerate clipped prediction')
        rows.append(dict(original_index=index,label=o['label'],bbox_xyxy=b.tolist(),confidence=conf))
    return rows


def region_arrays(row):
    """Padded12objects+null. Union rectangle includes the area BETWEEN endpoints."""
    roi=np.zeros((13,196),dtype='float32');union=np.zeros((169,196),dtype='float32')
    boxes=np.zeros((13,4),dtype='float32');confidence=np.zeros((13,1),dtype='float32')
    valid=np.zeros(13,dtype=bool);valid[12]=True;g=np.zeros((169,8),dtype='float32')
    if len(row['objects'])>12:raise ValueError('Too many predicted objects')
    for i,o in enumerate(row['objects']):
        b=np.asarray(o['bbox_xyxy'],dtype='float64')/np.array([row['width'],row['height']]*2)
        roi[i]=areas(b);boxes[i]=b;confidence[i]=o['confidence'];valid[i]=True
    for i in range(len(row['objects'])):
        for j in range(len(row['objects'])):
            if i==j:continue
            a,b=boxes[i].astype('float64'),boxes[j].astype('float64')
            union[i*13+j]=areas(np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])])
            g[i*13+j]=geometry(a,b)
    interaction=valid[:,None]&valid[None,:]&~np.eye(13,dtype=bool)
    interaction[12,:]=False;interaction[:,12]=False
    return dict(roi=roi,union=union,boxes=boxes,confidence=confidence,valid=valid,
        geometry=g,interaction=interaction.reshape(169))


def assertion(pair,vocab):
    s=pair['positive']['subject'];o=pair['positive']['object'];r=pair['relation'];q=pair['statement']
    with_is=f'the {s} is {r} the {o}';without=f'the {s} {r} the {o}'
    if q not in (with_is,without):raise ValueError('Unsupported frozen assertion')
    words=q.split();start=1+len(s.split())+int(q==with_is)
    if any(w not in vocab or vocab[w]<4 for w in words):raise ValueError('Assertion OOV; no substitution')
    ids=[1]+[vocab[w] for w in words]+[2]
    if len(ids)>30:raise ValueError('Assertion exceeds fixed maximum length')
    tail=np.zeros(len(ids)-1,dtype='float32');tail[start:len(words)]=1/(len(words)-start)
    predicate=np.zeros_like(tail);predicate[start:start+len(r.split())]=1/len(r.split())
    return dict(pair_id=pair['pair_id'],relation=r,
        stratum='contact_proximity' if r in ('touching','next to','near','far from','far away from') else 'projective_containment',
        positive=pair['positive']['image_id'],negative=pair['negative']['image_id'],
        statement=q,ids=ids,tail=tail.tolist(),predicate=predicate.tolist())


def schedule(n,size,step,seed):
    if n<size:raise ValueError('Not enough images for fixed global batch')
    batches=math.ceil(n/size);epoch,offset=divmod(step,batches)
    order=np.random.default_rng(seed+epoch).permutation(n)
    if len(order)%size:order=np.r_[order,order[:size-len(order)%size]]
    return order[offset*size:(offset+1)*size]


def attention_numpy(a,b,s,valid):
    """Independent FP64 oracle for user-run attention preflight; never a model."""
    def logsoftmax(x):
        maximum=x.max(-1,keepdims=True)
        return x-maximum-np.log(np.exp(x-maximum).sum(-1,keepdims=True))
    valid=np.asarray(valid,dtype=bool)
    if not valid.any(-1).all():raise ValueError('Every image needs at least null support')
    la=logsoftmax(np.where(valid[:,None,:],np.asarray(a,dtype='float64'),-np.inf))
    lb=logsoftmax(np.where(valid[:,None,:],np.asarray(b,dtype='float64'),-np.inf))
    k=valid.shape[-1];joint=(la[:,:,:,None]+lb[:,:,None,:]).reshape(*la.shape[:2],k*k)+s
    joint=logsoftmax(joint).reshape(*la.shape[:2],k,k)
    return np.exp(la),np.exp(lb),np.exp(joint)


def panel_ids(pairs):
    chosen=[]
    for stratum in ('projective_containment','contact_proximity'):
        pool=[p for p in pairs if p['stratum']==stratum]
        if len(pool)!=32:raise ValueError('Frozen heldout strata differ')
        chosen.extend(p['pair_id'] for p in sorted(pool,key=lambda p:hashlib.sha256(('shared-readout-panel-v1:'+p['pair_id']).encode()).hexdigest())[:16])
    return chosen


class Inputs:
    def __init__(self,prepared):
        import h5py
        prepared=Path(prepared);self.meta=json.loads((prepared/'metadata.json').read_bytes())
        with h5py.File(prepared/'inputs.h5','r') as f:
            if not f.attrs.get('complete') or f['coco_ids'][:].tolist()!=[r['coco_id'] for r in self.meta['rows']]:raise ValueError('Changed stage schema/order')
            self.arrays={k:f[k][:] for k in ('visual','prompt','mask')}
        self.geometry=[region_arrays(r) for r in self.meta['rows']]
        if any(not np.isfinite(a).all() for a in self.arrays.values()):raise ValueError('Nonfinite stage')

    def batch(self,indices):
        ix=list(map(int,indices));b={k:a[ix].astype('float32') if k!='mask' else a[ix].astype(bool) for k,a in self.arrays.items()}
        b.update({k:np.stack([self.geometry[i][k] for i in ix]) for k in self.geometry[0]})
        return b

    def relation_batch(self,split,indices):
        pairs=[self.meta['pairs'][split][int(i)] for i in indices];ix=[]
        for p in pairs:ix.extend([p['positive_index'],p['negative_index']])
        b=self.batch(ix);length=max(len(p['ids'])-1 for p in pairs)
        for key,dtype in (('ids','int64'),('targets','int64'),('tail','float32'),('predicate','float32')):
            a=np.zeros((len(pairs)*2,length),dtype=dtype)
            for i,p in enumerate(pairs):
                v=p['ids'][:-1] if key=='ids' else p['ids'][1:] if key=='targets' else p[key]
                a[2*i:2*i+2,:len(v)]=v
            b[key]=a
        return b
