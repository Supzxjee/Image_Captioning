"""Safe real-input contracts. No neural imports or execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def validate_arrays(a, images, vocab):
    n = len(images); total = len(a['features'])
    shapes = dict(features=(total,768), boxes=(total,4), scores=(total,), offsets=(n+1,), ids=(n,))
    types = dict(features=np.float16, boxes=np.float32, scores=np.float32, offsets=np.int64, ids=np.int64)
    if set(a) != set(shapes): raise ValueError('Region members differ')
    for key in shapes:
        if a[key].shape != shapes[key] or a[key].dtype != types[key] or not np.isfinite(a[key]).all():
            raise ValueError('Invalid region array: '+key)
    offsets = a['offsets']
    if offsets[0] != 0 or offsets[-1] != total or not ((np.diff(offsets)>=1)&(np.diff(offsets)<=150)).all():
        raise ValueError('Invalid offsets')
    for i, m in enumerate(images):
        if int(a['ids'][i]) != m['coco_id']: raise ValueError('Image order differs')
        b = a['boxes'][offsets[i]:offsets[i+1]]
        if len(b)!=len(m['region_kinds']) or m['width']<=0 or m['height']<=0: raise ValueError('Region metadata differs')
        if not ((b[:,0]>=0)&(b[:,1]>=0)&(b[:,2]<=m['width'])&(b[:,3]<=m['height'])&(b[:,2]>b[:,0])&(b[:,3]>b[:,1])).all():
            raise ValueError('Invalid boxes')
        if any(k not in ('predicted_roi','full_frame_fallback') for k in m['region_kinds']): raise ValueError('Unknown ROI kind')
        for seq in m['sequences']:
            if len(seq)!=54 or seq[0]!=1 or seq.count(2)!=1 or any(type(x)!=int or x<0 or x>=len(vocab) for x in seq):
                raise ValueError('Invalid target sequence')
            end=seq.index(2)
            if 0 in seq[:end] or any(seq[end+1:]): raise ValueError('Invalid EOS/PAD sequence')

def load_prepared(root):
    root=Path(root); p=read(HERE/'REAL_BANK_INPUT_PACKAGE.json')
    if not (root/'corrected_assets_receipt.json').is_file():
        candidates=list(root.rglob('corrected_assets_receipt.json'))
        if len(candidates)!=1: raise ValueError('Expected one corrected assets root')
        root=candidates[0].parent
    receipt=read(root/'corrected_assets_receipt.json')
    for field, expected in [('original_files_sha256',p['original_files_sha256']),('corrected_overlay_sha256',p['overlay_sha256'])]:
        if receipt.get(field)!=expected: raise ValueError('Input receipt differs: '+field)
    if receipt.get('selected_bank_file')!=p['selected_bank_file'] or receipt.get('original_kernel_version')!=p['original_kernel_version']:
        raise ValueError('Selected bank/version differs')
    for name, expected in {**p['original_files_sha256'],**p['overlay_sha256']}.items():
        if not (root/name).is_file() or sha(root/name)!=expected: raise ValueError('Missing/changed input: '+name)
    with np.load(root/'regions.npz', allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    metadata=read(root/'pilot_metadata.json'); vocab=read(root/'vocabulary.json'); bank=read(root/p['selected_bank_file'])
    images=metadata['images']
    if len(images)!=320 or sum(x['group']=='train' for x in images)!=256 or sum(x['group']=='val' for x in images)!=64:
        raise ValueError('Pilot groups differ')
    validate_arrays(arrays,images,vocab)
    if len(bank['noun'])!=5269 or len(bank['verb'])!=1594 or len(vocab)!=10000: raise ValueError('Bank/vocabulary size differs')
    for group in ('train','val'):
        if [images[i]['coco_id'] for i in p[group+'_indices']]!=p[group+'_ids']: raise ValueError('Smoke selection differs')
    return root,p,arrays,images,vocab,bank

def batch(arrays, images, indices):
    lengths=[int(arrays['offsets'][i+1]-arrays['offsets'][i]) for i in indices]
    x=np.zeros((len(indices),max(lengths),768),np.float32); valid=np.zeros(x.shape[:2],bool)
    seq=np.array([images[i]['sequences'][0] for i in indices],np.int64)
    seq=seq[:,:int((seq!=0).sum(1).max())]
    for j,i in enumerate(indices):
        lo,hi=arrays['offsets'][i:i+2]; x[j,:lengths[j]]=arrays['features'][lo:hi]; valid[j,:lengths[j]]=True
    return x,valid,seq
