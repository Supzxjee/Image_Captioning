"""Read-only recovery of prepared data; old unjournaled rows need exact source comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np

LEGACY_COMMIT='80b1844bfd68ba363e15309c475448a39620a56f'
LEGACY_PREPARER='547c6550af47680080a32ae682523f01b33194ab2f6394d6d9a99390188e8f66'
RECOVERER='experiments/concept_mass_grounding/recover_stage.py'
PREPARER='experiments/concept_mass_grounding/prepare_concept.py'
SHAPES={'visual':((197,768),'float16'),'prompt':((20,512),'float32'),'mask':((20,),'uint8')}

def normalized(values):
    out={k:np.asarray(values[k],dtype=dtype) for k,(shape,dtype) in SHAPES.items()}
    for k,(shape,_) in SHAPES.items():
        if out[k].shape!=shape or not np.isfinite(out[k]).all():raise ValueError('Bad cached row '+k)
    if not np.isin(out['mask'],[0,1]).all() or not out['mask'].any():raise ValueError('Invalid cached prompt mask')
    return out

def row_sha(values):
    h=hashlib.sha256()
    for k,v in normalized(values).items():h.update(k.encode());h.update(v.tobytes())
    return h.hexdigest()

def scientific_policy(policy):return {k:v for k,v in policy.items() if k!='prepare_seconds'}

class Previous:
    def __init__(self,root,cfg,meta,reward):
        import h5py
        self.f=None;self.stats=dict(journal_reused=0,legacy_revalidated=0,fresh_or_repaired=0)
        if not root:return
        root=Path(root)
        if root.resolve()==Path(cfg['output']).resolve():raise ValueError('Recovery output must differ from preserved source')
        old=json.loads((root/'protocol.json').read_bytes())
        if scientific_policy(old['policy'])!=scientific_policy(cfg['policy']):raise ValueError('Different scientific preparation policy')
        source=old['source_assets'].get(PREPARER)
        legacy=old['source_commit']==LEGACY_COMMIT and source==LEGACY_PREPARER
        current=source==cfg['source_assets'][PREPARER] and old['source_assets'].get(RECOVERER)==cfg['source_assets'].get(RECOVERER) and RECOVERER in cfg['source_assets']
        if not (legacy or current):raise ValueError('Unrecognized previous preparation implementation')
        if json.loads((root/'metadata.json').read_bytes())!=meta or json.loads((root/'object_targets.json').read_bytes())!=reward:raise ValueError('Previous data identities/targets differ')
        self.f=h5py.File(root/'inputs.h5','r')
        try:
            if self.f['coco_ids'][:].tolist()!=[r['coco_id'] for r in meta['rows']]:raise ValueError('Previous row identity differs')
            for k,(shape,dtype) in SHAPES.items():
                if self.f[k].shape!=(len(meta['rows']),*shape) or str(self.f[k].dtype)!=dtype:raise ValueError('Wrong previous tensor schema '+k)
            self.journal=self.f.get('row_sha256') if current else None
            if self.journal is not None and (self.journal.shape!=(len(meta['rows']),) or self.journal.dtype.kind!='S'):raise ValueError('Bad row journal')
        except BaseException:self.close();raise
    def verified(self,i,tokens,mask):
        if self.f is None or self.journal is None:return None
        expected=bytes(self.journal[i]).decode('ascii')
        if not expected:return None
        values={k:self.f[k][i] for k in SHAPES}
        if row_sha(values)!=expected:raise ValueError('Previous journal row failed integrity: '+str(i))
        if not np.array_equal(values['prompt'],tokens) or not np.array_equal(values['mask'],mask):raise ValueError('Previous prompt differs from original cache')
        self.stats['journal_reused']+=1;return normalized(values)
    def source_checked(self,i,source):
        source=normalized(source)
        if self.f is not None:
            old={k:self.f[k][i] for k in SHAPES}
            if all(np.array_equal(old[k],source[k]) for k in SHAPES):
                self.stats['legacy_revalidated']+=1;return normalized(old)
        self.stats['fresh_or_repaired']+=1;return source
    def close(self):
        if self.f is not None:self.f.close();self.f=None
