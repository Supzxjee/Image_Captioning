"""Pinned pilot policy, data validation and samplers; neural imports are lazy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from kv_data import digest, save, PREDICATES, INVERSE

CHECKPOINT_SHA = '5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d'
DATA_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
DETECTOR_SHA = '498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
CHAIR_SHA = '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
LEGACY_SHA = 'd7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
POLICY = dict(version='relation-kv-de-v1',updates=500,caption_batch=8,aux_batch=8,seed=42,
    lr_new=1e-4,lr_decoder=5e-6,weight_decay=.01,kl=.1,bce=.1,mp=.05,object_reward=.5,
    gradient_clip=1.,max_len=30,gpu_seconds=7200,evaluate_seconds=1800,
    benchmark_steps=20,forecast_fraction=.8,maximum_objects=6,predicates=list(PREDICATES),
    score_scale=10.,endpoint='500 only',amp=False,teacher='own raw C0',
    quota_accounting='UNVERIFIED; caps are wall limits, not billing estimates')
PAIRS = [(i,j) for i in range(6) for j in range(6) if i != j]
PARTNER = [PAIRS.index((j,i))*3+PREDICATES.index(INVERSE[p]) for i,j in PAIRS for p in PREDICATES]


def checked_cache(root, full=True):
    root = Path(root)
    inventory = json.loads((HERE/'KV_SAVE_V115_PACKAGE.json').read_bytes())['inventory']
    for name, expected in inventory['files'].items():
        path = root/name
        if Path(name).name != name or not path.is_file() or (full and digest(path) != expected):
            raise ValueError('Missing/changed prepared payload: '+name)
    receipt = json.loads((root/'cpu_preparation.json').read_bytes())
    if receipt['status'] != 'READY_FOR_MODEL_PREFLIGHT' or receipt['rows'] != 5472:
        raise ValueError('Wrong preparation receipt')
    return receipt


def cycles(population, length, seed):
    if population <= 0:
        raise ValueError('Empty sampler population')
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < length:
        result.extend(rng.permutation(population).tolist())
    return result[:length]


def schedules():
    return dict(bce=np.asarray(cycles(401,4000,43)).reshape(500,8),
                mp=cycles(58,500,44))


def tokenize_fields(aux, vocab):
    fields = sorted({r[k] for rows in aux['partitions'].values() for r in rows
                     for k in ('subject','predicate','object')})
    if len(vocab) != 10000 or [vocab[k] for k in ('<pad>','<bos>','<eos>','<unk>')] != [0,1,2,3]:
        raise ValueError('Native C0 vocabulary differs')
    rows = []
    unknowns = set()
    for field in fields:
        words = field.lower().split()
        ids = [vocab.get(w,3) for w in words]
        oov = [w for w in words if w not in vocab]
        if not words or not any(i >= 4 for i in ids) or (field in PREDICATES and oov):
            raise ValueError('Invalid/unknown predicate or all-unknown field: '+field)
        unknowns.update(oov)
        rows.append(dict(text=field,words=words,ids=ids,unk_mask=[i==3 for i in ids]))
    if unknowns != {'drier'} or len(fields) != 67:
        raise ValueError('OOV set/field source differs from pinned audit')
    occurrences = {}
    for split, facts in aux['partitions'].items():
        occurrences[split] = dict(facts=len(facts),affected_original_facts=sum(
            any(w not in vocab for k in ('subject','predicate','object') for w in r[k].lower().split()) for r in facts),
            roles={k:dict(fields=len(facts),affected_fields=sum(any(w not in vocab for w in r[k].lower().split()) for r in facts),
                tokens=sum(len(r[k].split()) for r in facts),unknown_tokens=sum(w not in vocab for r in facts for w in r[k].lower().split()))
                for k in ('subject','predicate','object')})
    return dict(fields=rows,occurrences=occurrences,native_unk_id=3,tokenization='lower().split()',
                unknown_words=sorted(unknowns),unique_unknown_fields=sum(any(r['unk_mask']) for r in rows))


def prepare(cfg):
    from captioning.object_reward import make_targets
    out = Path(cfg['output'])
    if out.exists():
        raise FileExistsError('Preserve old output; use a fresh runtime/path')
    if cfg['policy'] != POLICY or cfg['arm'] not in ('D','E'):
        raise ValueError('Changed pilot policy')
    out.mkdir(parents=True)
    save(out/'protocol.json',cfg)
    for key, expected in [('checkpoint',CHECKPOINT_SHA),('dataset_json',DATA_SHA),('detections',DETECTOR_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key]) != expected:
            raise ValueError('Changed '+key)
    if digest(Path(cfg['chair'])/'chair.pkl') != CHAIR_SHA:
        raise ValueError('CHAIR annotation source differs')
    receipt = checked_cache(cfg['prepared_root'])
    meta = json.loads((Path(cfg['prepared_root'])/'metadata.json').read_bytes())
    aux = json.loads((Path(cfg['prepared_root'])/'auxiliary_facts.json').read_bytes())
    fields = tokenize_fields(aux,meta['word2idx'])
    save(out/'text_tokens.json',fields)
    # Source cache deliberately stored zero prompts on auxiliary-only rows. Actual
    # decoder diagnostics require original prompts; never call zero-prompt C0 native parity.
    import torch
    import h5py
    from captioning.semantic_prompts import align_prompt_cache
    bundle = torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    held_ids = sorted({r['image_id'] for r in aux['partitions']['held']})
    if len(held_ids) != 98:raise ValueError('Held image count differs')
    row_index = {r['coco_id']:i for i,r in enumerate(meta['rows'])}
    held_rows = [row_index[i] for i in held_ids]
    sentinels = [0,1,3999,4000,4999]
    needed = held_rows+sentinels
    names = [meta['rows'][i]['filename'] for i in needed]
    aligned = align_prompt_cache(bundle['data'],names)
    if set(aligned) != set(names):raise ValueError('Original legacy prompts do not cover held decoder diagnostics')
    tokens,masks = [],[]
    for name in names:
        token,mask = aligned[name]['tokens'].float().numpy(),aligned[name]['mask'].numpy()
        if token.shape != (20,512) or mask.shape != (20,) or not np.isfinite(token).all() or not np.isin(mask,[0,1]).all() or not mask.any():
            raise ValueError('Invalid original prompt: '+name)
        tokens.append(token);masks.append(mask)
    with h5py.File(Path(cfg['prepared_root'])/'inputs.h5','r') as f:
        for i,position in enumerate(sentinels,len(held_rows)):
            if not np.array_equal(tokens[i],f['prompt'][position]) or not np.array_equal(masks[i],f['mask'][position]):
                raise ValueError('Legacy prompt sentinel differs from prepared native prompts')
    np.savez(out/'held_native_prompts.npz',indices=np.asarray(held_rows),coco_ids=np.asarray(held_ids),
             prompt=np.stack(tokens[:len(held_rows)]),mask=np.stack(masks[:len(held_rows)]))
    overlay_hash = hashlib.sha256()
    with np.load(out/'held_native_prompts.npz',allow_pickle=False) as overlay:
        for name in ('indices','coco_ids','prompt','mask'):
            value = overlay[name]
            overlay_hash.update(str((name,str(value.dtype),value.shape)).encode())
            overlay_hash.update(value.tobytes())
    del bundle,aligned,tokens,masks
    data = json.loads(Path(cfg['dataset_json']).read_bytes())
    detectors = json.loads(Path(cfg['detections']).read_bytes())
    targets = make_targets(dict(train=meta['rows'][:4000],val=meta['rows'][4000:5000]),data,
        Path(cfg['chair'])/'chair.pkl',detectors,'aliases_v2','unique_count_v2')
    save(out/'object_targets.json',targets)
    save(out/'preparation.json',dict(status='PILOT_PREPARATION_COMPLETE',
        input_receipt_sha256=digest(Path(cfg['prepared_root'])/'cpu_preparation.json'),
        effective_held_prompt_content_sha256=overlay_hash.hexdigest(),original_held_prompts=98,
        payload_sha256={n:digest(out/n) for n in ('text_tokens.json','object_targets.json','protocol.json','held_native_prompts.npz')},
        source_payload_sha256=receipt['payload_sha256'],model_executed=False))
    print('Prepared exact dataset and training-only reward; next user GPU preflight.',flush=True)


class Inputs:
    def __init__(self,root,overlay):
        import h5py
        self.root = Path(root)
        self.meta = json.loads((self.root/'metadata.json').read_bytes())
        self.aux = json.loads((self.root/'auxiliary_facts.json').read_bytes())
        self.fact_by_id = {r['fact_id']:r for rows in self.aux['partitions'].values() for r in rows}
        self.row = {r['coco_id']:i for i,r in enumerate(self.meta['rows'])}
        self.schedule = schedules()
        self.arrays = {}
        shapes = dict(visual=(197,768),prompt=(20,512),mask=(20,),roi=(6,196),union=(30,196),geometry=(30,8),pair_valid=(30,))
        with h5py.File(self.root/'inputs.h5','r') as f:
            if not f.attrs.get('complete') or f['coco_ids'][:].tolist() != [r['coco_id'] for r in self.meta['rows']]:
                raise ValueError('H5 completion/image IDs differ')
            if f['caption_valid'][:].tolist() != [i<5000 for i in range(5472)]:
                raise ValueError('Caption-valid rows differ')
            for key, shape in shapes.items():
                if f[key].shape != (5472,*shape):
                    raise ValueError('H5 shape differs: '+key)
                # Read once per worker; ~2GiB of immutable source arrays, no model feature extraction.
                self.arrays[key] = f[key][:]
                if not np.isfinite(self.arrays[key]).all():
                    raise ValueError('Nonfinite source '+key)
        with np.load(overlay,allow_pickle=False) as extra:
            held_ids = sorted({r['image_id'] for r in self.aux['partitions']['held']})
            indices = [self.row[i] for i in held_ids]
            if extra['indices'].tolist() != indices or extra['coco_ids'].tolist() != held_ids:
                raise ValueError('Native held-prompt overlay identity differs')
            self.arrays['prompt'][indices] = extra['prompt']
            self.arrays['mask'][indices] = extra['mask']
            if not self.arrays['mask'][indices].any(-1).all():
                raise ValueError('Held diagnostic lacks native prompts')
        self.text = tokenize_fields(self.aux,self.meta['word2idx'])

    def batch(self,indices,device):
        import torch
        return {key:torch.as_tensor(value[np.asarray(indices,dtype=int)],device=device,
            dtype=torch.bool if key in ('mask','pair_valid') else torch.float32) for key,value in self.arrays.items()}

    def fact_batch(self,facts,device):
        return self.batch([self.row[r['image_id']] for r in facts],device)

    def mp_facts(self,step):
        group = self.aux['multipositive']['fit'][self.schedule['mp'][step-1]]
        return group,[self.fact_by_id[i] for i in group['positive']+group['negative']]


def log_mp_numpy(scores,positive):
    scores = np.asarray(scores,dtype=float)
    if not 0 < positive < len(scores):
        raise ValueError('MP requires observed P and N')
    logden = np.log(np.exp(scores-scores.max()).sum())+scores.max()
    return float(logden-scores[:positive].mean())


def semantic_metrics(records,groups,score_key,weights=None):
    """AUROC by family + same-query concordance; image weights support cluster bootstrap."""
    weights = weights or {r['image_id']:1 for r in records}
    def concord(pos,neg):
        den = sum(weights.get(a['image_id'],0)*weights.get(b['image_id'],0) for a in pos for b in neg)
        if den == 0:return None
        num = sum(weights.get(a['image_id'],0)*weights.get(b['image_id'],0)*
            ((a[score_key]>b[score_key])+.5*(a[score_key]==b[score_key])) for a in pos for b in neg)
        return float(num/den)
    by_family = {}
    for family in ('contact','containment'):
        pool = [r for r in records if r['family']==family]
        by_family[family] = concord([r for r in pool if r['label']==1],[r for r in pool if r['label']==0])
    by_id = {r['fact_id']:r for r in records}
    group_scores = [concord([by_id[i] for i in g['positive']],[by_id[i] for i in g['negative']]) for g in groups]
    valid = [x for x in group_scores if x is not None]
    return dict(family_auc=by_family,macro_family_auc=float(np.mean(list(by_family.values())))
        if all(x is not None for x in by_family.values()) else None,
        query_concordance=float(np.mean(valid)) if valid else None,valid_query_groups=len(valid),
        invalid_query_groups=len(group_scores)-len(valid))
