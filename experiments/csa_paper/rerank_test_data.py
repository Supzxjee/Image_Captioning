"""Metadata-only contracts for the prelocked two-policy TEST evaluation."""
import io
import hashlib
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha,validate_arrays
from plain_test_data import exact_file
from rerank_data import image_paths,DATA_SHA,DETECTION_SHA
from captioning.object_consistency import index_detections
from captioning.multiscore_reranking import rerank_multiscore
from full_xe_decode import check_beam_rows
from candidate_data import content

HERE=Path(__file__).resolve().parent
POLICIES=('decoder_rank0','clip_object_fixed')


def checked_lock(out):
    lock=read(Path(out)/'test_lock.json');expected=read(HERE/'RERANK_TEST_INPUTS.json')
    if lock['checkpoint_sha256']!=expected['checkpoint_sha256'] or lock['selector']!=expected['frozen_selector'] or lock['clip']!=expected['clip']:
        raise ValueError('TEST checkpoint/selector/CLIP lock changed')
    if lock['images']!=5000 or lock['beam']!=5 or lock['max_new_tokens']!=20 or lock['cache_sha256']!=expected['test_receipt']['files']:
        raise ValueError('TEST population/decoding/cache lock changed')
    return lock


def resolve_inputs(cfg,locked=None):
    locked=locked or read(HERE/'RERANK_TEST_INPUTS.json')
    checkpoint=exact_file(cfg['checkpoint_root'],'best.bin',locked['checkpoint_sha256'])
    files={n:exact_file(cfg['test_root'],n,h) for n,h in locked['test_receipt']['files'].items()}
    return checkpoint,files


def rebuild_cache(cfg,locked=None):
    from prepare_test_regions import prepare as original_prepare
    locked=locked or read(HERE/'RERANK_TEST_INPUTS.json')
    visual=Path(cfg['visual_cache']);root=Path(cfg['test_root'])
    if not visual.is_dir():raise FileNotFoundError('Add original visual-cache H5 Dataset before restoring TEST cache: '+str(visual))
    if root.exists():raise ValueError('Preserve existing TEST restoration output; no automatic retry')
    root.mkdir(parents=True)
    image_root=Path(cfg['coco_root'])/'images'
    if not image_root.is_dir():image_root=Path(cfg['coco_root'])
    original_prepare(dict(dataset_json=cfg['dataset_json'],detections=cfg['detections'],visual_cache=str(visual),image_root=str(image_root),output=str(root)))
    for n,h in locked['test_receipt']['files'].items():exact_file(root,n,h)
    save(Path(cfg['output'])/'cache_restoration.json',dict(status='EXACT_TEST_CACHE_RESTORED',mode='original_H5_CPU_pooling',
        reference_cache_version=locked['test_cache_version'],file_sha256=locked['test_receipt']['files'],neural_extraction=False))
    print('Restored TEST NPZ/metadata/vocab byte-for-byte match prelocked reference; no model forward.',flush=True)


def prepare(cfg):
    from PIL import Image
    out=Path(cfg['output']);locked=read(HERE/'RERANK_TEST_INPUTS.json');checkpoint,files=resolve_inputs(cfg,locked)
    for key,h in (('dataset_json',DATA_SHA),('detections',DETECTION_SHA)):
        if sha(cfg[key])!=h:raise ValueError('Changed TEST source '+key)
    metadata=read(files['test_metadata.json']);vocab=read(files['vocabulary.json'])
    with np.load(files['test_regions.npz'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
    validate_arrays(a,metadata,vocab)
    ids=[r['coco_id'] for r in metadata]
    if len(metadata)!=5000 or len(set(ids))!=5000 or any(r['source_split']!='test' for r in metadata):raise ValueError('Not full TEST cache')
    if hashlib.sha256(str(ids).encode()).hexdigest()!=locked['test_receipt']['test_id_order_sha256']:raise ValueError('TEST ID order differs')
    dataset=read(cfg['dataset_json'])['images'];lookup={int(r['cocoid']):r for r in dataset if r['split']=='test'}
    if set(lookup)!=set(ids):raise ValueError('Wrong Karpathy TEST cohort')
    paths=image_paths(cfg['coco_root'],list(lookup.values()));detected=index_detections(read(cfg['detections']));payload=[];refs=[]
    for j,r in enumerate(metadata):
        identity=r['coco_id'];source=lookup[identity];name=r['filename']
        if source['filename']!=name or [s['raw'] for s in source['sentences']]!=r['captions']:raise ValueError('TEST filename/reference provenance differs')
        if name not in detected:raise ValueError('Missing detector row: '+name)
        d=detected[name]
        if any(not np.isfinite(v) or not .5<=v<=1 for v in d.values()):raise ValueError('Invalid accepted detector confidence')
        raw=paths[name].read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.size!=(r['width'],r['height']):raise ValueError('Raw image/cache dimensions differ')
            image.verify()
        payload.append(dict(coco_id=identity,filename=name,image_path=str(paths[name]),image_sha256=hashlib.sha256(raw).hexdigest(),detected=d))
        refs.append(dict(coco_id=identity,references=r['captions']))
        if (j+1)%500==0:print('Verified TEST cache/rawimages/detector',j+1,'/5000',flush=True)
    save(out/'inference_inputs.json',payload);save(out/'test_references.json',refs)
    save(out/'test_lock.json',dict(status='CSA161_FIXED_RERANK_TEST_LOCKED',test_used=True,training=False,
        checkpoint=str(checkpoint),checkpoint_sha256=locked['checkpoint_sha256'],checkpoint_version=locked['checkpoint_version'],
        cache_version=locked['test_cache_version'],cache_input_mode=cfg.get('test_cache_mode','verified_mounted'),
        cache_files={n:str(p) for n,p in files.items()},cache_sha256=locked['test_receipt']['files'],
        images=5000,beam=5,max_new_tokens=20,selector=locked['frozen_selector'],clip=locked['clip'],
        inference_sha256=sha(out/'inference_inputs.json'),references_sha256=sha(out/'test_references.json'),
        policy_selection='two prelocked policies; no TEST tuning/fallback; prior project TEST exposure declared'))


def load_features(lock):
    for n,h in lock['cache_sha256'].items():
        if sha(lock['cache_files'][n])!=h:raise ValueError('TEST payload changed: '+n)
    with np.load(lock['cache_files']['test_regions.npz'],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    return arrays,read(lock['cache_files']['vocabulary.json'])


def fixed_rank(row,clipscore,detected):
    if len(row['candidates'])!=5 or len(clipscore)!=5 or any(not np.isfinite(x) or not 0<=x<=2.50001 for x in clipscore):raise ValueError('Invalid TEST CLIP scores/width')
    scores=[c['logprob'] for c in row['candidates']]
    if not all(np.isfinite(scores)) or scores!=sorted(scores,reverse=True):raise ValueError('Invalid beam logprobs')
    enriched=[dict(c,clipscore=float(s)) for c,s in zip(row['candidates'],clipscore)]
    chosen,_=rerank_multiscore(dict(candidates=enriched),detected,.4,.3)
    return int(chosen['original_rank'])


def read_beams(out,lock):
    import json
    out=Path(out);inputs=read(out/'inference_inputs.json');by_id={}
    vocab=read(lock['cache_files']['vocabulary.json'])
    for rank in (0,1):
        path=out/f'beams_rank{rank}.jsonl';w=read(out/f'decode_worker{rank}.json')
        if not w['complete'] or w['images']!=2500 or w['checkpoint_sha256']!=lock['checkpoint_sha256'] or sha(path)!=w['beams_sha256']:raise ValueError('Incomplete TEST decode worker')
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [r['coco_id'] for r in rows]!=[r['coco_id'] for r in inputs[rank::2]]:raise ValueError('TEST beam ID order differs')
        for row in rows:
            candidates=row['candidates']
            check_beam_rows([([c['ids'] for c in candidates],[c['logprob'] for c in candidates])],1,10000)
            if any(c['caption']!=' '.join(vocab[t] for t in content(c['ids'])) for c in candidates):raise ValueError('Beam token/caption differs')
            by_id[row['coco_id']]=row
    if len(by_id)!=5000:raise ValueError('Not full TEST beams')
    return [by_id[r['coco_id']] for r in inputs]
