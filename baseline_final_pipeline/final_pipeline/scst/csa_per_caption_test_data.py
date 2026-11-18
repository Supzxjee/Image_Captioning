"""Fixed best9 TEST prompt/patch contracts; NumPy only, no model execution."""
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
import test199_data as shared
from semantic_data import ARRAYS,record_arrays,checked_root

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'
VERSION='csa_structure_per_caption_full_xe10_resume124177'


def validate_lock(value):
    baseline=valid_cohort()
    if any(value[k]!=baseline[k] for k in ('expected_test_ids','test_id_sha256','beam_width','max_new_tokens','length_normalization','reranking')):raise ValueError('Changed fixed TEST cohort/decode')
    if value['source_version']!=VERSION or value['checkpoint_sha256']!=CHECKPOINT_SHA or value['epoch']!=9 or value['updates']!=159399 or value['method']!='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1':raise ValueError('Need fixed bank+prompt best9, not endpoint10')
    return value


def valid_cohort():
    import hashlib
    value=shared.lock()
    value.update(expected_test_ids=VALID_IDS,test_id_sha256=hashlib.sha256(json.dumps(VALID_IDS,separators=(',',':')).encode()).hexdigest())
    return value

def lock():
    value=valid_cohort();value.update(schema='csa_per_caption_best9_test_v1',source_version=VERSION,checkpoint_sha256=CHECKPOINT_SHA,
        training_commit='5eb012279e6fd0b5e47a56710ed5e3efa7449d2f',epoch=9,updates=159399,method='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1')
    return validate_lock(value)


def checkpoint(root):
    root=Path(root);files=[root] if root.is_file() else sorted(root.rglob('best.bin'))
    matches=[p for p in files if p.is_file() and sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Missing exact version9 best.bin SHAb0c5bde0; no final/latest fallback')
    return matches[0]


def runtime_policy(training,current):
    if training!='2.11.0+cu128' or current not in ('2.11.0+cu128','2.11.0+cu130'):raise RuntimeError('Unreviewed Torch build; keep log, do not blindly reinstall.')
    return dict(training_torch=training,inference_torch=current,cuda_wheel_changed=training!=current,
        policy='Same publicTorch2.11.0 observed CUDA wheels; strictstate and actual GPU beam-path check required.',bitwise_cross_build_equivalence_established=False)


def validate_checkpoint_state(ck):
    s,i=ck['state'],ck['identity'];expected=lock()
    if s['epoch']!=10 or s['batch_cursor']!=0 or s['phase']!='train' or s['updates']!=159399 or s['caption_events']!=5100723 or s['best_epoch']!=9 or len(s['history'])!=9:raise ValueError('Wrong best9 snapshot cursor')
    if i['source']['commit']!=expected['training_commit'] or i['method']!=expected['method'] or i['epochs']!=10 or i['global_batch']!=32:raise ValueError('Wrong checkpoint architecture/training identity')
    if not {'core.nouns','core.verbs'}<=set(ck['model']):raise ValueError('Missing frozen TRAIN banks in checkpoint buffers')
    return expected


def configure_valid(cfg):
    global VALID_IDS
    if sha(cfg['dataset_json'])!=shared.DATA_SHA:raise ValueError('Changed Karpathy metadata')
    def val_rows(ds, locked):
        rows=sorted((r for r in ds['images'] if r['split']=='val'),key=lambda r:int(r['cocoid']))
        if [int(r['cocoid']) for r in rows]!=locked['expected_test_ids']:raise ValueError('Wrong VALID IDs')
        return rows
    VALID_IDS=sorted(int(r['cocoid']) for r in read(cfg['dataset_json'])['images'] if r['split']=='val')
    if len(VALID_IDS)!=5000 or len(set(VALID_IDS))!=5000:raise ValueError('Need VALID5000')
    shared.selected_test_rows=val_rows

def prompt_inputs(cfg):
    configure_valid(cfg)
    inv=read(HERE/'FULL_PROMPT_FRESH_V175_RESULT.json')['transport_inventory']['cache_files']
    wanted={n:inv[n] for n in ('cache_receipt.json','cohort.json','prompt_links.jsonl','prompt_texts.json','prompt_embeddings.npy')}
    root=checked_root(cfg['prompt_test_root'],'cache_receipt.json',wanted)
    cohort=read(root/'cohort.json')
    if cohort['test'] or sorted(cohort['val'])!=VALID_IDS:raise ValueError('Wrong prompt175 VALID cohort')
    texts=read(root/'prompt_texts.json')
    vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if texts!=sorted(set(texts)) or vectors.shape!=(410,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all():raise ValueError('Changed prompt175 embeddings')
    with (root/'prompt_links.jsonl').open(encoding='utf8') as stream:
        rows=[r for r in map(json.loads,stream) if r['group']=='val']
    if sorted(r['coco_id'] for r in rows)!=VALID_IDS:raise ValueError('Wrong VALID prompt links')
    return rows,texts,vectors,wanted

def prepare(cfg):
    rows,texts,vectors,seals=prompt_inputs(cfg)
    for row in rows:record_arrays(row,texts)
    shared.prepare(cfg,locked=lock(),weight_path=checkpoint(cfg['checkpoint_root']))
    stage_prompts(cfg,rows,texts,vectors,seals)


def stage_prompts(cfg,rows,texts,vectors,seals):
    encoded=[record_arrays(row,texts) for row in rows]
    stage=Path(cfg['stage']);out=Path(cfg['output']);ids=np.load(stage/'ids.npy',allow_pickle=False);lookup={r['coco_id']:j for j,r in enumerate(rows)}
    order=[lookup[int(i)] for i in ids]
    for key in ARRAYS:np.save(stage/('sem_'+key+'.npy'),np.stack([encoded[j][key] for j in order]))
    np.save(stage/'sem_text_vectors.npy',vectors)
    receipt=read(out/'stage_receipt.json');receipt['method']='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1';receipt['prompt175_files_sha256']=seals
    receipt['stage_files_sha256'].update({p.name:sha(p) for p in stage.glob('sem_*.npy')});save(out/'stage_receipt.json',receipt)
    saved=read(out/'test_lock.json');saved['stage_receipt_sha256']=sha(out/'stage_receipt.json');save(out/'test_lock.json',saved)
    save(out/'semantic_test_stage.json',dict(images=5000,objects=sum(len(r['objects']) for r in rows),relations=sum(len(r['relations']) for r in rows),
        text_vectors=410,geometry='same training record_arrays ROI box->fractional14x14patch support; endpoint IDs',TRAIN_TEST_features_read=False,old_ROI_features_used=False,old_vocabulary_used=False))


def load(cfg,verify=True):
    result=shared.load(cfg,verify=verify,validator=validate_lock)
    if result[2].get('method')!='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1':raise ValueError('Semantic TEST stage incomplete')
    return result
