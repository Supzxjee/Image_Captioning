"""Fixed best9 TEST prompt/patch contracts; NumPy only, no model execution."""
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
import test199_data as shared
from semantic_data import ARRAYS,record_arrays,checked_root

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='7c11be52fce9cd90dadc55fa514d4017d540f40f630232a68bfe4306c754158b'
VERSION='ducanh2403/kltnnnnnff/9'


def validate_lock(value):
    baseline=shared.lock()
    if any(value[k]!=baseline[k] for k in ('expected_test_ids','test_id_sha256','beam_width','max_new_tokens','length_normalization','reranking')):raise ValueError('Changed fixed TEST cohort/decode')
    if value['source_version']!=VERSION or value['checkpoint_sha256']!=CHECKPOINT_SHA or value['epoch']!=9 or value['updates']!=159399 or value['method']!='bank_prompt_patch_v1':raise ValueError('Need fixed bank+prompt best9, not endpoint10')
    return value


def lock():
    value=shared.lock();value.update(schema='bank_prompt_best9_test_v1',source_version=VERSION,checkpoint_sha256=CHECKPOINT_SHA,
        training_commit='f5f6eb05d7c89917b0719637dae4e948ad73f531',epoch=9,updates=159399,method='bank_prompt_patch_v1')
    return validate_lock(value)


def checkpoint(root):
    root=Path(root);files=[root] if root.is_file() else sorted(root.rglob('best.bin'))
    matches=[p for p in files if p.is_file() and sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Missing exact version9 best.bin SHA7c11be52; no final/latest fallback')
    return matches[0]


def runtime_policy(training,current):
    if training!='2.11.0+cu130' or current not in ('2.11.0+cu128','2.11.0+cu130'):raise RuntimeError('Unreviewed Torch build; keep log, do not blindly reinstall.')
    return dict(training_torch=training,inference_torch=current,cuda_wheel_changed=training!=current,
        policy='Same publicTorch2.11.0 observed CUDA wheels; strictstate and actual GPU beam-path check required.',bitwise_cross_build_equivalence_established=False)


def validate_checkpoint_state(ck):
    s,i=ck['state'],ck['identity'];expected=lock()
    if s['epoch']!=10 or s['batch_cursor']!=0 or s['phase']!='train' or s['updates']!=159399 or s['caption_events']!=5100723 or s['best_epoch']!=9 or len(s['history'])!=9:raise ValueError('Wrong best9 snapshot cursor')
    if i['source']['commit']!=expected['training_commit'] or i['method']!=expected['method'] or i['epochs']!=10 or i['global_batch']!=32:raise ValueError('Wrong checkpoint architecture/training identity')
    if not {'noun_bank','verb_bank'}<=set(ck['model']):raise ValueError('Missing frozen TRAIN banks in checkpoint buffers')
    return expected


def prompt_inputs(cfg):
    inventory=read(HERE/'PROMPT_TEST_PREPARATION_RESULT.json')['transport_inventory']['cache_files']
    # The old ROI feature array/vocabulary are not inputs to the new patch model.
    wanted={n:inventory[n] for n in ('receipt.json','summary.json','prompt_links.jsonl','prompt_texts.json','prompt_embeddings.npy')}
    root=checked_root(cfg['prompt_test_root'],'receipt.json',wanted);texts=read(root/'prompt_texts.json');vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if texts!=sorted(set(texts)) or len(texts)!=258 or vectors.shape!=(258,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,rtol=0,atol=2e-6):raise ValueError('Changed TEST181 text space')
    rows=[json.loads(line) for line in (root/'prompt_links.jsonl').read_text(encoding='utf-8').splitlines()]
    expected=lock()['expected_test_ids']
    if len(rows)!=5000 or sorted(r['coco_id'] for r in rows)!=expected or any(r['group']!='test' for r in rows):raise ValueError('Wrong/duplicate prompt TEST IDs')
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
    receipt=read(out/'stage_receipt.json');receipt['method']='bank_prompt_patch_v1';receipt['prompt181_files_sha256']=seals
    receipt['stage_files_sha256'].update({p.name:sha(p) for p in stage.glob('sem_*.npy')});save(out/'stage_receipt.json',receipt)
    saved=read(out/'test_lock.json');saved['stage_receipt_sha256']=sha(out/'stage_receipt.json');save(out/'test_lock.json',saved)
    save(out/'semantic_test_stage.json',dict(images=5000,objects=sum(len(r['objects']) for r in rows),relations=sum(len(r['relations']) for r in rows),
        text_vectors=258,geometry='same training record_arrays ROI box->fractional14x14patch support; endpoint IDs',TRAIN_VALID_features_read=False,old_ROI_features_used=False,old_vocabulary_used=False))


def load(cfg,verify=True):
    result=shared.load(cfg,verify=verify,validator=validate_lock)
    if result[2].get('method')!='bank_prompt_patch_v1':raise ValueError('Semantic TEST stage incomplete')
    return result
