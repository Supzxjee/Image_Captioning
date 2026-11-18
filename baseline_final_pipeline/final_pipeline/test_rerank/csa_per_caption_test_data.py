"""Fixed best9 TEST prompt/patch contracts; NumPy only, no model execution."""
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
import test199_data as shared
from semantic_data import ARRAYS,record_arrays,checked_root

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='80e1280b9eef216c5db421a738c7a5e0550bb34b1a62a2a13b2c8e76d40b1a24'
XE_SOURCE_SHA='b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'
PARENT_SHA='56a90b2223b3bb64e433a342d048389dc3e80693fa574c2c1543e31e39a4377b'
VERSION='csa_scst_only1500_rerank_valid5000'


def validate_lock(value):
    baseline=shared.lock()
    if any(value[k]!=baseline[k] for k in ('expected_test_ids','test_id_sha256','beam_width','max_new_tokens','length_normalization','reranking')):raise ValueError('Changed fixed TEST cohort/decode')
    if value['source_version']!=VERSION or value['checkpoint_sha256']!=CHECKPOINT_SHA or value['epoch']!=9 or value['updates']!=1500 or value['method']!='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1':raise ValueError('Wrong frozen SCST1500 TEST lock')
    return value


def lock():
    value=shared.lock();value.update(schema='csa_per_caption_best9_test_v1',source_version=VERSION,checkpoint_sha256=CHECKPOINT_SHA,
        training_commit='5eb012279e6fd0b5e47a56710ed5e3efa7449d2f',epoch=9,updates=1500,method='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1')
    return validate_lock(value)


def checkpoint(root):
    root=Path(root);files=[root] if root.is_file() else sorted(root.rglob('pilot_checkpoint.bin'))
    matches=[p for p in files if p.is_file() and sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Missing exact v94 update1500 checkpoint SHA80e1280b')
    return matches[0]


def runtime_policy(training,current):
    if training!='2.11.0+cu128' or current not in ('2.11.0+cu128','2.11.0+cu130'):raise RuntimeError('Unreviewed Torch build; keep log, do not blindly reinstall.')
    return dict(training_torch=training,inference_torch=current,cuda_wheel_changed=training!=current,
        policy='Same publicTorch2.11.0 observed CUDA wheels; strictstate and actual GPU beam-path check required.',bitwise_cross_build_equivalence_established=False)


def validate_checkpoint_state(ck):
    if ck['updates']!=1500 or ck['source_sha256']!=XE_SOURCE_SHA or ck['parent_sha256']!=PARENT_SHA:
        raise ValueError('Wrong SCST1500 checkpoint lineage')
    if not {'core.nouns','core.verbs'}<=set(ck['model']):
        raise ValueError('Missing frozen TRAIN banks')
    return lock()


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
    receipt=read(out/'stage_receipt.json');receipt['method']='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1';receipt['prompt181_files_sha256']=seals
    receipt['stage_files_sha256'].update({p.name:sha(p) for p in stage.glob('sem_*.npy')});save(out/'stage_receipt.json',receipt)
    saved=read(out/'test_lock.json');saved['stage_receipt_sha256']=sha(out/'stage_receipt.json');save(out/'test_lock.json',saved)
    save(out/'semantic_test_stage.json',dict(images=5000,objects=sum(len(r['objects']) for r in rows),relations=sum(len(r['relations']) for r in rows),
        text_vectors=258,geometry='same training record_arrays ROI box->fractional14x14patch support; endpoint IDs',TRAIN_VALID_features_read=False,old_ROI_features_used=False,old_vocabulary_used=False))


def load(cfg,verify=True):
    result=shared.load(cfg,verify=verify,validator=validate_lock)
    if result[2].get('method')!='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1':raise ValueError('Semantic TEST stage incomplete')
    return result
