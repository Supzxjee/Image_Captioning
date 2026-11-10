"""Small TRAIN-only prompt staging; reuse existing pinned bank/instance contracts."""
import json
from pathlib import Path
import numpy as np
from real_bank_data import read, save, sha
from semantic_data import ARRAYS, checked_inputs, record_arrays


def probe_events(caption_image, relation_valid):
    images, first = np.unique(caption_image, return_index=True)
    if not np.array_equal(images, np.arange(len(relation_valid))) or len(images) < 32 or not relation_valid.any():
        raise ValueError('Missing TRAIN256 prompt-role probe coverage; send ZIP')
    order = np.argsort(-relation_valid.any(1).astype(np.int8), kind='stable')[:32]
    return first[order]


def selected_prompts(rows, ids, texts):
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate selected TRAIN IDs')
    index = {i: j for j, i in enumerate(ids)}
    arrays = {k: np.zeros((len(ids),) + shape, dtype=dtype) for k, (shape, dtype) in ARRAYS.items()}
    seen = set()
    for row in rows:
        i = row['coco_id']
        if type(i) is not int:
            raise ValueError('Noninteger prompt ID')
        if i not in index:
            continue
        if i in seen or row['group'] != 'train':
            raise ValueError('Duplicate/foreign split selected prompt')
        one = record_arrays(row, texts)
        for k, value in one.items():
            arrays[k][index[i]] = value
        seen.add(i)
    if seen != set(ids):
        raise ValueError('Missing selected TRAIN prompts')
    return arrays


def stage_prompts(cfg):
    stage = Path(cfg['stage'])
    inv, root, cohort, texts, vectors, _, banks, bank_package = checked_inputs(cfg)
    ids = np.load(stage/'coco_ids.npy', allow_pickle=False)
    if len(ids) != 256 or not set(map(int, ids)) <= set(cohort['train']):
        raise ValueError('Smoke must use exactly256TRAINimages')
    with (root/'prompt_links.jsonl').open(encoding='utf-8') as stream:
        arrays = selected_prompts((json.loads(line) for line in stream), ids, texts)
    for k, value in arrays.items():
        np.save(stage/('sem_'+k+'.npy'), value)
    for k, value in banks.items():
        np.save(stage/('sem_'+k+'_bank.npy'), value)
    np.save(stage/'sem_text_vectors.npy', vectors)
    names = ['sem_'+k+'.npy' for k in ARRAYS] + ['sem_noun_bank.npy', 'sem_verb_bank.npy', 'sem_text_vectors.npy']
    receipt = dict(status='CSA_STRUCTURE_TRAIN256_SEMANTICS_COMPLETE', images=256,
                   objects=int(arrays['object_valid'].sum()), relations=int(arrays['relation_valid'].sum()),
                   files_sha256={n: sha(stage/n) for n in names}, prompt_source=inv['cache_files'],
                   bank_source=bank_package['files'], test_used=False, validation_used=False,
                   note='Full cache manifests verified; only selected TRAIN visual rows staged. No encoder execution.')
    save(stage/'csa_semantics.json', receipt)
    save(Path(cfg['output'])/'semantic_stage.json', receipt)
    print('CSA structure semantic stage:', receipt['images'], 'TRAIN images;', receipt['objects'], 'object slots;', receipt['relations'], 'relation slots', flush=True)


def checked_semantics(stage):
    stage = Path(stage); receipt = read(stage/'csa_semantics.json')
    if receipt['status'] != 'CSA_STRUCTURE_TRAIN256_SEMANTICS_COMPLETE' or receipt['images'] != 256 or receipt['test_used'] or receipt['validation_used']:
        raise ValueError('Wrong semantic scope')
    for n, h in receipt['files_sha256'].items():
        if Path(n).name != n or sha(stage/n) != h:
            raise ValueError('Changed smoke semantic payload: '+n)
    return receipt
