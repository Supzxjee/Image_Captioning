"""Frozen M2-only TEST contracts; metadata only, no neural imports."""
from pathlib import Path
from real_bank_data import read, save, sha
import test199_data as shared

HERE = Path(__file__).resolve().parent
VERSION = 'ducanh2403/2-vi-cai-kia-qua-nhieu/245'
CHECKPOINT_SHA = '3fe31a921742e841c0d3863aac9ab9f00b0cd6b46289f3cf36029ac1fa168226'
METHOD = 'bank_only_parallel3_xe10_v1'
runtime_policy = shared.runtime_policy


def validate_lock(value):
    original = shared.lock()
    for key in ('expected_test_ids', 'test_id_sha256', 'beam_width', 'max_new_tokens', 'length_normalization', 'reranking'):
        if value[key] != original[key]:
            raise ValueError('Changed fixed TEST cohort or decoding rule: '+key)
    expected = dict(source_version=VERSION, checkpoint_sha256=CHECKPOINT_SHA, epoch=9,
                    updates=159399, method=METHOD, seed=42)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Need exact M2-only v245 best epoch9')
    return value


def lock():
    value = shared.lock()
    value.update(schema='bank_only_parallel3_test5000_v1', source_version=VERSION,
                 checkpoint_sha256=CHECKPOINT_SHA, epoch=9, updates=159399, method=METHOD,
                 seed=42, training_commit='5eb012279e6fd0b5e47a56710ed5e3efa7449d2f')
    return validate_lock(value)


def checkpoint(root):
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob('best.bin'))
    matches = [p for p in files if p.is_file() and sha(p) == CHECKPOINT_SHA]
    if len(matches) != 1:
        raise ValueError('Need one exact v245 best.bin; no final/latest/full-model fallback')
    return matches[0]


def validate_checkpoint_state(ck):
    s, i = ck['state'], ck['identity']
    wanted = dict(epoch=10, batch_cursor=0, phase='train', updates=159399,
                  caption_events=5100723, best_epoch=9)
    if any(s[k] != v for k, v in wanted.items()) or len(s['history']) != 9:
        raise ValueError('Wrong M2-only best9 snapshot cursor')
    if [r['epoch'] for r in s['history']] != list(range(1, 10)):
        raise ValueError('Incomplete checkpoint history')
    expected = dict(method=METHOD, seed=42, epochs=10, global_batch=32, dropout=.2,
                    decoder_layers=3, semantic_conditioning=True, image_specific_prompts=False, sharing=False,
                    sampler='image_grouped_42_plus_epoch')
    if any(i.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong M2-only architecture/training identity')
    sealed = read(HERE/'M2_ONLY_TRAINING_LOCK.json')
    if i['source']['sha256'] != sealed['source_sha256']:
        raise ValueError('Changed M2-only training sources')
    if i['source']['commit'] != lock()['training_commit']:
        raise ValueError('Wrong pinned training ancestor')
    if abs(s['best_CIDEr']-sealed['best_VALID_CIDEr']) > 1e-12:
        raise ValueError('Wrong VALID-selected best checkpoint')
    names = ck['model']
    prefixes = ('core.project.', 'core.noun_agg.', 'core.verb_agg.', 'core.object1.',
                'core.object2.', 'core.relation.', 'core.embedding.', 'core.decoder.', 'core.head.')
    if len(names) != 166 or any(not (name.startswith(prefixes) or name in ('core.nouns','core.verbs')) for name in names):
        raise ValueError('Expected only M2/core states, no M3 or baseline-only checkpoint')
    if any(not any(name.startswith(prefix) for name in names) for prefix in prefixes):
        raise ValueError('Missing M2 or decoder state')
    if tuple(names['core.nouns'].shape) != (5269,512) or tuple(names['core.verbs'].shape) != (1594,512):
        raise ValueError('Wrong checkpoint bank dimensions')
    if tuple(names['core.embedding.weight'].shape) != (10000, 512):
        raise ValueError('Wrong frozen TRAIN vocabulary or embedding shape')
    return lock()


def prepare(cfg):
    if cfg.get('semantic') or not cfg.get('bank_only'):
        raise ValueError('M2-only TEST requires bank_only=True/semantic=False')
    shared.prepare(cfg, locked=lock(), weight_path=checkpoint(cfg['checkpoint_root']))
    out = Path(cfg['output'])
    receipt = read(out/'stage_receipt.json')
    receipt['method'] = METHOD
    save(out/'stage_receipt.json', receipt)
    locked = read(out/'test_lock.json')
    locked['stage_receipt_sha256'] = sha(out/'stage_receipt.json')
    save(out/'test_lock.json', locked)


def load(cfg, verify=True):
    values = shared.load(cfg, verify=verify, validator=validate_lock)
    if values[2].get('method') != METHOD:
        raise ValueError('Wrong M2-only TEST staging method')
    return values
