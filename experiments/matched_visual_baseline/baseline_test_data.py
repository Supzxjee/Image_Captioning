"""Frozen matched-baseline TEST contracts; metadata only, no neural imports."""
from pathlib import Path
from real_bank_data import read, save, sha
import test199_data as shared

HERE = Path(__file__).resolve().parent
VERSION = 'ducanh3/notebook7bc83c4c76/188'
CHECKPOINT_SHA = '1eade1fca3b872091b2f2a345a04d8141a887d3fcd1e5772e4952a495c5f4e6b'
METHOD = 'matched_visual_parallel3_xe10_v1'
runtime_policy = shared.runtime_policy


def validate_lock(value):
    original = shared.lock()
    for key in ('expected_test_ids', 'test_id_sha256', 'beam_width', 'max_new_tokens', 'length_normalization', 'reranking'):
        if value[key] != original[key]:
            raise ValueError('Changed fixed TEST cohort or decoding rule: '+key)
    expected = dict(source_version=VERSION, checkpoint_sha256=CHECKPOINT_SHA, epoch=9,
                    updates=159399, method=METHOD, seed=42)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Need exact matched baseline v188 best epoch9')
    return value


def lock():
    value = shared.lock()
    value.update(schema='matched_visual_parallel3_test5000_v1', source_version=VERSION,
                 checkpoint_sha256=CHECKPOINT_SHA, epoch=9, updates=159399, method=METHOD,
                 seed=42, training_commit='5eb012279e6fd0b5e47a56710ed5e3efa7449d2f')
    return validate_lock(value)


def checkpoint(root):
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob('best.bin'))
    matches = [p for p in files if p.is_file() and sha(p) == CHECKPOINT_SHA]
    if len(matches) != 1:
        raise ValueError('Need one exact v188 best.bin; no final/latest/full-model fallback')
    return matches[0]


def validate_checkpoint_state(ck):
    s, i = ck['state'], ck['identity']
    wanted = dict(epoch=10, batch_cursor=0, phase='train', updates=159399,
                  caption_events=5100723, best_epoch=9)
    if any(s[k] != v for k, v in wanted.items()) or len(s['history']) != 9:
        raise ValueError('Wrong matched baseline best9 snapshot cursor')
    if [r['epoch'] for r in s['history']] != list(range(1, 10)):
        raise ValueError('Incomplete checkpoint history')
    expected = dict(method=METHOD, seed=42, epochs=10, global_batch=32, dropout=.2,
                    decoder_layers=3, semantic_conditioning=False, sharing=False,
                    sampler='image_grouped_42_plus_epoch')
    if any(i.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong matched baseline architecture/training identity')
    sealed = read(HERE/'BASELINE_TRAINING_LOCK.json')
    if i['source']['sha256'] != sealed['source_sha256']:
        raise ValueError('Changed matched-baseline training sources')
    if i['source']['commit'] != lock()['training_commit']:
        raise ValueError('Wrong pinned training ancestor')
    if abs(s['best_CIDEr']-sealed['best_VALID_CIDEr']) > 1e-12:
        raise ValueError('Wrong VALID-selected best checkpoint')
    names = ck['model']
    prefixes = ('core.project.', 'core.embedding.', 'core.decoder.', 'core.head.')
    if len(names) != 77 or any(not name.startswith(prefixes) for name in names):
        raise ValueError('Semantic modules present or baseline decoder state missing')
    if tuple(names['core.embedding.weight'].shape) != (10000, 512):
        raise ValueError('Wrong frozen TRAIN vocabulary or embedding shape')
    return lock()


def prepare(cfg):
    if cfg.get('semantic') or not cfg.get('matched_baseline'):
        raise ValueError('Visual-only TEST requires matched_baseline=True/semantic=False')
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
        raise ValueError('Wrong baseline TEST staging method')
    return values
