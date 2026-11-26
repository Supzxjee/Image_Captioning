"""Metadata-only epoch transition and exact continuation contracts."""
import math

PARENT_VERSION = 'ducanh3/notebook7bc83c4c76/199'
PARENT_SHA = '5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c'
XE_SHA = 'a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'
IMAGES, GLOBAL_BATCH, PARENT_STEPS, END_STEPS = 113287, 32, 3541, 7082


def validate_parent(state, expected):
    if (state['identity'] != expected['identity'] or state['updates'] != PARENT_STEPS
            or state['cursor_images'] != IMAGES or state['source_sha256'] != XE_SHA
            or not expected['complete_epoch'] or expected['sha256'] != PARENT_SHA):
        raise ValueError('Need complete exact epoch1/v199 state')
    if len(state['rng']) != 2 or not {'model', 'optimizer'} <= state.keys():
        raise ValueError('Need full model/Adam/two-rank RNG, not pilot weights')
    for rng in state['rng']:
        if not {'cpu', 'cuda', 'numpy'} <= rng.keys():
            raise ValueError('Missing rank RNG state')


def make_identity(cfg, expected, torch_version, numpy_version, vocabulary_sha):
    old = expected['identity']
    if (torch_version != old['torch'] or numpy_version != old['numpy']
            or vocabulary_sha != old['vocabulary_sha256']
            or cfg['cache_manifest_sha256'] != old['cache_manifest_sha256']):
        raise ValueError('Changed software, vocabulary or cache; no fallback')
    return dict(method='beam_mean_scst_epoch2_speed_v1', epoch=2, seed=43,
                parent_version=PARENT_VERSION, parent_sha256=PARENT_SHA, source_sha256=XE_SHA,
                parent_identity=old, source_hashes=cfg['source_hashes'],
                train_images=IMAGES, cumulative_image_events=2*IMAGES,
                parent_updates=PARENT_STEPS, endpoint_updates=END_STEPS,
                order='same original sealed TRAIN order, once per epoch',
                optimization='audited batched prompt transfer/deferred finite-logit check')


def validate_resume(receipt, identity):
    cursor, step = receipt['cursor_images'], receipt['updates']
    if (receipt['identity'] != identity or receipt['epoch'] != 2
            or receipt['parent_sha256'] != PARENT_SHA or receipt['source_sha256'] != XE_SHA
            or type(cursor) is not int or type(step) is not int or not 0 <= cursor <= IMAGES
            or step != PARENT_STEPS+math.ceil(cursor/GLOBAL_BATCH)):
        raise ValueError('Foreign epoch2 recipe, lineage or cursor')
    if cursor != IMAGES and cursor != (step-PARENT_STEPS)*GLOBAL_BATCH:
        raise ValueError('Resume cursor is not an epoch2 optimizer boundary')
    if receipt['complete_epoch'] != (cursor == IMAGES):
        raise ValueError('Wrong epoch2 completeness')


def validate_eval_binding(binding):
    if (binding['epoch'] != 2 or binding['updates'] != END_STEPS or binding['seed'] != 43
            or binding['cursor_images'] != IMAGES or binding['parent_sha256'] != PARENT_SHA
            or binding['source_sha256'] != XE_SHA or not binding['complete_epoch']):
        raise ValueError('Refuse partial/foreign endpoint for TEST')
    digest = binding['sha256']
    if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or digest in (PARENT_SHA, '3b03acc29f025b07dc358846b30618ae9c83f8634bc12bf7905e9d908b971f24')):
        raise ValueError('Wrong epoch2 endpoint SHA')
    return binding
