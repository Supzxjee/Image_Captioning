"""Verify the sealed TRAIN-only manifest, then join five-reference prediction positions."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from pilot_data import digest


def read_manifest(cfg, dataset, tokenizer):
    directory = Path(cfg['noun_manifest'])
    receipt_path = directory / 'manifest_receipt.json'
    if digest(receipt_path) != cfg['noun_receipt_sha256']:
        raise ValueError('Different noun manifest receipt; do not replace the locked labels.')
    receipt = json.loads(receipt_path.read_text())
    for name, expected in receipt['payload_sha256'].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError('Manifest payload differs: ' + name)
    vocab_sha = hashlib.sha256(json.dumps(tokenizer.word2idx, sort_keys=True).encode()).hexdigest()
    if vocab_sha != receipt['vocabulary_sha256']:
        raise ValueError('Tokenizer vocabulary/order differs.')
    spans = json.loads((directory / 'spans.json').read_text())
    masks = json.loads((directory / 'mask_bank.json').read_text())
    schedule = json.loads((directory / 'supervision_schedule.json').read_text())
    groups = {r['coco_id']: r for r in cfg['groups']['train']}
    if len(groups) != 10000 or len(schedule) != 10000 or {r['coco_id'] for r in schedule} != set(groups):
        raise ValueError('Main TRAIN10k image population differs.')
    span_ids = {s['span_id'] for s in spans}
    scheduled = [s for row in schedule for s in row['span_ids']]
    if len(span_ids) != 17904 or len(spans) != 17904 or len(scheduled) != 17904 or set(scheduled) != span_ids:
        raise ValueError('Span IDs/count differ.')
    by_image = defaultdict(list)
    for s in spans:
        group = groups[s['coco_id']]
        if group['image_id'] != s['karpathy_imgid'] or not 0 <= s['caption_index'] < 5:
            raise ValueError('Span image/reference identity differs.')
        row = dataset['images'][group['image_id']]
        caption = row['sentences'][s['caption_index']]['raw']
        ids, _ = tokenizer.encode(caption)
        positions = s['decoder_logit_positions']
        if caption != s['caption'] or max(positions) >= len(ids) - 1:
            raise ValueError('Caption/truncation differs.')
        if [int(ids[p+1]) for p in positions] != s['token_ids'] or set(s['token_ids']) & {0,1,2,3}:
            raise ValueError('Wrong teacher prediction offset/token IDs.')
        mask = masks[s['mask_key']]
        a = np.asarray(mask['patch_area_fraction'], dtype=np.float32).reshape(-1)
        if mask['coco_id'] != s['coco_id'] or mask['category'] != s['category']:
            raise ValueError('Mask endpoint/class differs.')
        if a.shape != (196,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() or a.sum() < 3.9999:
            raise ValueError('Invalid fractional square-resize patch mask.')
        by_image[s['coco_id']].append(s)
    if len(by_image) != 4495 or sum(s['category']=='person' for s in spans) != 4476:
        raise ValueError('Supervised population/person cap differs.')
    for row in schedule:
        if groups[row['coco_id']]['image_id'] != row['karpathy_imgid'] or set(row['span_ids']) != {s['span_id'] for s in by_image[row['coco_id']]}:
            raise ValueError('Schedule join differs.')
    return by_image, masks, max(map(len, by_image.values())), max(len(s['token_ids']) for s in spans)


def tensors_for(rows, masks, slots, width):
    result = dict(span_valid=np.zeros(slots, dtype=bool), span_person=np.zeros(slots, dtype=bool),
        span_caption=np.zeros(slots, dtype=np.int64), span_positions=np.zeros((slots,width), dtype=np.int64),
        span_token_valid=np.zeros((slots,width), dtype=bool), span_masks=np.zeros((slots,196), dtype=np.float32))
    for i, row in enumerate(rows):
        n = len(row['decoder_logit_positions'])
        result['span_valid'][i] = True
        result['span_person'][i] = row['category']=='person'
        result['span_caption'][i] = row['caption_index']
        result['span_positions'][i,:n] = row['decoder_logit_positions']
        result['span_token_valid'][i,:n] = True
        result['span_masks'][i] = np.asarray(masks[row['mask_key']]['patch_area_fraction']).reshape(-1)
    return result
