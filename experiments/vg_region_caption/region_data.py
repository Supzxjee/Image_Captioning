"""Locked weak VG supervision and patch geometry. No torch or model execution."""
from collections import Counter, defaultdict
import hashlib
import json
import math

GRID, MIN_PATCHES, MAX_WORDS, MAX_PER_IMAGE = 14, 4, 28, 5


def key(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def identity(value):
    return key(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False))


def patch_mask(box):
    if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
        raise ValueError('Invalid normalized box')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Box outside original image; never clip/expand')
    # Existing cache uses bilinear resize to224 square, with no center crop.
    return [x1 <= (x+.5)/GRID < x2 and y1 <= (y+.5)/GRID < y2
            for y in range(GRID) for x in range(GRID)]


def vocabulary(dataset):
    counts = Counter()
    for row in dataset['images']:
        if row['split'] in ('train', 'restval'):
            for sentence in row['sentences'][:5]:
                counts.update(sentence['raw'].lower().split())
    return {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
            **{word: i for i, (word, _) in enumerate(counts.most_common(9996), 4)}}


def select(rows, word2idx, train_ids, excluded):
    """Same image-first schedule for both arms; no visual-review labels consulted."""
    buckets, rejects, seen = defaultdict(list), Counter(), set()
    total_words = oov_words = 0
    for row in rows:
        cid = row['coco_id']
        if cid not in train_ids or cid in excluded or row['source_split'] not in ('train', 'restval'):
            rejects['excluded_image_or_split'] += 1
            continue
        words = row['phrase'].lower().split()
        total_words += len(words)
        unknown = sum(w not in word2idx for w in words)
        oov_words += unknown
        if not words or len(words) > MAX_WORDS:
            rejects['empty_or_over_28_words'] += 1
            continue
        if unknown:
            rejects['any_oov_token'] += 1
            continue
        mask = patch_mask(row['region_normalized'])
        if sum(mask) < MIN_PATCHES:
            rejects['fewer_than_four_patch_centers'] += 1
            continue
        signature = (cid, tuple(words), tuple(row['region_normalized']))
        if signature in seen:
            rejects['duplicate_phrase_box'] += 1
            continue
        seen.add(signature)
        buckets[cid].append(dict(coco_id=cid, region_id=row['region_id'], phrase=row['phrase'],
            region_normalized=row['region_normalized'], patch_mask=mask,
            encoded=[word2idx['<bos>']] + [word2idx[w] for w in words] + [word2idx['<eos>']]))
    bank = []
    for cid in sorted(buckets):
        bank.extend(sorted(buckets[cid], key=lambda r: key(f"vg-region-cap-v1:{cid}:{r['region_id']}"))[:MAX_PER_IMAGE])
    if not bank:
        raise ValueError('No representable VG phrases with >=4 original-region patch centers')
    report = dict(rejects=dict(rejects), eligible_images=len(buckets), retained_phrases=len(bank),
        total_words_before_text_filter=total_words, oov_words_before_text_filter=oov_words,
        oov_fraction_before_text_filter=oov_words/total_words if total_words else 0.,
        policy='Original text; lowercase/split identical caption tokenizer; zero OOV, <=28 words, BOS/EOS, >=4 patch centers; max5 per image; no nearest fallback')
    return bank, report


def schedule(bank, count=10000):
    by_image = defaultdict(list)
    for index, row in enumerate(bank):
        by_image[row['coco_id']].append(index)
    if not by_image:
        raise ValueError('Empty bank')
    result, cycle = [], 0
    # Each complete cycle visits every eligible image once before repeating any image.
    while len(result) < count:
        images = sorted(by_image, key=lambda cid: key(f'vg-region-image-v1:{cycle}:{cid}'))
        for cid in images:
            result.append(by_image[cid][cycle % len(by_image[cid])])
            if len(result) == count:
                break
        cycle += 1
    return result
