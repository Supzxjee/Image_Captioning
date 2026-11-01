"""Pure NumPy/data contracts; importing this file never imports a model."""
from collections import Counter
import re
import unicodedata

import numpy as np

PAD, BOS, EOS, UNK = range(4)
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']


def roi_pool(patches, objects, width, height, cap=150):
    """Pixel xyxy -> fractional-area-weighted raw patch features, no class inputs."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3 or patches.shape[0] != patches.shape[1]:
        raise ValueError('Expected square patch grid [G,G,C], without CLS')
    if not np.isfinite(patches).all() or not np.isfinite([width, height]).all():
        raise ValueError('Nonfinite patch or dimensions')
    if width <= 0 or height <= 0 or cap < 1 or not isinstance(objects, list):
        raise ValueError('Invalid dimensions, cap, or detector record')
    if not objects:
        return patches.mean(axis=(0, 1), keepdims=False)[None], np.array([[0, 0, width, height]], np.float32), ['full_frame_fallback']
    checked = []
    for index, obj in enumerate(objects):
        box = np.asarray(obj['bbox'], dtype=np.float32)
        score = float(obj['conf'])
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Malformed detection')
        box = np.clip(box, [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Nonpositive detection area; do not turn corruption into empty detection')
        checked.append((-score, index, box))
    selected = sorted(checked, key=lambda item: item[:2])[:cap]
    grid = patches.shape[0]
    axis = np.arange(grid, dtype=np.float32)
    pooled, boxes = [], []
    for _, _, box in selected:
        x0, y0, x1, y1 = box / np.array([width, height, width, height]) * grid
        wx = np.maximum(0, np.minimum(axis + 1, x1) - np.maximum(axis, x0))
        wy = np.maximum(0, np.minimum(axis + 1, y1) - np.maximum(axis, y0))
        weights = (wy[:, None] * wx[None, :]).astype(np.float32)
        pooled.append(np.einsum('ij,ijc->c', weights, patches) / weights.sum())
        boxes.append(box)
    return np.stack(pooled), np.stack(boxes), ['predicted_roi'] * len(pooled)


def train_captions(dataset):
    for image in dataset['images']:
        if image['split'] in ('train', 'restval'):
            for sentence in image['sentences']:
                tokens = sentence['tokens']
                if not isinstance(tokens, list) or not all(isinstance(t, str) and t for t in tokens):
                    raise ValueError('Malformed caption tokens')
                yield [unicodedata.normalize('NFKC', t).lower() for t in tokens]


def vocabulary(dataset, limit=10000):
    if limit < 4:
        raise ValueError('Vocabulary limit includes four specials')
    counts = Counter(t for caption in train_captions(dataset) for t in caption if t not in SPECIALS)
    return SPECIALS + sorted(counts, key=lambda word: (-counts[word], word))[:limit - 4]


def encode(tokens, words, maximum=54):
    if maximum < 2:
        raise ValueError('Sequence must retain BOS/EOS')
    lookup = {word: i for i, word in enumerate(words)}
    content = [lookup.get(unicodedata.normalize('NFKC', t).lower(), UNK) for t in tokens[:maximum - 2]]
    return [BOS] + content + [EOS] + [PAD] * (maximum - len(content) - 2)


def bank_words(tagged_lemmas, minimum=5):
    """Input must come from pinned POS model applied to TRAIN captions only."""
    counts = {'noun': Counter(), 'verb': Counter()}
    for lemma, pos in tagged_lemmas:
        word = unicodedata.normalize('NFKC', lemma).lower()
        kind = 'noun' if pos in ('NOUN', 'PROPN') else 'verb' if pos == 'VERB' else None
        if kind and re.fullmatch('[a-z]+', word):
            counts[kind][word] += 1
    return {kind: [(w, c[w]) for w in sorted(c, key=lambda w: (-c[w], w)) if c[w] >= minimum]
            for kind, c in counts.items()}
