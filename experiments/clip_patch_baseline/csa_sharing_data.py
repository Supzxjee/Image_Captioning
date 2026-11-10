"""NumPy-only local deduplication and matched TRAIN benchmark orders."""
import numpy as np


def image_plan(caption_image, events, shared):
    events = np.asarray(events)
    if events.ndim != 1 or not len(events) or events.dtype.kind not in 'iu' or (events < 0).any() or (events >= len(caption_image)).any():
        raise ValueError('Expected nonempty valid integer caption event IDs')
    images = np.asarray(caption_image)[events]
    if not shared:
        return np.arange(len(events), dtype=np.int64), None
    _, first, inverse = np.unique(images, return_index=True, return_inverse=True)
    if not np.array_equal(images[first][inverse], images):
        raise ValueError('Sharing inverse does not reconstruct caption image order')
    return first.astype(np.int64), inverse.astype(np.int64)


def benchmark_orders(caption_image):
    images = np.asarray(caption_image)
    if images.ndim != 1 or len(images) != 1280 or len(np.unique(images)) != 256:
        raise ValueError('Expected complete TRAIN256/1280 caption pool')
    # Stable grouping works without assuming caption rows were originally sorted.
    by_image = np.argsort(images, kind='stable')
    ordered = images[by_image]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, len(images)]
    rng = np.random.default_rng(42)
    grouped = np.concatenate([by_image[boundaries[i]:boundaries[i+1]] for i in rng.permutation(256)])
    random = np.random.default_rng(42).permutation(len(images))
    return dict(random=random, grouped=grouped)


def local_events(events, rank):
    if rank not in (0, 1) or not 17 <= len(events) <= 32:
        raise ValueError('Expected global17..32 captions on two ranks')
    return np.asarray(events[rank*16:(rank+1)*16], dtype=np.int64)
