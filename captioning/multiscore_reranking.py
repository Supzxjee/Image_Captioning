"""Candidate re-ranking with decoder, CLIP image-text and object scores."""
from .object_consistency import object_consistency


def _minmax(values):
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def rerank_multiscore(record, detected, clip_weight, object_weight,
                      hallucination_penalty=1.0):
    """Return the best candidate under weights that sum with decoder to one."""
    if clip_weight < 0 or object_weight < 0 or clip_weight + object_weight > 1:
        raise ValueError('Require nonnegative CLIP/OCC weights with sum <= 1.')
    candidates = record.get('candidates', [])
    if not candidates:
        raise ValueError(f'Image {record.get("image_id")}: no candidates')
    if any('clipscore' not in candidate for candidate in candidates):
        raise ValueError(f'Image {record.get("image_id")}: candidate lacks clipscore')

    # Beam candidates are ordered by the accumulated raw log-probability in
    # inference.py. Use that same score so auxiliary weights of zero reproduce
    # rank 0 exactly; avg_logprob would silently introduce length normalization.
    language = [float(candidate['logprob']) for candidate in candidates]
    clip = [float(candidate['clipscore']) for candidate in candidates]
    language_scores, clip_scores = _minmax(language), _minmax(clip)
    decoder_weight = 1.0 - clip_weight - object_weight
    enriched = []
    for rank, candidate in enumerate(candidates):
        consistency = object_consistency(
            candidate['caption'], detected, hallucination_penalty)
        object_score = ((consistency['score'] + hallucination_penalty) /
                        (1.0 + hallucination_penalty))
        combined = (decoder_weight * language_scores[rank] +
                    clip_weight * clip_scores[rank] +
                    object_weight * object_score)
        enriched.append({
            **candidate,
            'original_rank': rank,
            'decoder_score': float(language_scores[rank]),
            'clip_score_normalized': float(clip_scores[rank]),
            'object_score': float(object_score),
            'combined_score': float(combined),
            'object_consistency': consistency,
        })
    selected = max(enriched, key=lambda item: (item['combined_score'],
                                                -item['original_rank']))
    return selected, enriched
