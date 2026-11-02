"""Narrow, evidence-backed metadata rule; no model imports."""
from collections import Counter


def corrected_lemma(surface, pos, original_lemma):
    if pos == 'VERB' and surface.lower() in ('riding', 'rides') and original_lemma.lower() == 'rid':
        return 'ride'
    return original_lemma


def corrected_bank(original, moved_count):
    if not isinstance(moved_count, int) or moved_count <= 0 or original['minimum_count'] != 5:
        raise ValueError('Positive measured count and original min5 policy required')
    before = Counter({row['word']: row['count'] for row in original['verb']})
    if before['rid'] < moved_count or not before['ride']:
        raise ValueError('Correction exceeds baseline counts or baseline ride absent')
    after = before.copy()
    after['rid'] -= moved_count
    after['ride'] += moved_count
    if sum(after.values()) != sum(before.values()):
        raise ValueError('Count mass changed')
    result = dict(original)
    result['verb'] = [dict(word=w, count=after[w]) for w in sorted(after, key=lambda w: (-after[w], w)) if after[w] >= 5]
    result['lemma_amendment'] = dict(version='verb_riding_rides_rid_to_ride_v1',
        surface_forms=['riding', 'rides'], pos='VERB', original_lemma='rid', corrected_lemma='ride',
        moved_tokens=moved_count, other_verbs_unchanged=True, nouns_unchanged=True)
    return result, dict(before_rid=before['rid'], before_ride=before['ride'],
        after_rid=after['rid'], after_ride=after['ride'], moved_tokens=moved_count,
        below_minimum_omitted_mass=sum(n for n in after.values() if n < 5))
