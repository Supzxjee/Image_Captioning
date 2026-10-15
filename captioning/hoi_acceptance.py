"""Selective HOI evaluation: missing evidence abstains; unknown labels stay unknown."""
import torch
from .hoi_transfer_eval import node_key, VERB_NAMES


def score_inventory(rows, prediction, mapping, verbs, correspondence):
    raw = torch.zeros((len(rows), 6))
    native = torch.zeros_like(raw)
    available = torch.zeros_like(raw, dtype=torch.bool)
    reasons = []
    lookup = {(int(s), int(o)): i for i, (s, o) in enumerate(zip(
        prediction['subject_indices'], prediction['object_indices']))}
    official = prediction['official_output']
    scores = {(int(s), int(o), int(v)): float(p) for (s, o), v, p in zip(
        official['pairing'].T, official['labels'], official['scores'])}
    for i, row in enumerate(rows):
        s = mapping.get(node_key(row['subject'], row['subject_box']))
        o = mapping.get(node_key(row['object'], row['object_box']))
        reason = ['missing_instance'] * 6
        if s is not None and o is not None:
            pair = (s['index'], o['index'])
            if pair in lookup:
                for c, name in enumerate(VERB_NAMES):
                    verb = verbs.index(name)
                    if verb in correspondence[int(prediction['labels'][o['index']])]:
                        available[i, c] = True
                        raw[i, c] = prediction['logits'][lookup[pair], verb].sigmoid()
                        native[i, c] = scores[pair[0], pair[1], verb]
                        reason[c] = 'available'
                    else:
                        reason[c] = 'unsupported_object_verb'
            else:
                reason = ['missing_pair'] * 6
        reasons.append(reason)
    return raw, native, available, reasons


def average_precision(scores, labels):
    """Non-interpolated AP, grouped score ties; known labels only."""
    if not (labels == 1).any():
        return None
    order = scores.argsort(descending=True, stable=True)
    scores, labels = scores[order], labels[order]
    ends = torch.cat([scores[1:] != scores[:-1], torch.tensor([True])])
    tp = (labels == 1).cumsum(0)[ends].double()
    n = torch.arange(1, len(labels) + 1)[ends].double()
    increments = tp - torch.cat([torch.zeros(1), tp[:-1]])
    return float(((tp / n) * increments).sum() / (labels == 1).sum())


def acceptance_metrics(probabilities, labels, available, decisions):
    known = labels >= 0
    positive = labels == 1
    accepted = decisions == 1
    tp = int((accepted & positive).sum())
    fp = int((accepted & (labels == 0)).sum())
    ap = [average_precision(probabilities[:, c][known[:, c] & available[:, c]],
                            labels[:, c][known[:, c] & available[:, c]]) for c in range(6)]
    return {'known_cells': int(known.sum()), 'known_positives': int(positive.sum()),
            'available_known_cells': int((known & available).sum()),
            'available_positives': int((positive & available).sum()),
            'accepted_true': tp, 'accepted_false': fp,
            'accepted_unknown': int((accepted & ~known).sum()),
            'known_precision': tp / (tp + fp) if tp + fp else None,
            'full_pool_positive_recall': tp / int(positive.sum()) if positive.any() else None,
            'unavailable_known': int((known & ~available).sum()),
            'abstained_known': int((known & (decisions == -1)).sum()),
            'ap_by_predicate': ap,
            'macro_ap_available_known': sum(p for p in ap if p is not None) / sum(p is not None for p in ap)
                if any(p is not None for p in ap) else None}
