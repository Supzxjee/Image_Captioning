"""Evaluate explicit reviewed target substitutions without relabeling unknowns."""
from collections import defaultdict
import torch
from .supervised_relations import RELATIONS


def validate_target_cases(rows, cases):
    seen, claim_labels = set(), {}
    for case in cases:
        i, j, c = case['positive_row_index'], case['alternate_row_index'], case['relation_index']
        if not (0 <= i < len(rows) and 0 <= j < len(rows) and 0 <= c < len(RELATIONS)):
            raise ValueError('Invalid target row/predicate index.')
        a, b = rows[i], rows[j]
        if (a['image_id'], a['filename'], a['subject_id'], a['subject_box']) != (b['image_id'], b['filename'], b['subject_id'], b['subject_box']):
            raise ValueError('Target substitution must keep the same image and subject instance.')
        if a['object_id'] == b['object_id'] or a['object_box'] == b['object_box']:
            raise ValueError('Target substitution must change the localized target.')
        if a['labels'][c] != 1 or b['labels'][c] != -1:
            raise ValueError('Expected source positive and originally unknown alternate.')
        if (i, j, c) in seen:
            raise ValueError('Duplicate reviewed target contrast.')
        seen.add((i, j, c))
        for index, label in ((i, 1), (j, 0)):
            key = index, c
            if key in claim_labels and claim_labels[key] != label:
                raise ValueError('Contradictory human labels across eligible cases.')
            claim_labels[key] = label
    if not seen:
        raise ValueError('No eligible reviewed target contrasts.')
    return claim_labels


def target_metrics(scores, decisions, rows, cases, selected=None):
    """Report ranking AND acceptance; abstaining on everything is not success."""
    truth = validate_target_cases(rows, cases)
    if scores.shape != decisions.shape or scores.shape != (len(rows), len(RELATIONS)) or not torch.isfinite(scores).all():
        raise ValueError('Target prediction coverage differs.')
    details = []
    for case in cases:
        i, j, c = case['positive_row_index'], case['alternate_row_index'], case['relation_index']
        gap = float(scores[i, c] - scores[j, c])
        item = {'case_id': case['case_id'], 'image_id': rows[i]['image_id'], 'predicate': RELATIONS[c],
                'same_class': rows[i]['object'] == rows[j]['object'], 'gap': gap,
                'positive_decision': int(decisions[i, c]), 'alternate_decision': int(decisions[j, c])}
        if selected is not None:
            item.update(positive_selected=bool(selected[i, c]), alternate_selected=bool(selected[j, c]))
        details.append(item)

    def aggregate(items):
        images = defaultdict(list)
        for item in items:
            images[item['image_id']].append(int(item['gap'] > 0))
        return {'cases': len(items), 'images': len(images), 'ranking_wins': sum(x['gap'] > 0 for x in items),
                'ranking_ties': sum(x['gap'] == 0 for x in items),
                'ranking_win_rate': sum(x['gap'] > 0 for x in items) / len(items) if items else None,
                'image_macro_win_rate': sum(sum(v)/len(v) for v in images.values()) / len(images) if images else None,
                'positive_supported': sum(x['positive_decision'] == 1 for x in items),
                'wrong_target_supported': sum(x['alternate_decision'] == 1 for x in items),
                'both_supported': sum(x['positive_decision'] == x['alternate_decision'] == 1 for x in items),
                'both_abstained': sum(x['positive_decision'] == x['alternate_decision'] == -1 for x in items),
                'positive_supported_alternate_not_supported': sum(x['positive_decision'] == 1 and x['alternate_decision'] != 1 for x in items),
                'positive_supported_alternate_contradicted': sum(x['positive_decision'] == 1 and x['alternate_decision'] == 0 for x in items)}

    def selected_claims(mask):
        tp = sum(int(bool(mask[i, c])) for (i, c), label in truth.items() if label == 1)
        fp = sum(int(bool(mask[i, c])) for (i, c), label in truth.items() if label == 0)
        positives = sum(label == 1 for label in truth.values())
        return {'unique_known_claims': len(truth), 'unique_known_positive_claims': positives,
                'true_selected': tp, 'false_selected': fp,
                'precision_on_reviewed': tp / (tp + fp) if tp + fp else None,
                'recall_on_reviewed': tp / positives if positives else None}

    result = {'all': aggregate(details),
              'same_class': aggregate([x for x in details if x['same_class']]),
              'different_class': aggregate([x for x in details if not x['same_class']]),
              'by_predicate': {r: aggregate([x for x in details if x['predicate'] == r]) for r in RELATIONS},
              'unique_claims_before_cap': selected_claims(decisions == 1)}
    if selected is not None:
        if selected.shape != decisions.shape or (selected & (decisions != 1)).any():
            raise ValueError('Cap cannot select a rejected/abstained claim.')
        result['unique_claims_after_cap'] = selected_claims(selected)
        result['case_counts_after_cap'] = {
            'positive_selected': sum(x['positive_selected'] for x in details),
            'wrong_target_selected': sum(x['alternate_selected'] for x in details),
            'positive_selected_wrong_target_not_selected': sum(x['positive_selected'] and not x['alternate_selected'] for x in details)}
    return result, details


def target_changes(reference, learned):
    """Paired changes, retaining image counts so correlated predicates are visible."""
    if len({r['case_id'] for r in reference}) != len(reference) or len({r['case_id'] for r in learned}) != len(learned):
        raise ValueError('Duplicate case predictions.')
    before, after = {r['case_id']: r for r in reference}, {r['case_id']: r for r in learned}
    if before.keys() != after.keys():
        raise ValueError('Different target cases in paired comparison.')
    changes = []
    for key, old in before.items():
        new = after[key]
        if any(old[k] != new[k] for k in ('image_id', 'predicate', 'same_class')):
            raise ValueError('Paired target identity differs.')
        category = ('both_correct' if old['gap'] > 0 and new['gap'] > 0 else
                    'regressed' if old['gap'] > 0 else 'fixed' if new['gap'] > 0 else 'both_not_correct')
        changes.append({'case_id': key, 'image_id': old['image_id'], 'predicate': old['predicate'],
                        'same_class': old['same_class'], 'category': category,
                        'prior_gap': old['gap'], 'learned_gap': new['gap'],
                        'gap_change': new['gap'] - old['gap']})
    counts = {category: {'cases': sum(x['category'] == category for x in changes),
                        'images': len({x['image_id'] for x in changes if x['category'] == category})}
              for category in ('both_correct', 'regressed', 'fixed', 'both_not_correct')}
    return counts, changes
