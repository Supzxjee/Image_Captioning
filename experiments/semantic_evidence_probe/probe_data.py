"""Verified pair construction and tie-aware summaries; no model or training imports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bounds(subject, obj):
    s, o = np.asarray(subject, dtype=np.float32), np.asarray(obj, dtype=np.float32)
    for value in (s, o):
        if value.shape != (4,) or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any() or (value[2:] <= value[:2]).any():
            raise ValueError('Invalid normalized xyxy box')
    return np.stack((s, o, np.r_[np.minimum(s[:2], o[:2]), np.maximum(s[2:], o[2:])])).tolist()


def human_pairs(cases, annotations, excluded, captions, relation_text):
    result = []
    for case in cases['cases']:
        if case['relation'] not in ('holding', 'riding'):
            continue
        a = annotations['annotations'].get(case['case_id'], {})
        if case['filename'] in excluded or captions[case['filename']]['split'] not in ('train', 'restval'):
            continue
        if any(a.get(key) != 'clear' for key in ('subject_box_status', 'A_box_status', 'B_box_status')):
            continue
        status = (a.get('A_status'), a.get('B_status'))
        if status not in (('supported', 'contradicted'), ('contradicted', 'supported')):
            continue
        targets = case['targets']
        if targets[0]['label'] != targets[1]['label']:
            continue
        candidates = [dict(text=relation_text(case['relation'], t['label']), bounds=bounds(case['subject_box'], t['bbox'])) for t in targets]
        result.append(dict(case_id=case['case_id'], filename=case['filename'], task='wrong_object_instance',
                           predicate=case['relation'], object_label=targets[0]['label'], candidates=candidates,
                           positive_index=status.index('supported'), label_source='human_explicit_supported_vs_contradicted'))
    return result


def vcoco_pairs(records, excluded, captions, relation_text, per_predicate=50):
    candidates = defaultdict(list)
    # Restrict verb swaps to classes plausibly associated with riding; no phone-vs-horse shortcut.
    ride_objects = {'bicycle', 'motorcycle', 'horse', 'elephant', 'skateboard', 'surfboard', 'snowboard', 'skis', 'boat', 'car', 'bus', 'truck', 'train'}
    for image in records:
        if image['filename'] in excluded or captions[image['filename']]['split'] not in ('train', 'restval'):
            continue
        objects = {r['instance_id']: r for r in image['instances']}
        labels = {(r['subject_id'], r['predicate']): r['label'] for r in image['actor_action_labels']}
        for edge in image['relations']:
            sid, oid, verb = edge['subject_id'], edge['object_id'], edge['predicate']
            if verb not in ('holding', 'riding'):
                continue
            subject, obj = objects[sid], objects[oid]
            assert labels.get((sid, verb)) == 1 and subject['category'] == 'person'
            b = bounds(subject['box'], obj['box'])
            positive = dict(text=relation_text(verb, obj['category']), bounds=b)
            alternate = 'riding' if verb == 'holding' else 'holding'
            common = dict(filename=image['filename'], predicate=verb, object_label=obj['category'], label_source='vcoco_positive_role_and_explicit_actor_zero')
            if obj['category'] in ride_objects and labels.get((sid, alternate)) == 0:
                candidates[('wrong_predicate', verb)].append(dict(common, case_id=f'vcoco_verb_{image["image_id"]}_{sid}_{oid}_{verb}',
                    task='wrong_predicate', candidates=[positive, dict(text=relation_text(alternate, obj['category']), bounds=b)], positive_index=0))
            for other in objects.values():
                if other['instance_id'] == sid or other['category'] != 'person' or labels.get((other['instance_id'], verb)) != 0:
                    continue
                candidates[('wrong_subject_instance', verb)].append(dict(common,
                    case_id=f'vcoco_actor_{image["image_id"]}_{sid}_{oid}_{other["instance_id"]}_{verb}', task='wrong_subject_instance',
                    candidates=[positive, dict(text=positive['text'], bounds=bounds(other['box'], obj['box']))], positive_index=0))
    result = []
    for task in ('wrong_predicate', 'wrong_subject_instance'):
        groups = []
        for verb in ('holding', 'riding'):
            # One pair per image/predicate; selection depends only on stable IDs, never model scores.
            ordered = sorted(candidates[(task, verb)], key=lambda c: identity(c['case_id']))
            seen, selected = set(), []
            for row in ordered:
                if row['filename'] not in seen:
                    seen.add(row['filename']); selected.append(row)
            groups.append(selected)
        if task == 'wrong_predicate':
            # Balance positive verbs WITHIN each object class, before seeing any model score.
            remaining = per_predicate
            for label in sorted(ride_objects):
                per_class = [[r for r in g if r['object_label'] == label] for g in groups]
                count = min(remaining, *(len(g) for g in per_class))
                result.extend(row for group in per_class for row in group[:count])
                remaining -= count
        else:
            count = min(per_predicate, *(len(g) for g in groups))
            result.extend(row for group in groups for row in group[:count])
    return result


def finalize(rows, captions):
    result = []
    for original in rows:
        row = dict(original)
        row['coco_id'] = int(Path(row['filename']).stem.rsplit('_', 1)[1])
        row['caption_split'] = captions[row['filename']]['split']
        if int(identity(row['case_id'])[:2], 16) % 2:
            row['candidates'] = row['candidates'][::-1]
            row['positive_index'] = 1-row['positive_index']
        result.append(row)
    assert len({r['case_id'] for r in result}) == len(result)
    return sorted(result, key=lambda r: (r['task'], r['case_id']))


def summarize(rows):
    output = {}
    for task in sorted({r['task'] for r in rows}):
        selected = [r for r in rows if r['task'] == task]
        methods = {}
        for method in ('boxed', 'global_view', 'zero_visual'):
            by_image = defaultdict(list)
            margins, ties = [], 0
            for row in selected:
                positive = row['positive_index']
                scores = row['scores'][method]
                if len(scores) != 2 or not all(math.isfinite(s) for s in scores):
                    raise ValueError('Nonfinite or incomplete scores')
                margin = scores[positive]-scores[1-positive]
                tie = abs(margin) <= 1e-7
                margins.append(margin); ties += tie
                by_image[row['coco_id']].append(.5 if tie else float(margin > 0))
            values = np.asarray([np.mean(v) for v in by_image.values()])
            rng = np.random.default_rng(20261019)
            intervals = np.percentile([values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)], [2.5, 97.5])
            methods[method] = dict(image_macro_pair_accuracy=float(values.mean()), image_cluster_bootstrap95=intervals.tolist(),
                                   pair_accuracy=float(sum(sum(v) for v in by_image.values())/len(selected)),
                                   ties=ties, mean_cosine_margin=float(np.mean(margins)))
        output[task] = dict(pairs=len(selected), images=len({r['coco_id'] for r in selected}),
                            predicates=dict(Counter(r['predicate'] for r in selected)), methods=methods)
    return output
