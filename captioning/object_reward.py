"""Training-only COCO object reward, independent of caption evaluation tools.

Unsupported means absent from the COCO annotation support set, not proven
physical absence. A high-confidence detector can make an unsupported label
neutral, never establish a positive training label.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle
import re
import numpy as np
from .object_consistency import COCO_LABELS, extract_caption_objects, index_detections


class _CacheRecord:
    pass


class _AnnotationReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('chair', '__main__') and name == 'CHAIR': return _CacheRecord
        if module == 'collections' and name == 'defaultdict': return defaultdict
        if module == 'builtins' and name in ('set', 'list', 'dict', 'tuple'):
            import builtins
            return getattr(builtins, name)
        raise ValueError(f'Unexpected annotation-cache type: {module}.{name}')


CACHE_SHA256 = '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'


@lru_cache(maxsize=32768)
def recognized_objects(text, parser_version='legacy'):
    if parser_version not in ('legacy', 'aliases_v2'): raise ValueError('Unknown object parser version.')
    labels = set(extract_caption_objects(text))
    if parser_version == 'aliases_v2':
        if re.search(r'\bdesks?\b', text.lower()): labels.add('dining table')
        if re.search(r'\bturkeys?\b', text.lower()): labels.add('bird')
    # Do not copy context-ambiguous mother/baby => person mappings from CHAIR.
    return frozenset(labels)


def make_targets(groups, dataset, cache_path, detections, parser_version='legacy'):
    """Filter BEFORE accessing labels; never export held-out image annotations."""
    cache_path = Path(cache_path)
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != CACHE_SHA256:
        raise ValueError('Object annotation cache differs from the inspected pinned source.')
    with cache_path.open('rb') as stream: source = _AnnotationReader(stream).load()
    selected = groups['train']; heldout = {r['coco_id'] for r in groups['val']}
    if heldout & {r['coco_id'] for r in selected}: raise ValueError('Training/validation overlap.')
    detector = index_detections(detections, min_confidence=.9)
    rows = []; empty_annotations = 0
    for index, r in enumerate(selected, 1):
        original = dataset['images'][r['image_id']]
        if original['split'] not in ('train', 'restval') or original['filename'] != r['filename']:
            raise ValueError('Object reward attempted to use a non-training or mismatched image.')
        if r['coco_id'] not in source.imid_to_objects: raise ValueError('Missing train annotation.')
        if r['filename'] not in detector: raise ValueError('Missing training detector entry.')
        support = set(source.imid_to_objects[r['coco_id']])
        if not support <= set(COCO_LABELS): raise ValueError('Invalid annotation support labels.')
        empty_annotations += not bool(support)
        refs = original['sentences'][:5]
        if len(refs) != 5: raise ValueError('Need five training references.')
        salience = {label for ref in refs for label in recognized_objects(ref['raw'], parser_version)}
        # Both parsers can disagree on aliases. A training reference is positive
        # evidence, never a negative label merely because another parser missed it.
        support |= salience
        unknown = set(detector[r['filename']])-support
        rows.append({**r, 'supported': sorted(support), 'salient': sorted(salience),
                     'detector_neutral': sorted(unknown)})
        if index % 1000 == 0 or index == len(selected):
            print(f'Training-only object targets: {index}/{len(selected)}', flush=True)
    return {'metadata': {'variant': 'coco_train_object_reward_v1', 'images': len(rows),
            'parser_version': parser_version,
            'empty_annotation_sets_before_reference_union': empty_annotations,
            'reward_eligible_images': sum(bool(r['supported']) for r in rows),
            'annotation_cache_sha256': CACHE_SHA256,
            'support_source': 'Pinned COCO instance + reference-caption annotation union',
            'salience_source': 'Only five selected training-image references; positive labels only',
            'detector_policy': 'confidence>=.9 absent-from-annotation classes neutral, not rewarded or penalized',
            'formula': '.5 * salient_unique_recall - unsupported_unique_fraction',
            'outside_80_classes': 'neutral', 'heldout_reward_labels_exported': False}, 'data': rows}


class TrainingObjectReward:
    def __init__(self, rows, parser_version='legacy'):
        if parser_version not in ('legacy', 'aliases_v2'): raise ValueError('Unknown object parser version.')
        self.parser_version = parser_version
        self.rows = {r['image_id']: r for r in rows}
        if len(self.rows) != len(rows): raise ValueError('Duplicate object reward image IDs.')
        for r in rows:
            if not set(r['salient']) <= set(r['supported']): raise ValueError('Salience must be supported.')
            if set(r['detector_neutral']) & set(r['supported']): raise ValueError('Conflicting object states.')
            if not (set(r['supported']) | set(r['detector_neutral'])) <= set(COCO_LABELS):
                raise ValueError('Unsupported annotation vocabulary.')

    def score(self, ids, captions):
        if len(ids) != len(captions): raise ValueError('Caption/ID count mismatch.')
        values, diagnostics = [], []
        for image_id, text in zip(ids, captions):
            r = self.rows[image_id]
            predicted = set(recognized_objects(text, self.parser_version))
            support, salient, neutral = (set(r[k]) for k in ('supported', 'salient', 'detector_neutral'))
            if not support:
                values.append(0.)
                diagnostics.append({'available': False, 'salient_recall': 0., 'unsupported_fraction': 0.,
                    'unsupported_classes': [], 'neutral_classes': sorted(predicted), 'recognized_classes': sorted(predicted)})
                continue
            unsupported = predicted-support-neutral
            # Neutral labels cannot dilute the unsupported fraction.
            judged = predicted-neutral
            penalty = len(unsupported)/max(len(judged), 1)
            coverage = len(predicted & salient)/len(salient) if salient else 0.
            value = .5*coverage-penalty
            values.append(value)
            diagnostics.append({'available': True, 'salient_recall': coverage, 'unsupported_fraction': penalty,
                                'unsupported_classes': sorted(unsupported),
                                'neutral_classes': sorted(predicted & neutral),
                                'recognized_classes': sorted(predicted)})
        return np.asarray(values, dtype=np.float32), diagnostics


def combine_rewards(cider, objects, weight):
    cider, objects = np.asarray(cider), np.asarray(objects)
    if cider.shape != objects.shape or not np.isfinite(weight) or weight < 0:
        raise ValueError('Invalid reward shapes or coefficient.')
    return cider+weight*objects
