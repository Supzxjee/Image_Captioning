"""Separate actor/action support from conditional target attachment.

Target labels must be explicit confirmed instance labels. Other targets are
unknown, not softmax negatives. Multiple true targets may coexist.
"""
import math
import random
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from .reviewed_targets import validate_target_cases


def actor_examples(rows, views):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['image_id'], row['subject_id'])].append(i)
    features, labels, keys, row_to_actor = [], [], [], torch.empty(len(rows), dtype=torch.long)
    for key, indices in sorted(groups.items()):
        group = torch.tensor([rows[i]['labels'] for i in indices])
        if ((group == 0).any(0) & (group == 1).any(0)).any():
            raise ValueError('Source actor labels conflict; do not use reviewed target-zero labels here.')
        label = torch.full((group.shape[1],), -1, dtype=torch.long)
        label[(group == 0).any(0)] = 0
        label[(group == 1).any(0)] = 1
        # Only subject crops; no target/union evidence can change the action score.
        features.append(F.normalize(views[indices, 0].detach().mean(0), dim=-1))
        labels.append(label); keys.append(key); row_to_actor[indices] = len(keys) - 1
    return torch.stack(features), torch.stack(labels), keys, row_to_actor


class ActionSupportHead(nn.Module):
    def __init__(self, dim=512, predicates=6):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, predicates))

    def forward(self, subject):
        return self.network(subject.detach())


def target_features(views, geometry, text):
    if views.ndim != 3 or views.shape[1] != 3 or geometry.shape != (len(views), 10) or text.shape[0] != len(views) or text.shape[2] != views.shape[2]:
        raise ValueError('Invalid target evidence dimensions.')
    if not all(torch.isfinite(x).all() for x in (views, geometry, text)):
        raise ValueError('Nonfinite target evidence.')
    a, b, union = F.normalize(views.detach(), dim=-1).unbind(1)
    words = F.normalize(text.detach(), dim=-1)
    similarity = torch.stack([torch.einsum('bd,bkd->bk', v, words) for v in (a, b, union)], -1)
    # Geometry is supplied evidence, not a rule assigning action labels.
    boxes = geometry[:, :8].reshape(-1, 2, 4)
    size = boxes[:, :, 2:] - boxes[:, :, :2]
    if not (size > 0).all():
        raise ValueError('Empty target/subject box.')
    intersection = (torch.minimum(boxes[:, 0, 2:], boxes[:, 1, 2:]) -
                    torch.maximum(boxes[:, 0, :2], boxes[:, 1, :2])).clamp_min(0).prod(-1)
    areas = size.prod(-1)
    iou = intersection / (areas.sum(-1) - intersection).clamp_min(1e-8)
    log_ratio = (areas[:, 1] / areas[:, 0]).log().clamp(-5, 5) / 5
    shared = torch.cat([geometry.detach(), size.flatten(1), iou[:, None], log_ratio[:, None], (a*b).sum(-1, keepdim=True)], -1)
    return torch.cat([similarity, shared[:, None].expand(-1, words.shape[1], -1)], -1)


class ConditionalTargetHead(nn.Module):
    """126 parameters; bounded, zero-initialized correction to the frozen prior."""
    def __init__(self, predicates=6, correction_bound=.5):
        super().__init__()
        if not 0 < correction_bound <= 1:
            raise ValueError('Invalid correction bound.')
        self.bound = correction_bound
        self.weight = nn.Parameter(torch.zeros(predicates, 20))
        self.bias = nn.Parameter(torch.zeros(predicates))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != self.weight.shape or not torch.isfinite(features).all():
            raise ValueError('Unexpected conditional-target features.')
        prior = 10 * features[:, :, 2] - 3
        correction = self.bound * torch.tanh(torch.einsum('brd,rd->br', features.detach(), self.weight) + self.bias)
        return prior + correction, prior, correction


def reviewed_labels(rows, cases):
    truth = validate_target_cases(rows, cases)
    result = torch.full((len(rows), 6), -1, dtype=torch.long)
    for (i, c), value in truth.items():
        result[i, c] = value
    return result


def target_objective(logits, rows, cases, ranking_weight=.2):
    """Equal image weight; source/unknown targets never enter this loss.

    BCE gives absolute target supervision; logistic contrasts compare only
    reviewed positive/negative endpoints of the SAME image/subject/predicate.
    No constraint makes the supported targets compete with one another.
    """
    labels = reviewed_labels(rows, cases)
    groups = defaultdict(list)
    for case in cases:
        groups[rows[case['positive_row_index']]['image_id']].append(case)
    classifications, contrasts = [], []
    for items in groups.values():
        endpoints = {(case[key], case['relation_index']) for case in items
                     for key in ('positive_row_index', 'alternate_row_index')}
        ii, cc = zip(*sorted(endpoints))
        classifications.append(F.binary_cross_entropy_with_logits(logits[list(ii), list(cc)], labels[list(ii), list(cc)].float()))
        gaps = torch.stack([logits[c['positive_row_index'], c['relation_index']] -
                            logits[c['alternate_row_index'], c['relation_index']] for c in items])
        contrasts.append(F.softplus(-gaps).mean())
    bce, rank = torch.stack(classifications).mean(), torch.stack(contrasts).mean()
    return bce + ranking_weight * rank, bce, rank


def image_folds(rows, cases, count=5, seed=42):
    validate_target_cases(rows, cases)
    images = sorted({rows[c['positive_row_index']]['image_id'] for c in cases})
    if count < 2 or len(images) < count:
        raise ValueError('Insufficient reviewed images for cross-validation.')
    random.Random(seed).shuffle(images)
    folds = []
    for index in range(count):
        test = set(images[index::count])
        remaining = sorted(set(images) - test)
        random.Random(seed + 1000 + index).shuffle(remaining)
        calibration = set(remaining[:max(2, math.ceil(len(remaining) / 4))])
        train = set(remaining) - calibration
        if not train or train & calibration or (train | calibration) & test:
            raise ValueError('Overlapping/empty nested image partitions.')
        folds.append({'fold': index, 'fit_images': sorted(train), 'calibration_images': sorted(calibration), 'evaluation_images': sorted(test)})
    return folds


def combine_decisions(action, target):
    if action.shape != target.shape or not (((action == -1) | (action == 0) | (action == 1)).all() and
                                            ((target == -1) | (target == 0) | (target == 1)).all()):
        raise ValueError('Invalid factorized decisions.')
    result = torch.full_like(action, -1)
    result[(action == 0) | (target == 0)] = 0
    result[(action == 1) & (target == 1)] = 1
    return result


def relative_target_prior(prior, rows):
    """Subtract the same-actor/predicate mean of unique detector-target priors.

    No label or annotation ID is consulted. This is a frozen score reference,
    not softmax exclusivity or an assumption that other targets are false.
    Learned corrections are added AFTER centering, so unknown targets acquire
    no classification gradient through the reference.
    """
    if prior.shape != (len(rows), 6) or not torch.isfinite(prior).all():
        raise ValueError('Invalid target prior coverage.')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['filename'], row['subject'], tuple(row['subject_box']))].append(i)
    result = torch.empty_like(prior)
    for indices in groups.values():
        unique = {}
        for i in indices:
            unique.setdefault((rows[i]['object'], tuple(rows[i]['object_box'])), i)
        reference = prior[list(unique.values())].detach().mean(0)
        result[indices] = prior[indices].detach() - reference
    return result
