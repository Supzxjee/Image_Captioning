"""Fixed cohorts and matched decoder adaptation parameters for a small pilot."""
import random
from pathlib import PurePosixPath
import torch


def filename_index(cache):
    result = {}
    for key, value in cache.items():
        name = PurePosixPath(str(key).replace('\\', '/')).name
        if name in result:
            raise ValueError(f'Ambiguous filename: {name}')
        result[name] = value
    return result


def action_entry(entry):
    copied = dict(entry)
    mask = torch.zeros_like(entry['mask'])
    for slot, claim in enumerate(entry.get('relations', [])):
        if slot < mask.numel() and claim.get('relation') in {'holding', 'riding', 'carrying', 'wearing'}:
            mask[slot] = entry['mask'][slot]
    copied['mask'] = mask
    return copied


def choose_training(records, prompts, relations, excluded, count=512, seed=42):
    candidates = []
    for record in records:
        name = record['filename']
        if (record['split'] in {'train', 'restval'} and name not in excluded and
                name in prompts and name in relations and record.get('sentences') and
                bool(action_entry(relations[name])['mask'].any())):
            candidates.append(record)
    candidates.sort(key=lambda r: r['filename'])
    if len({r['filename'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate training filenames')
    if len(candidates) < count:
        raise ValueError(f'Need {count} eligible training images, found {len(candidates)}')
    return random.Random(seed).sample(candidates, count)


def adapt_parameters(model):
    """Freeze encoder and decoder language path; adapt cross-attention and norm2."""
    model.requires_grad_(False)
    parameters = []
    names = []
    for name, parameter in model.decoder.named_parameters():
        if '.multihead_attn.' in name or '.norm2.' in name:
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(name)
    if not parameters:
        raise ValueError('No decoder cross-attention parameters found')
    return parameters, names
