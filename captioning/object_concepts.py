"""Validation and filename alignment for per-object CLIP text embeddings."""
from pathlib import PurePosixPath


def align_object_concept_cache(cache, image_paths):
    if not isinstance(cache, dict):
        raise ValueError('Object concept cache must be a mapping.')
    indexed = {}
    for key, entry in cache.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous object concept filename: {filename}')
        if not isinstance(entry, dict) or 'tokens' not in entry or 'mask' not in entry:
            raise ValueError(f'Invalid object concept entry for {filename}.')
        tokens, mask = entry['tokens'], entry['mask']
        if tokens.ndim != 2 or tokens.size(1) != 512:
            raise ValueError(f'{filename}: expected object tokens (O, 512), got {tokens.shape}')
        if mask.ndim != 1 or mask.size(0) != tokens.size(0):
            raise ValueError(f'{filename}: object mask does not match tokens.')
        indexed[filename] = entry
    aligned = {}
    for image_path in image_paths:
        filename = PurePosixPath(str(image_path).replace('\\', '/')).name
        if filename in indexed:
            aligned[image_path] = indexed[filename]
    return aligned


def build_object_concept_index(source, records, objects_field='objects', name_key='label',
                               confidence_key='conf', min_confidence=0.5, max_objects=10):
    """Select unique detector labels by confidence without consulting reference captions."""
    if not isinstance(source, dict):
        raise ValueError('Detection source must be a filename-to-entry mapping.')
    indexed = {}
    for key, entry in source.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous detection filename: {filename}')
        indexed[filename] = entry
    result = {}
    for record in records:
        filename = record['filename']
        if filename not in indexed:
            raise ValueError(f'Missing detections for {filename}')
        entry = indexed[filename]
        objects = entry if isinstance(entry, list) else entry.get(objects_field)
        if not isinstance(objects, list):
            raise ValueError(f'{filename}: expected a list in {objects_field!r}.')
        accepted = {}
        for detection in objects:
            if isinstance(detection, str):
                label, confidence = detection, 1.0
            elif isinstance(detection, dict):
                label = detection.get(name_key)
                confidence = detection.get(confidence_key)
            else:
                continue
            if not isinstance(label, str) or not isinstance(confidence, (int, float)):
                continue
            label = ' '.join(label.lower().strip().split())
            confidence = float(confidence)
            if label and confidence >= min_confidence:
                accepted[label] = max(accepted.get(label, 0.0), confidence)
        selected = sorted(accepted.items(), key=lambda item: (-item[1], item[0]))[:max_objects]
        result[filename] = selected
    return result
