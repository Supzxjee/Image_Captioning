"""Objects-only text from saved detector output, never from caption references."""
from pathlib import PurePosixPath


def object_names(entry, objects_field='objects', name_key=''):
    objects = entry if isinstance(entry, list) else entry.get(objects_field)
    if not isinstance(objects, list):
        raise ValueError(f'Expected an object list in field {objects_field!r}; '
                         'embedding tokens or a full prompt cannot recover original detections.')
    names = []
    for obj in objects:
        if isinstance(obj, str):
            name = obj
        elif isinstance(obj, dict):
            if name_key:
                name = obj.get(name_key)
            else:
                labels = [obj[k] for k in ('name', 'label', 'class_name') if k in obj]
                if not labels or any(value != labels[0] for value in labels):
                    raise ValueError('Missing/ambiguous detector label; set --name-key explicitly.')
                name = labels[0]
        else:
            raise ValueError('Objects must be class-name strings or detection dictionaries.')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Detector class name must be a nonempty string, not a numeric class ID.')
        name = ' '.join(name.lower().strip().split())
        if name not in names:
            names.append(name)
    return names


def build_object_texts(source, records, objects_field='objects', name_key=''):
    if not isinstance(source, dict):
        raise ValueError('Source must map image paths/filenames to saved object entries.')
    indexed = {}
    for key, entry in source.items():
        name = PurePosixPath(str(key).replace('\\', '/')).name
        if name in indexed:
            raise ValueError(f'Ambiguous source filename: {name}')
        indexed[name] = entry
    texts = {}
    for record in records:
        name = record['filename']
        if name not in indexed:
            raise ValueError(f'Missing detector objects for {name}')
        names = object_names(indexed[name], objects_field, name_key)
        # Retain detector order; repeated class names appear once. Empty detections
        # stay empty instead of introducing a fabricated object or caption.
        texts[name] = ', '.join(names)
    return texts
