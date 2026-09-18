"""Image-only scene schemas and portable prompt-cache keys (no ML imports)."""
import json
from pathlib import PurePosixPath

SPATIAL_RELATIONS = {'left of', 'right of', 'above', 'below', 'in front of',
                     'behind', 'on', 'under', 'inside', 'next to'}
SCENE_INSTRUCTION = '''Inspect only this image. Return one JSON object, no prose or caption.
Schema: {"objects": [{"name": "person", "attributes": ["red shirt"]}],
"relations": [{"subject": "person", "predicate": "next to", "object": "bicycle"}]}.
Use English. List 1 to 4 salient visible objects, most important first, with 0 to 2
short visible attributes each. Names must be unique. List 0 to 2 clearly visible
spatial relations, most important first. Relation subject/object must exactly
match an object name. Predicates allowed: left of, right of, above, below,
in front of, behind, on, under, inside, next to. Do not guess hidden details.
If uncertain omit attributes/relations. Never write a complete image caption.'''


def clean_phrase(value):
    if not isinstance(value, str):
        raise ValueError('Scene phrases must be strings.')
    value = ' '.join(value.lower().strip().split())
    if not value or len(value.split()) > 6 or len(value) > 80:
        raise ValueError(f'Invalid scene phrase: {value!r}')
    return value


def parse_scene(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    scene = json.loads(text)
    if not isinstance(scene, dict) or set(scene) != {'objects', 'relations'}:
        raise ValueError('Expected objects and relations only.')
    objects, relations = scene['objects'], scene['relations']
    if not isinstance(objects, list) or not 1 <= len(objects) <= 4:
        raise ValueError('Expected 1 to 4 objects.')
    normalized, names = [], set()
    for item in objects:
        if not isinstance(item, dict) or set(item) != {'name', 'attributes'}:
            raise ValueError('Invalid object schema.')
        name = clean_phrase(item['name'])
        attrs = item['attributes']
        if name in names or not isinstance(attrs, list) or len(attrs) > 2:
            raise ValueError('Duplicate name or invalid attributes.')
        names.add(name)
        normalized.append({'name': name, 'attributes': [clean_phrase(a) for a in attrs]})
    if not isinstance(relations, list) or len(relations) > 2:
        raise ValueError('Expected at most 2 relations.')
    normalized_relations = []
    for item in relations:
        if not isinstance(item, dict) or set(item) != {'subject', 'predicate', 'object'}:
            raise ValueError('Invalid relation schema.')
        relation = {key: clean_phrase(value) for key, value in item.items()}
        if (relation['subject'] not in names or relation['object'] not in names
                or relation['subject'] == relation['object']
                or relation['predicate'] not in SPATIAL_RELATIONS):
            raise ValueError('Unsupported or ungrounded relation.')
        normalized_relations.append(relation)
    return {'objects': normalized, 'relations': normalized_relations}


def scene_prompt(scene, variant):
    objects = [' '.join(item['attributes'] + [item['name']]) for item in scene['objects']]
    if variant == 'objects':
        return '; '.join(objects)
    if variant != 'spatial':
        raise ValueError('Unknown prompt variant.')
    # Relation first so the fixed 20-token budget does not discard it last.
    relations = ['{subject} {predicate} {object}'.format(**r) for r in scene['relations']]
    return '; '.join(relations + objects)


def align_prompt_cache(cache, image_paths):
    """Legacy exact paths supported; filename mapping cannot silently collide."""
    by_name = {}
    for key, value in cache.items():
        name = PurePosixPath(str(key).replace('\\', '/')).name
        if name in by_name:
            raise ValueError(f'Ambiguous prompt cache filename: {name}')
        by_name[name] = value
    aligned = {}
    for path in image_paths:
        name = PurePosixPath(str(path).replace('\\', '/')).name
        if path in cache:
            aligned[path] = cache[path]
        elif name in by_name:
            aligned[path] = by_name[name]
    return aligned
