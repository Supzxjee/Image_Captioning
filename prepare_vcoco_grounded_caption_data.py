"""Join human role links to COCO GT boxes, retaining every available positive edge.

This exports supervision, NOT inferred prompts. Missing roles remain unknown and
no caption text or group caption is synthesized from an annotation count.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from audit_vcoco_alignment_data import CAPTION_SHA256, FOCUS, SOURCE_SHA256, caption_index, digest, groups_digest
from captioning.supervised_relations import decode_action, normalized_box


GROUP_SHA256 = '60af7979e900ff8afa4fc51d722b827e1051e5630348deae2799d75f179037b8'
RELATION_GROUPS = ('relation_fit', 'relation_calibration', 'relation_dev')


def instance_index(instances):
    def unique(rows, key):
        result = {int(row[key]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f'Duplicate COCO {key}.')
        return result
    images = unique(instances['images'], 'id')
    annotations = unique(instances['annotations'], 'id')
    categories = {key: row['name'] for key, row in unique(instances['categories'], 'id').items()}
    if any(row['image_id'] not in images or row['category_id'] not in categories for row in annotations.values()):
        raise ValueError('COCO annotation references missing image or category.')
    return images, annotations, categories


def grounded_images(actions, instances, captions, selected):
    images, annotations, categories = instance_index(instances)
    by_action = {row['action_name']: decode_action(row) for row in actions}
    if len(by_action) != len(actions) or any(name not in by_action for name, _, _ in FOCUS):
        raise ValueError('Duplicate or missing focus action.')
    selected = set(selected)
    source_ids = {row['image_id'] for values in by_action.values() for row in values.values()}
    if selected - source_ids:
        raise ValueError('Selected image is outside the annotation source split.')
    records = {}
    for iid in sorted(selected):
        if iid not in images or iid not in captions or captions[iid]['split'] not in ('train', 'restval'):
            raise ValueError('Source image missing or outside caption training.')
        image, caption = images[iid], captions[iid]
        if any(not math.isfinite(image[key]) or image[key] <= 0 for key in ('width', 'height')):
            raise ValueError('Invalid source image dimensions.')
        if image['file_name'] != caption['filename']:
            raise ValueError('COCO and caption filenames disagree.')
        records[iid] = {'image_id': iid, 'filename': caption['filename'], 'filepath': caption.get('filepath', ''),
                        'caption_split': caption['split'], 'width': image['width'], 'height': image['height'],
                        'instances': {}, 'relations': [], 'actor_action_labels': []}
    failures, missing_roles = Counter(), Counter()

    def resolve(aid, iid):
        if aid not in annotations:
            return None, 'missing_annotation'
        annotation = annotations[aid]
        if annotation['image_id'] != iid:
            raise ValueError('Role/actor annotation belongs to a different image.')
        if annotation.get('iscrowd', 0):
            return None, 'crowd_box'
        try:
            bbox = normalized_box(annotation, images[iid])
        except (ValueError, KeyError, TypeError):
            return None, 'invalid_box'
        return {'instance_id': aid, 'category_id': annotation['category_id'],
                'category': categories[annotation['category_id']], 'box': bbox}, None

    for name, role, predicate in FOCUS:
        for sid, row in sorted(by_action[name].items()):
            if role not in row['roles']:
                raise ValueError('Required action role is absent from source schema.')
            iid = row['image_id']
            if iid not in selected:
                continue
            subject, error = resolve(sid, iid)
            if error:
                failures[f'{predicate}:subject:{error}'] += 1
                continue
            if subject['category'] != 'person':
                raise ValueError('Actor is not a person.')
            record = records[iid]
            record['instances'][sid] = subject
            record['actor_action_labels'].append({'subject_id': sid, 'predicate': predicate, 'label': row['label']})
            if row['label'] != 1:
                continue
            oid = row['roles'].get(role, 0)
            if oid <= 0:
                missing_roles[predicate] += 1
                continue
            if oid == sid:
                raise ValueError('Self-referential positive role.')
            target, error = resolve(oid, iid)
            if error:
                failures[f'{predicate}:object:{error}'] += 1
                continue
            record['instances'][oid] = target
            record['relations'].append({'subject_id': sid, 'predicate': predicate, 'object_id': oid})
    for record in records.values():
        record['instances'] = [record['instances'][key] for key in sorted(record['instances'])]
        record['relations'].sort(key=lambda row: (row['subject_id'], row['predicate'], row['object_id']))
        keys = [(row['subject_id'], row['predicate'], row['object_id']) for row in record['relations']]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate positive relation.')
    stats = relation_statistics(list(records.values()))
    stats['unusable_endpoints'] = dict(failures)
    stats['positive_actions_missing_role'] = dict(missing_roles)
    return list(records.values()), stats


def relation_statistics(records):
    relation_stats = {}
    for _, _, predicate in FOCUS:
        links = [(record['image_id'], row) for record in records for row in record['relations']
                 if row['predicate'] == predicate]
        relation_stats[predicate] = {'links': len(links), 'images': len({iid for iid, _ in links})}
    repeated_images = Counter()
    multi_actor = 0
    for record in records:
        categories = {row['instance_id']: row['category'] for row in record['instances']}
        subjects = defaultdict(set)
        for row in record['relations']:
            subjects[(row['predicate'], categories[row['object_id']])].add(row['subject_id'])
        repeated = [key for key, value in subjects.items() if len(value) >= 2]
        if repeated:
            multi_actor += 1
        for predicate, category in repeated:
            repeated_images[f'{predicate}:{category}'] += 1
    counts = [len(record['relations']) for record in records]
    return {'images': len(records), 'positive_links': sum(counts),
            'images_with_positive_links': sum(count > 0 for count in counts),
            'images_with_multiple_links': sum(count >= 2 for count in counts),
            'maximum_links_per_image': max(counts, default=0),
            'links_per_image_histogram': dict(Counter(str(count) for count in counts)),
            'relations': relation_stats,
            'images_with_multiple_actors_same_predicate_object_class': multi_actor,
            'multi_actor_image_counts_by_predicate_object_class': dict(sorted(repeated_images.items())),
            'maximum_relation_cap_applied': False}


def prepare(actions, instances, captions, groups):
    required = {key: set(groups[key]) for key in RELATION_GROUPS}
    if any(a & b for index, a in enumerate(required.values()) for b in list(required.values())[index + 1:]):
        raise ValueError('Relation partitions overlap by image.')
    data, stats = {}, {}
    for key in RELATION_GROUPS:
        source = actions['val' if key == 'relation_dev' else 'train']
        data[key], stats[key] = grounded_images(source, instances, captions, required[key])
    return data, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sources', 'instances-json', 'groups', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory.')
    if groups_digest(args.groups) != GROUP_SHA256:
        raise ValueError('Locked image groups differ.')
    if digest(args.sources / 'dataset_coco.json') != CAPTION_SHA256:
        raise ValueError('Caption input differs from pinned baseline.')
    actions = {}
    for split in ('train', 'val'):
        path = args.sources / f'vcoco_{split}.json'
        if digest(path) != SOURCE_SHA256['vcoco_' + split]:
            raise ValueError('V-COCO source identity differs.')
        actions[split] = json.loads(path.read_text())
    source_provenance = json.loads(args.instances_json.with_suffix('.provenance.json').read_text())
    if digest(args.instances_json) != source_provenance['output_sha256']:
        raise ValueError('COCO subset provenance hash differs.')
    dataset = json.loads((args.sources / 'dataset_coco.json').read_text())
    caption_index(dataset)  # Validate row identity and split vocabulary before joining.
    captions = {row['cocoid']: row for row in dataset['images']}
    groups = json.loads(args.groups.read_text())
    data, statistics = prepare(actions, json.loads(args.instances_json.read_text()), captions, groups)
    summary = {
        'status': 'GT_BOX_JOIN_COMPLETE_NOT_DETECTOR_OR_CAPTION_RESULT',
        'variant': 'vcoco_grounded_caption_all_positive_links_v1', 'statistics': statistics,
        'sources': {'caption_sha256': CAPTION_SHA256, 'groups_sha256': GROUP_SHA256,
                    **SOURCE_SHA256, 'coco_subset_sha256': digest(args.instances_json),
                    'coco_official_member': source_provenance['member'],
                    'coco_official_member_sha256': source_provenance['uncompressed_member_sha256']},
        'coordinate_format': 'normalized_xyxy', 'box_source': 'official COCO ground truth',
        'relation_cap': None, 'caption_metrics_computed': False, 'training_run': False,
        'annotation_rules': [
            'Each edge links one exact subject instance to one exact object instance.',
            'All annotated positive links are retained after endpoint validation; no top-one/top-three selection.',
            'One instance may participate in several predicates and several edges.',
            'Missing links do not establish a negative relation or a negative target.',
            'Actor action labels are separate from pair links; absent actor action is not a generic wrong-target label.',
            'Annotation cardinality is not the exact visible count and does not synthesize a group caption.',
            'GT-linked candidates are supervision only; inference must enumerate candidates without GT role IDs.',
            'No model prediction, detector recall, caption improvement or novelty follows from this preparation.',
        ],
    }
    args.output.mkdir(parents=True)
    summary['output_sha256'] = {}
    for key, records in data.items():
        path = args.output / (key + '_images.json')
        path.write_text(json.dumps(records, separators=(',', ':')), encoding='utf-8')
        summary['output_sha256'][path.name] = digest(path)
    (args.output / 'box_join_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
