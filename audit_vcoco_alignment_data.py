"""Audit human action/role labels and lock image groups; no training or inference.

Role IDs establish annotated links, not detector coverage or complete negatives.
Official V-COCO test labels are deliberately not an input to this preparation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from captioning.supervised_relations import decode_action


REVISION = '489cc4db74f2f10ab4b134f67da3874afbf245ab'
FOCUS = (('hold', 'obj', 'holding'), ('ride', 'instr', 'riding'))
CAPTION_SHA256 = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
SOURCE_SHA256 = {
    'vcoco_train': '2267afe7adc0a6c3e8ac90fb35f6d3a5bd5cd3c7edebf0622d0cbb7a67fddf19',
    'vcoco_val': '2025bf42ce985a079ad88fd052aba47d85f1531c5b04a95804427c8bcb75eedd',
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def caption_index(dataset):
    result = {}
    for index, row in enumerate(dataset['images']):
        iid = int(row['cocoid'])
        filename_id = int(Path(row['filename']).stem.rsplit('_', 1)[1])
        if iid != filename_id or iid in result:
            raise ValueError('Duplicate or inconsistent COCO image identity.')
        if row['split'] not in ('train', 'restval', 'val', 'test'):
            raise ValueError('Unknown caption split.')
        result[iid] = {'caption_row': index, 'caption_split': row['split']}
    return result


def summarize(actions, caption):
    names = [item['action_name'] for item in actions]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate V-COCO action.')
    by_action = {item['action_name']: decode_action(item) for item in actions}
    image_ids = {int(row['image_id']) for rows in by_action.values() for row in rows.values()}
    missing = image_ids - caption.keys()
    if missing:
        raise ValueError(f'{len(missing)} source images missing from caption metadata.')
    eligible = {iid for iid in image_ids if caption[iid]['caption_split'] in ('train', 'restval')}
    result = {
        'all_images': len(image_ids),
        'caption_split_images': dict(Counter(caption[i]['caption_split'] for i in sorted(image_ids))),
        'eligible_images': len(eligible), 'relations': {},
    }
    for action, role, relation in FOCUS:
        if action not in by_action:
            raise ValueError(f'Missing action {action}.')
        if any(role not in row['roles'] for row in by_action[action].values()):
            raise ValueError(f'Missing role {action}/{role}.')
        rows = [(sid, row) for sid, row in by_action[action].items() if row['image_id'] in eligible]
        links = [(sid, row['roles'][role], row['image_id']) for sid, row in rows
                 if row['label'] == 1 and row['roles'][role] > 0]
        if any(sid == oid for sid, oid, _ in links):
            raise ValueError('Role object must differ from actor.')
        result['relations'][relation] = {
            'actor_labels': dict(Counter(str(row['label']) for _, row in rows)),
            'positive_role_links': len(links),
            'positive_role_images': len({iid for _, _, iid in links}),
            'positive_action_missing_role': sum(row['label'] == 1 and row['roles'][role] == 0
                                                for _, row in rows),
        }
    return result, image_ids, eligible


def make_partition(caption, official_train, official_val, seed=20261018):
    if official_train & official_val:
        raise ValueError('Official relation train/val image overlap.')
    # Relation dev stays in caption train/restval: it never becomes caption validation.
    ordered = sorted(official_train)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 10 or not official_val:
        raise ValueError('Insufficient eligible relation images.')
    count = max(1, round(.2 * len(ordered)))
    groups = {
        'relation_fit': sorted(ordered[count:]),
        'relation_calibration': sorted(ordered[:count]),
        'relation_dev': sorted(official_val),
    }
    # Exactly reproduce existing SCST tune1000 selection, including original row order.
    validation = sorted((iid for iid in caption if caption[iid]['caption_split'] == 'val'),
                        key=lambda iid: caption[iid]['caption_row'])
    if len(validation) != 5000:
        raise ValueError('Expected pinned Karpathy validation5000.')
    random.Random(20261017).shuffle(validation)
    groups['caption_tune1000'] = sorted(validation[:1000])
    groups['caption_validation_remainder4000'] = sorted(validation[1000:])
    forbidden = set(groups['relation_calibration']) | set(groups['relation_dev'])
    others = [iid for iid in caption if caption[iid]['caption_split'] in ('train', 'restval')
              and iid not in forbidden and iid not in official_train]
    others.sort()
    random.Random(seed).shuffle(others)
    # Include all fit images; fill to 10k without relation calibration/dev images.
    fit = groups['relation_fit']
    if len(fit) > 10000 or len(fit) + len(others) < 10000:
        raise ValueError('Cannot construct declared caption train10k.')
    groups['caption_pilot_train10000'] = sorted(fit + others[:10000 - len(fit)])
    train_ids = set(groups['caption_pilot_train10000'])
    if train_ids & forbidden or any(caption[i]['caption_split'] not in ('train', 'restval') for i in train_ids):
        raise ValueError('Caption pilot overlaps relation dev/calibration or caption heldout.')
    relation_groups = [set(groups[k]) for k in ('relation_fit', 'relation_calibration', 'relation_dev')]
    if any(a & b for i, a in enumerate(relation_groups) for b in relation_groups[i + 1:]):
        raise ValueError('Relation image partitions overlap.')
    return groups


def audit(dataset, actions):
    caption = caption_index(dataset)
    summaries, all_ids, eligible = {}, {}, {}
    for split in ('train', 'val'):
        summaries[split], all_ids[split], eligible[split] = summarize(actions[split], caption)
    if all_ids['train'] & all_ids['val']:
        raise ValueError('Official V-COCO source splits overlap.')
    groups = make_partition(caption, eligible['train'], eligible['val'])
    # Actual supervision support in each locked relation partition.
    support = {}
    for key in ('relation_fit', 'relation_calibration', 'relation_dev'):
        source = actions['val' if key == 'relation_dev' else 'train']
        ids = set(groups[key])
        restricted = {iid: {**row, 'caption_split': row['caption_split'] if iid in ids else 'val'}
                      for iid, row in caption.items()}
        support[key] = summarize(source, restricted)[0]['relations']
    summary = {
        'status': 'ANNOTATION_INVENTORY_COMPLETE_NOT_MODEL_READY',
        'variant': 'vcoco_grounded_caption_week1_data_audit_v1',
        'vcoco_revision': REVISION, 'source_splits': summaries,
        'caption_images': len(caption),
        'caption_split_images': dict(Counter(row['caption_split'] for row in caption.values())),
        'locked_group_sizes': {key: len(value) for key, value in groups.items()},
        'locked_relation_support': support,
        'source_heldout_images_excluded': sum(s['all_images'] - s['eligible_images'] for s in summaries.values()),
        'caption_heldout_in_relation_fit_calibration_dev': 0,
        'caption_pilot_overlap_relation_calibration_dev': 0,
        'official_vcoco_test_annotations_loaded': False, 'caption_metrics_computed': False,
        'test_captions_or_scores_used': False, 'gpu_training_run': False,
        'limits': [
            'Counts describe actor/action annotations and role links, not localized detector pair labels.',
            'A different or missing role object is unknown, never automatically a false target.',
            'Known actor-action absence can supervise that action; positive actions may have several valid targets.',
            'Object categories/boxes, detector recall and cache coverage are NOT verified by this inventory.',
            'Previously inspected V-COCO development data are not a new independent benchmark.',
            'Existing caption checkpoint saw train/restval; scorer image separation does not imply unseen to captioner.',
            'Caption pilot starts from a full-train checkpoint: 10k is adaptation size, not total training history.',
        ],
    }
    return summary, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('caption-json', 'vcoco-train', 'vcoco-val', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite a locked cohort.')
    if digest(args.caption_json) != CAPTION_SHA256:
        raise ValueError('Caption file differs from pinned SCST input identity.')
    for key, expected in SOURCE_SHA256.items():
        if digest(getattr(args, key)) != expected:
            raise ValueError(f'{key} differs from pinned official V-COCO revision.')
    dataset = json.loads(args.caption_json.read_text(encoding='utf-8'))
    actions = {split: json.loads(getattr(args, 'vcoco_' + split).read_text(encoding='utf-8'))
               for split in ('train', 'val')}
    summary, groups = audit(dataset, actions)
    summary['input_sha256'] = {key: digest(getattr(args, key))
                               for key in ('caption_json', 'vcoco_train', 'vcoco_val')}
    args.output.mkdir(parents=True)
    partition = args.output / 'locked_image_groups.json'
    partition.write_text(json.dumps(groups, indent=2), encoding='utf-8')
    summary['locked_image_groups_sha256'] = digest(partition)
    (args.output / 'data_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
