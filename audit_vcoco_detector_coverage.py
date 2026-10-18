"""Coverage of all fixed V-COCO links by cached detector candidates; no inference.

Candidates are formed without role labels. Ground truth is used only afterwards
for a class-aware one-to-one localization audit, not to invent relation negatives.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_vcoco_alignment_data import digest
from captioning.detection_audit import index_source
from captioning.detector_pair_matching import match_instances, normalized_detection_box
from captioning.relation_prompts import select_detections
from captioning.supervised_relations import normalized_box
from prepare_vcoco_grounded_caption_data import RELATION_GROUPS, instance_index


DETECTION_SHA = '498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'


def candidate_inventory(entry, width, height, cap):
    selected = select_detections(entry, min_confidence=.5, max_objects=cap, duplicate_iou=.8)
    detections, rejected = [], 0
    for item in selected:
        if not 0 <= item.confidence <= 1:
            rejected += 1
            continue
        try:
            box = normalized_detection_box(item.box, width, height, 'pixel_xyxy')
        except ValueError:
            rejected += 1
            continue
        detections.append({'label': item.label, 'box': box, 'confidence': item.confidence})
    # Ordered human -> any other instance. No GT class/predicate compatibility gate.
    pair_count = sum(item['label'] == 'person' for item in detections) * max(0, len(detections) - 1)
    return detections, pair_count, rejected


def measure(records, indexed, instances, cap):
    images, anns, categories = instance_index(instances)
    by_image = defaultdict(dict)
    for aid, ann in anns.items():
        if ann.get('iscrowd', 0):
            continue
        try:
            box = normalized_box(ann, images[ann['image_id']])
        except ValueError:
            continue
        by_image[ann['image_id']][aid] = {'label': categories[ann['category_id']], 'box': box}
    reports, counts, outcomes = [], Counter(), defaultdict(Counter)
    for record in records:
        iid, filename = record['image_id'], record['filename']
        missing = filename not in indexed
        detections, candidate_pairs, rejected = ([], 0, 0) if missing else candidate_inventory(
            indexed[filename], record['width'], record['height'], cap)
        # Compete against ALL valid GT instances in the image, not just role endpoints.
        matched = match_instances(by_image[iid], detections, min_iou=.5)
        links = []
        for link in record['relations']:
            subject, obj = matched.get(link['subject_id']), matched.get(link['object_id'])
            status = ('both_matched' if subject and obj else 'subject_missing' if obj else
                      'object_missing' if subject else 'both_missing')
            outcomes[link['predicate']]['total'] += 1
            outcomes[link['predicate']][status] += 1
            links.append({**link, 'status': status,
                          'subject_detection_index': subject['index'] if subject else None,
                          'object_detection_index': obj['index'] if obj else None})
        counts['images'] += 1
        counts['missing_cache_images'] += missing
        counts['empty_retained_detection_images'] += not missing and not detections
        counts['images_without_candidate_pairs'] += candidate_pairs == 0
        counts['candidate_pairs'] += candidate_pairs
        counts['invalid_selected_boxes'] += rejected
        counts['positive_links'] += len(links)
        counts['localized_positive_links'] += sum(link['status'] == 'both_matched' for link in links)
        positive = bool(links)
        counts['positive_link_images'] += positive
        counts['positive_images_with_any_localized_link'] += positive and any(l['status'] == 'both_matched' for l in links)
        counts['positive_images_with_all_links_localized'] += positive and all(l['status'] == 'both_matched' for l in links)
        counts['images_without_focus_links'] += not positive
        counts['candidate_pairs_on_images_without_focus_links'] += candidate_pairs if not positive else 0
        reports.append({'image_id': iid, 'filename': filename,
                        'cache_status': 'missing' if missing else 'present',
                        'detections': detections, 'candidate_pairs': candidate_pairs,
                        'ground_truth_instances_for_matching': len(by_image[iid]),
                        'matched_gt_instances': len(matched), 'links': links})
    result = dict(counts)
    result['link_localization_recall'] = counts['localized_positive_links'] / counts['positive_links'] if counts['positive_links'] else None
    result['per_predicate'] = {}
    for name, values in outcomes.items():
        result['per_predicate'][name] = {**dict(values), 'localization_recall': values['both_matched'] / values['total']}
    return result, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('grounded-data', 'instances-json', 'detections', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh audit directory.')
    if digest(args.detections) != DETECTION_SHA:
        raise ValueError('Detection cache differs from pinned baseline input.')
    source_summary = json.loads((args.grounded_data / 'box_join_summary.json').read_text())
    if digest(args.instances_json) != source_summary['sources']['coco_subset_sha256']:
        raise ValueError('COCO subset identity differs.')
    indexed = index_source(json.loads(args.detections.read_text()))
    instances = json.loads(args.instances_json.read_text())
    groups = {}
    seen = set()
    for group in RELATION_GROUPS:
        path = args.grounded_data / f'{group}_images.json'
        if digest(path) != source_summary['output_sha256'][path.name]:
            raise ValueError('Grounded source export identity differs.')
        groups[group] = json.loads(path.read_text())
        ids = [row['image_id'] for row in groups[group]]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping source images.')
        seen.update(ids)
    args.output.mkdir(parents=True)
    summary = {
        'status': 'CACHED_DETECTOR_COVERAGE_AUDITED_NOT_RELATION_ACCURACY',
        'detections_sha256': DETECTION_SHA, 'grounded_summary_sha256': digest(args.grounded_data / 'box_join_summary.json'),
        'configuration': {'confidence': .5, 'nms_iou': .8, 'match_iou': .5,
                          'caps': [10, 100], 'coordinate_format': 'pixel_xyxy',
                          'matching': 'class-aware IoU-descending one-to-one greedy over ALL noncrowd valid GT objects',
                          'candidate_generation': 'every retained person -> every other retained instance; no predicate filter'},
        'results': {}, 'output_sha256': {}, 'new_model_inference': False,
        'caption_evaluation': False, 'training': False,
        'limits': ['Caps compare a fixed deployment budget and a diagnostic reference; no cap selected by this audit.',
                   'Localization recall over annotated positive links is not HOI AP, relation accuracy or candidate precision.',
                   'Candidate pairs on images with no positive focus link are not established false relations.',
                   'Missing cache record differs from a valid empty detection list; both remain in full denominators.',
                   'Matching is deterministic greedy, not maximum matching; audit protocol may affect ambiguous overlaps.',
                   'Detector pretraining provenance is not audited here; no unseen-image generalization claim.',
                   'GT mappings are audit outputs, not permitted inference candidate selectors.'],
    }
    for cap in (10, 100):
        summary['results'][str(cap)] = {}
        for group, records in groups.items():
            result, reports = measure(records, indexed, instances, cap)
            path = args.output / f'cap{cap}_{group}_reports.json'
            path.write_text(json.dumps(reports, separators=(',', ':')), encoding='utf-8')
            summary['results'][str(cap)][group] = result
            summary['output_sha256'][path.name] = digest(path)
            print('cap', cap, group, result, flush=True)
    (args.output / 'detector_coverage_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
