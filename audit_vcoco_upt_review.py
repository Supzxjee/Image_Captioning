"""Replay returned UPT audit decisions and annotation labels; no model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

from audit_vcoco_alignment_data import digest
from captioning.vcoco_relation_audit import audit_image, fit_thresholds, evaluate


def verify_native_row(row, ref, annotations, image, categories):
    detections = row['detection_details']
    if len(detections) != row['detections']:
        raise ValueError('Detection inventory differs.')
    if any(not math.isfinite(d['confidence']) or not 0 <= d['confidence'] <= 1 for d in detections):
        raise ValueError('Invalid detection confidence.')
    claims = [{k: c[k] for k in ('subject', 'object', 'predicate', 'score')} for c in row['candidates']]
    replay = audit_image(ref, {'detections': detections, 'claims': claims, 'native_pairs': row['native_pairs']},
                         annotations, image, categories)
    if any(row[k] != value for k, value in replay.items()):
        raise ValueError('Box/IoU/label replay differs.')
    raw = {(c['subject'], c['object'], c['predicate']): c['score'] for c in row['raw_claims']}
    if len(raw) != len(row['raw_claims']):
        raise ValueError('Duplicate raw claim.')
    expected_keys = set()
    for (s, o, p), score in raw.items():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid raw score.')
        if detections[s]['confidence'] * detections[o]['confidence'] > 0:
            expected_keys.add((s, o, p))
    if {(c['subject'], c['object'], c['predicate']) for c in claims} != expected_keys:
        raise ValueError('Native/raw supported pair inventory differs.')
    for c in claims:
        s, o, p = c['subject'], c['object'], c['predicate']
        expected = raw[s, o, p] * detections[s]['confidence']**2.8 * detections[o]['confidence']**2.8
        if not math.isclose(c['score'], expected, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError('Native score differs from official detector-prior formula.')


def calibration_diagnostic(rows, predicate):
    candidates = [c for r in rows for c in r['candidates'] if c['predicate'] == predicate and c['label'] >= 0]
    grouped = {}
    for c in candidates:
        grouped.setdefault(c['score'], []).append(c['label'])
    tp = count = 0
    best = None
    for score in sorted(grouped, reverse=True):
        count += len(grouped[score]); tp += sum(grouped[score])
        if count >= 5 and (best is None or tp / count > best['known_precision']):
            best = {'known_precision': tp / count, 'threshold': score, 'accepted_known': count,
                    'accepted_true': tp, 'accepted_false_known': count - tp}
    return {'maximum_observed_known_precision_with_at_least_five_accepts': best,
            'scope': 'Calibration-only failure diagnostic, not a new selected policy or development result.'}


def riding_action_diagnostic(rows, sources):
    actions = defaultdict(set)
    hashes = {'train': '2267afe7adc0a6c3e8ac90fb35f6d3a5bd5cd3c7edebf0622d0cbb7a67fddf19',
              'val': '2025bf42ce985a079ad88fd052aba47d85f1531c5b04a95804427c8bcb75eedd'}
    for split, expected in hashes.items():
        path = sources / f'vcoco_{split}.json'
        if digest(path) != expected:
            raise ValueError('V-COCO action source changed.')
        for action in json.loads(path.read_text()):
            for aid, label in zip(action['ann_id'], action['label']):
                if label == 1:
                    actions[aid].add(action['action_name'])
    best = calibration_diagnostic(rows, 'riding')['maximum_observed_known_precision_with_at_least_five_accepts']
    if best is None:
        return None
    objects, activity, count = Counter(), Counter(), 0
    for row in rows:
        for c in row['candidates']:
            if c['predicate'] == 'riding' and c['label'] == 0 and c['score'] >= best['threshold']:
                count += 1
                objects[row['detection_details'][c['object']]['label']] += 1
                sports = actions[c['matched_subject_id']] & {'surf', 'skateboard', 'snowboard', 'ski'}
                activity['+'.join(sorted(sports)) if sports else 'none_of_four'] += 1
    return {'scope': 'Calibration-only diagnostic at maximum-precision tail; NOT an accepted policy.',
            'threshold': best['threshold'], 'known_false_claims': count,
            'predicted_object_classes': dict(objects), 'actor_positive_vcoco_sports_actions': dict(activity),
            'limit': 'Actor action label alone does not establish the proposed target or shared HICO/V-COCO semantics.'}


def run(args):
    with zipfile.ZipFile(args.archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError('Archive duplicate entries or CRC failure.')
        raw = {name: archive.read(name) for name in archive.namelist()}
    cfg, reported, locked = [json.loads(raw[name]) for name in
                              ('protocol.json', 'run_summary.json', 'locked_calibration_policy.json')]
    if reported['status'] != 'FROZEN_RELATION_AUDIT_COMPLETE' or 'failure.json' in raw:
        raise ValueError('Not a completed relation audit.')
    if hashlib.sha256(raw['protocol.json']).hexdigest() != reported['protocol_sha256']:
        raise ValueError('Protocol hash mismatch.')
    if hashlib.sha256(raw['locked_calibration_policy.json']).hexdigest() != reported['policy_sha256']:
        raise ValueError('Policy hash mismatch.')
    if cfg['checkpoint_sha256'] != '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c':
        raise ValueError('Unexpected frozen model.')
    source = {}
    for group in ('relation_calibration', 'relation_dev'):
        for row in json.loads((args.grounded_data / f'{group}_images.json').read_text()):
            source[row['image_id']] = {**row, 'group': group}
    if {r['image_id'] for r in cfg['images']} != set(source) or len(cfg['images']) != len(source):
        raise ValueError('Protocol image inventory differs.')
    for row in cfg['images']:
        if row != source[row['image_id']]:
            raise ValueError('Protocol image/annotation metadata differs from verified source.')
    instances = json.loads(args.instances.read_text())
    if digest(args.instances) != cfg['instances_sha256']:
        raise ValueError('COCO instance identity differs.')
    anns = {r['id']: r for r in instances['annotations']}
    categories = {r['id']: r['name'] for r in instances['categories']}
    image_metadata = {r['id']: r for r in instances['images']}
    by_image = defaultdict(list)
    for ann in instances['annotations']:
        by_image[ann['image_id']].append(ann)
    rows = []
    for rank in (0, 1):
        stamp = json.loads(raw[f'upt_{rank}.protocol.json'])
        if stamp != {'rank': rank, 'configuration': cfg}:
            raise ValueError('Shard provenance differs.')
        values = [json.loads(line) for line in raw[f'upt_{rank}.jsonl'].splitlines()]
        if [r['image_id'] for r in values] != [r['image_id'] for r in cfg['images'][rank::2]]:
            raise ValueError('Shard IDs incomplete or reordered.')
        rows += values
    known_cells = unknown_cells = 0
    for row in rows:
        ref = source[row['image_id']]
        if row['group'] != ref['group'] or row['filename'] != ref['filename']:
            raise ValueError('Image metadata differs.')
        if cfg.get('score_mode', 'raw') == 'native':
            verify_native_row(row, ref, by_image[row['image_id']], image_metadata[row['image_id']], categories)
        positives = {(r['subject_id'], r['predicate'], r['object_id']) for r in ref['relations']}
        actions = {(r['subject_id'], r['predicate']): r['label'] for r in ref['actor_action_labels']}
        node_map, inverse, keys = {}, {}, set()
        available = set()
        for c in row['candidates']:
            s, o, p = c['subject'], c['object'], c['predicate']
            key = s, p, o
            if key in keys or s == o or not 0 <= s < row['detections'] or not 0 <= o < row['detections']:
                raise ValueError('Duplicate or invalid candidate pair.')
            keys.add(key)
            if p not in ('holding', 'riding') or not math.isfinite(c['score']) or not 0 <= c['score'] <= 1:
                raise ValueError('Invalid candidate score/predicate.')
            sid, oid = c['matched_subject_id'], c['matched_object_id']
            for index, aid in ((s, sid), (o, oid)):
                if aid is None:
                    continue
                if anns[aid]['image_id'] != row['image_id']:
                    raise ValueError('Matched instance from another image.')
                if index in node_map and node_map[index] != aid or aid in inverse and inverse[aid] != index:
                    raise ValueError('Mapping is not consistently one-to-one.')
                node_map[index] = aid; inverse[aid] = index
            if sid is not None and categories[anns[sid]['category_id']] != 'person':
                raise ValueError('Matched subject is not person.')
            expected = 1 if (sid, p, oid) in positives else 0 if actions.get((sid, p), -1) == 0 else -1
            if c['label'] != expected:
                raise ValueError('Known/unknown label differs from source.')
            known_cells += expected >= 0; unknown_cells += expected == -1
            if expected == 1:
                available.add((sid, p, oid))
        if {(p['subject_id'], p['predicate'], p['object_id']) for p in row['positive_links']} != positives:
            raise ValueError('Positive denominator differs.')
        for p in row['positive_links']:
            key = p['subject_id'], p['predicate'], p['object_id']
            if p['available'] != (key in available) or p['available'] and not p['localized']:
                raise ValueError('Positive availability flags inconsistent.')
    cal = [r for r in rows if r['group'] == 'relation_calibration']
    dev = [r for r in rows if r['group'] == 'relation_dev']
    policy = fit_thresholds(cal)
    if policy != locked or locked != reported['policy']:
        raise ValueError('Calibration-only threshold replay differs.')
    if evaluate(cal, locked) != reported['calibration'] or evaluate(dev, locked) != reported['development']:
        raise ValueError('Recomputed metrics differ.')
    report = {'status': 'RETURNED_ARTIFACT_AUDIT_PASS', 'archive_sha256': digest(args.archive),
              'score_mode': cfg.get('score_mode', 'raw'),
              'box_and_prior_replay': cfg.get('score_mode', 'raw') == 'native',
              'archive_bytes': args.archive.stat().st_size, 'code_commit': cfg['code_commit'],
              'calibration_images': len(cal), 'development_images': len(dev),
              'candidate_known_labels_verified': known_cells, 'candidate_unknown_labels_verified': unknown_cells,
              'policies': locked, 'development': evaluate(dev, locked),
              'calibration_failure_diagnostics': {p: calibration_diagnostic(cal, p) for p in ('holding', 'riding')},
              'limits': ['CRC/identity/annotation labels/thresholds/metrics replayed; no new inference.',
                         ('Boxes/IoU/labels and native score formula replayed from saved detections/raw scores; model inference not replayed.'
                          if cfg.get('score_mode', 'raw') == 'native' else
                          'Raw audit omits boxes/native scores; no independent IoU or detector-prior replay.'),
                         'No threshold lowering or development-based policy selection performed.',
                         'Reused development and incomplete role annotation; no full-proposal precision guarantee.'],
              'caption_metrics': False, 'new_gpu_run': False, 'promote_to_prompt_source': False}
    if args.vcoco_sources is not None:
        if cfg.get('score_mode', 'raw') != 'native':
            raise ValueError('Action diagnostic requires saved native detection details.')
        report['riding_ontology_diagnostic'] = riding_action_diagnostic(cal, args.vcoco_sources)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcoco-sources', type=Path)
    for key in ('archive', 'grounded-data', 'instances', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
