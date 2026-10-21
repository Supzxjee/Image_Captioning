"""Read-only VG/COCO/selected-pair feasibility audit. No feature extraction/training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from captioning.object_consistency import COCO_LABELS, SPECIAL_ALIASES

DATASET_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
RECEIPT_SHA = '873f6da0e1b4c4f7d7408b67c593c07ce56e5b4ca3fa1db5f9ce5e04aa188920'
HUMANS = set('person man woman boy girl child baby guy gentleman lady teenager rider skier surfer snowboarder skateboarder player'.split())
SPATIAL = {'on', 'in', 'at', 'near', 'by', 'with', 'has', 'of', 'behind', 'above', 'below', 'under',
           'over', 'beside', 'next to', 'in front of', 'on top of', 'part of', 'attached to', 'is', 'are', 'have'}
ALIASES = defaultdict(set)
for index, label in enumerate(COCO_LABELS):
    for name in [label, *SPECIAL_ALIASES.get(label, ())]:
        if name not in {'vehicle', 'vehicles', 'monitor', 'monitors'}:
            ALIASES[name].add(index)
for name in HUMANS:
    ALIASES[name].add(0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def iter_json_array(stream, chunk=1024**2):
    """Incremental top-level array parser, bounded by the largest image graph."""
    decoder = json.JSONDecoder()
    buffer, eof, first = '', False, True
    def more():
        nonlocal buffer, eof
        text = stream.read(chunk)
        eof = text == ''
        buffer += text
    more()
    buffer = buffer.lstrip()
    if not buffer.startswith('['):
        raise ValueError('Expected top-level JSON array')
    buffer = buffer[1:]
    while True:
        buffer = buffer.lstrip()
        if not buffer and not eof:
            more(); continue
        if buffer.startswith(']'):
            if buffer[1:].strip() or stream.read().strip():
                raise ValueError('Trailing JSON data')
            return
        if not first:
            if not buffer.startswith(','):
                raise ValueError('Missing array separator')
            buffer = buffer[1:].lstrip()
        while True:
            buffer = buffer.lstrip()
            try:
                obj, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Truncated/malformed array') from None
                more()
        yield obj
        first = False
        buffer = buffer[end:]


def norm(text):
    return ' '.join(re.findall(r'[a-z]+', str(text).lower()))


def lemma(token):
    if token.endswith('ies'):
        token = token[:-3] + 'y'
    elif token.endswith('ing'):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif token.endswith('ed'):
        token = token[:-2]
    elif token.endswith('s') and len(token) > 3:
        token = token[:-1]
    return token[:-1] if token.endswith('e') else token


def contains(phrase, text, stem=False):
    a, b = norm(phrase).split(), norm(text).split()
    if stem:
        a, b = [lemma(t) for t in a], [lemma(t) for t in b]
    return bool(b) and any(a[i:i+len(b)] == b for i in range(len(a)-len(b)+1))


def names(obj):
    raw = obj.get('names', [obj.get('name', '')])
    return [norm(n) for n in (raw if isinstance(raw, list) else [raw]) if norm(n)]


def mapped_label(obj):
    labels = set().union(*(ALIASES[n] for n in names(obj)))
    return next(iter(labels)) if len(labels) == 1 else None


def box(obj, width, height):
    values = [obj.get(k) for k in ('x', 'y', 'w', 'h')]
    if not all(isinstance(v, (int, float)) for v in values) or min(width, height, values[2], values[3]) <= 0:
        return None
    x, y, w, h = values
    if x < 0 or y < 0 or x+w > width+1 or y+h > height+1:
        return None
    return [x/width, y/height, (x+w)/width, (y+h)/height]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection / union if union > 0 else 0


def match_pair(record, subject_label, object_label, subject_box, object_box):
    found = []
    for slot, (labels, boxes) in enumerate(zip(record['selected_labels'], record['boxes_original_normalized'])):
        if labels != [subject_label, object_label]:
            continue
        overlap = [iou(subject_box, boxes[0]), iou(object_box, boxes[1])]
        if min(overlap) >= .5:
            found.append((slot, overlap))
    return found


def allow_image(cid, coco, cached, heldout, excluded):
    return cid in cached and cid in coco and cached[cid]['split'] == 'train' \
        and coco[cid]['split'] in ('train', 'restval') and cid not in heldout | excluded


def audit(dataset_path, cache_path, annotations):
    assert digest(dataset_path) == DATASET_SHA, 'COCO split JSON differs'
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8'))
    coco = {int(r['cocoid']): r for r in dataset['images']}
    heldout = {i for i, r in coco.items() if r['split'] in ('val', 'test')}
    excluded = set(json.loads((ROOT.parent / 'upt_hidden_pair/excluded_phrase_ids.json').read_text()))
    with zipfile.ZipFile(cache_path) as archive:
        assert hashlib.sha256(archive.read('cache_receipt.json')).hexdigest() == RECEIPT_SHA
        receipt = json.loads(archive.read('cache_receipt.json'))
        cached = {}
        for rank in range(2):
            name = f'pairs_rank{rank}.jsonl'
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == receipt['files'][name]
            for row in map(json.loads, payload.splitlines()):
                cid = int(row['coco_id'])
                assert cid not in cached
                assert coco[cid]['imgid'] == row['image_id']
                cached[cid] = row
    with zipfile.ZipFile(annotations / 'image_data.json.zip') as archive:
        metadata = json.loads(archive.read('image_data.json'))
    by_vg = {int(r['image_id']): r for r in metadata}
    assert len(by_vg) == len(metadata)
    mapped = [r for r in metadata if r.get('coco_id') in coco]
    split_counts = Counter(coco[r['coco_id']]['split'] for r in mapped)
    allowed = {v: m for v, m in by_vg.items() if allow_image(m.get('coco_id'), coco, cached, heldout, excluded)}
    funnel, rejected = Counter(), Counter()
    candidates, seen = [], set()
    region_source = annotations / 'region_graphs.json.zip'
    with zipfile.ZipFile(region_source) as archive, archive.open('region_graphs.json') as binary:
        for count, image in enumerate(iter_json_array(io.TextIOWrapper(binary, encoding='utf-8')), 1):
            vg_id = int(image.get('image_id', image.get('id', -1)))
            if count % 10000 == 0:
                print('Scanned', count, 'VG images; candidates', len(candidates), flush=True)
            if vg_id not in allowed:
                continue
            m = allowed[vg_id]; cid = int(m['coco_id']); record = cached[cid]
            funnel['allowed_cached_train_images_seen'] += 1
            if abs(m['width']/m['height'] - record['original_wh'][0]/record['original_wh'][1]) > .002:
                rejected['image_aspect_mismatch'] += 1; continue
            if record['selected_pair_count'] == 0:
                rejected['no_selected_pairs_images'] += 1; continue
            funnel['allowed_images_with_selected_pairs'] += 1
            for region in image['regions']:
                funnel['regions_in_cached_train_with_pairs'] += 1
                relations = region.get('relationships', [])
                if len(relations) != 1:
                    rejected['zero_or_multiple_graph_relations'] += 1; continue
                funnel['one_graph_relation'] += 1
                objects = {o['object_id']: o for o in region.get('objects', [])}
                rel = relations[0]
                s, o = objects.get(rel.get('subject_id')), objects.get(rel.get('object_id'))
                if s is None or o is None or s['object_id'] == o['object_id'] or len(objects) != 2:
                    rejected['not_exactly_two_resolved_objects'] += 1; continue
                sl, ol = mapped_label(s), mapped_label(o)
                if sl != 0 or ol in (0, None):
                    rejected['not_unambiguous_human_to_coco_object'] += 1; continue
                funnel['human_coco_object_relation'] += 1
                predicate, phrase = norm(rel.get('predicate', '')), region.get('phrase', '').strip()
                words = norm(phrase).split()
                if predicate in SPATIAL or not predicate or len(words) < 3 or len(words) > 25 or set(words) & {'not', 'no', 'never', 'without', 'and', 'or'}:
                    rejected['spatial_complex_negated_or_length'] += 1; continue
                if not contains(phrase, predicate, stem=True) or not any(contains(phrase, n) for n in names(s)) or not any(contains(phrase, n) for n in names(o)):
                    rejected['phrase_not_lexically_tied_to_edge'] += 1; continue
                funnel['explicit_single_edge_phrase'] += 1
                sb, ob = box(s, m['width'], m['height']), box(o, m['width'], m['height'])
                if sb is None or ob is None:
                    rejected['invalid_boxes'] += 1; continue
                matches = match_pair(record, sl, ol, sb, ob)
                if not matches:
                    rejected['no_same_class_selected_pair_both_iou_ge_05'] += 1; continue
                if len(matches) != 1:
                    rejected['multiple_qualifying_selected_pairs'] += 1; continue
                slot, overlap = matches[0]
                key = (cid, slot, norm(phrase))
                if key in seen:
                    rejected['duplicate_image_slot_phrase'] += 1; continue
                seen.add(key)
                candidates.append(dict(vg_image_id=vg_id, coco_id=cid, image_id=record['image_id'],
                    filename=record['filename'], filepath=record['filepath'], source_split=coco[cid]['split'],
                    region_id=region['region_id'], relationship_id=rel.get('relationship_id'), phrase=phrase,
                    predicate=predicate, subject_names=names(s), object_names=names(o),
                    subject_id=s['object_id'], object_id=o['object_id'], object_class=COCO_LABELS[ol],
                    pair_slot=slot, selected_native_row=record['selected_native_rows'][slot],
                    proposal_indices=record['selected_pairs'][slot], endpoint_iou=overlap,
                    vg_boxes_normalized=[sb, ob], predicted_boxes_normalized=record['boxes_original_normalized'][slot],
                    original_wh=record['original_wh'], cache_image_sha256=record['image_sha256'],
                    status='RULE_FILTERED_CANDIDATE_NOT_HUMAN_VERIFIED'))
    assert not ({r['coco_id'] for r in candidates} & (heldout | excluded))
    predicate_counts = Counter(r['predicate'] for r in candidates)
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row['predicate']].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: hashlib.sha256(f"vg-audit-v1:{r['coco_id']}:{r['region_id']}".encode()).hexdigest())
    sample, chosen = [], set()
    while len(sample) < min(30, len({r['coco_id'] for r in candidates})):
        progressed = False
        for pred in sorted(grouped, key=lambda x: (-predicate_counts[x], x)):
            while grouped[pred] and grouped[pred][0]['coco_id'] in chosen:
                grouped[pred].pop(0)
            if grouped[pred] and len(sample) < 30:
                row = grouped[pred].pop(0); sample.append(row); chosen.add(row['coco_id']); progressed = True
        if not progressed:
            break
    report = dict(status='DATA_FEASIBILITY_RULE_AUDIT_COMPLETE_PENDING_VISUAL_REVIEW',
        model_or_training_executed=False, source_dataset_sha256=DATASET_SHA,
        cache_review_sha256=digest(cache_path), hidden_receipt_sha256=RECEIPT_SHA,
        vg_metadata_images=len(metadata), vg_metadata_coco_split_counts=dict(split_counts),
        heldout_all_coco_val_test=len(heldout), known_exposed_ids_excluded=len(excluded),
        vg_images_in_cache_train_before_exclusion=sum(m.get('coco_id') in cached and cached[m['coco_id']]['split']=='train' for m in metadata),
        allowed_cached_train_vg_images=len(allowed), funnel=dict(funnel), rejected=dict(rejected),
        candidates=len(candidates), candidate_images=len({r['coco_id'] for r in candidates}),
        unique_phrases=len({norm(r['phrase']) for r in candidates}), predicate_counts=dict(predicate_counts.most_common()),
        object_counts=dict(Counter(r['object_class'] for r in candidates).most_common()),
        practical_target=dict(images=1000, phrases=2000, status='Not a statistical power guarantee; no automatic GO.'),
        candidates_are_verified=False, test_used_for_learning=False,
        limits=['Single-edge/two-object/explicit-lexical filters are conservative; rejected does not mean false.',
                'COCO identity and aspect checked; visual review still must verify image/box orientation and phrase truth.',
                'Matching uses only the top8 selected cached pairs; no claim about all native UPT proposals.',
                'No GT replacement at inference; no synthetic phrases, no model scoring, no negatives manufactured.',
                'Exact COCO IDs prevent known split leakage; unknown visual duplicates/pretraining overlap not exhaustively audited.'])
    return report, candidates, sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache-review', type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / 'review'; output.mkdir(exist_ok=True)
    report, rows, sample = audit(args.dataset, args.cache_review, output / 'downloads')
    for name, data in [('RULE_AUDIT.json', report), ('candidates.json', rows), ('sample30.json', sample)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
