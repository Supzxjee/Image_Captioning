"""Minimal VG phrase–region inventory and locked 8/8/8 sample. No model calls."""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile
from audit_data import ROOT, DATASET_SHA, RECEIPT_SHA, digest, iter_json_array, allow_image, mapped_label

OUTPUT = ROOT / 'review' / 'region24'
DATASET = Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
CACHE = Path('C:/Users/leduc/Downloads/upt_hidden_pairs_11k_review.zip')
SPATIAL = re.compile(r'\b(on|in|under|over|above|below|behind|beside|near|between|next to|in front of)\b', re.I)
HUMAN_WORD = re.compile(r'\b(person|people|man|men|woman|women|boy|girl|child|children|baby|surfer|sufer|skier|rider|snowboarder)\b', re.I)
RELATION_WORD = re.compile(r'\b(on|in|of|with|under|over|above|below|behind|beside|near|between|has|have|wear\w*|hold\w*|rid\w*|carrying|carry|using|feeding|looking|standing|sitting|laying|lying|touching|throwing|catching|next to|in front of)\b', re.I)


def region_box(region, metadata):
    """Strict source box validity; never clip, expand or repair annotations."""
    fields = [region.get(k) for k in ('x', 'y', 'width', 'height')]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in fields):
        return None
    x, y, w, h = fields
    width, height = metadata['width'], metadata['height']
    if min(w, h, width, height) <= 0 or x < 0 or y < 0 or x+w > width or y+h > height:
        return None
    return [x/width, y/height, (x+w)/width, (y+h)/height]


def diagnostic_group(region, selected_pairs):
    # Tags stratify inspection only; graph/text tags are not semantic truth labels.
    if selected_pairs == 0:
        return 'no_selected_UPT_pair'
    objects = {o['object_id']: o for o in region.get('objects', [])}
    for rel in region.get('relationships', []):
        subject, obj = objects.get(rel.get('subject_id')), objects.get(rel.get('object_id'))
        if subject is not None and obj is not None and mapped_label(subject) == 0 and mapped_label(obj) != 0:
            return 'human_object'
    if region.get('relationships') or SPATIAL.search(region['phrase']):
        return 'spatial_or_other'
    return 'other_description'


def sample24(rows, exclude_images):
    groups = ['human_object', 'spatial_or_other', 'no_selected_UPT_pair']
    chosen, sample = set(exclude_images), []
    for group in groups:
        candidates = sorted((r for r in rows if r['diagnostic_group'] == group
                             and (group == 'no_selected_UPT_pair' or RELATION_WORD.search(r['phrase']))
                             and (group != 'human_object' or HUMAN_WORD.search(r['phrase']))),
            key=lambda r: hashlib.sha256(f"vg-region24-v1:{r['coco_id']}:{r['region_id']}".encode()).hexdigest())
        selected = []
        for row in candidates:
            if row['coco_id'] not in chosen:
                selected.append(row); chosen.add(row['coco_id'])
                if len(selected) == 8:
                    break
        if len(selected) != 8:
            raise ValueError('Insufficient distinct images in stratum: ' + group)
        sample.extend(selected)
    return sample


def run():
    assert digest(DATASET) == DATASET_SHA
    coco = {int(r['cocoid']): r for r in json.loads(DATASET.read_text(encoding='utf-8'))['images']}
    heldout = {cid for cid, r in coco.items() if r['split'] in ('val', 'test')}
    excluded = set(json.loads((ROOT.parent / 'upt_hidden_pair/excluded_phrase_ids.json').read_text()))
    cached = {}
    with zipfile.ZipFile(CACHE) as archive:
        assert hashlib.sha256(archive.read('cache_receipt.json')).hexdigest() == RECEIPT_SHA
        receipt = json.loads(archive.read('cache_receipt.json'))
        for rank in range(2):
            name = f'pairs_rank{rank}.jsonl'; raw = archive.read(name)
            assert hashlib.sha256(raw).hexdigest() == receipt['files'][name]
            for row in map(json.loads, raw.splitlines()):
                cid = row['coco_id']; assert cid not in cached
                assert coco[cid]['imgid'] == row['image_id']
                cached[cid] = row
    downloads = ROOT / 'review/downloads'
    with zipfile.ZipFile(downloads / 'image_data.json.zip') as archive:
        metadata = json.loads(archive.read('image_data.json'))
    allowed = {m['image_id']: m for m in metadata if allow_image(m.get('coco_id'), coco, cached, heldout, excluded)}
    inventory, counts, rejects = [], Counter(), Counter()
    with zipfile.ZipFile(downloads / 'region_graphs.json.zip') as archive, archive.open('region_graphs.json') as raw:
        for index, image in enumerate(iter_json_array(io.TextIOWrapper(raw, encoding='utf-8')), 1):
            if index % 20000 == 0:
                print('Scanned', index, 'images; valid region candidates', len(inventory), flush=True)
            if image['image_id'] not in allowed:
                continue
            m = allowed[image['image_id']]; cid = m['coco_id']; c = cached[cid]
            counts['allowed_cached_train_images'] += 1
            if abs(m['width']/m['height'] - c['original_wh'][0]/c['original_wh'][1]) > .002:
                rejects['image_aspect_mismatch'] += 1; continue
            counts['geometry_compatible_images'] += 1
            for region in image['regions']:
                counts['regions_before_minimal_filters'] += 1
                phrase = region.get('phrase')
                if not isinstance(phrase, str) or not phrase.strip():
                    rejects['empty_phrase'] += 1; continue
                normalized = region_box(region, m)
                if normalized is None:
                    rejects['invalid_region_box'] += 1; continue
                inventory.append(dict(coco_id=cid, image_id=c['image_id'], vg_image_id=m['image_id'],
                    source_split=coco[cid]['split'], filename=c['filename'], filepath=c['filepath'],
                    original_wh=c['original_wh'], cache_image_sha256=c['image_sha256'],
                    vg_wh=[m['width'], m['height']], region_id=region['region_id'], phrase=phrase,
                    vg_region_xywh=[region[k] for k in ('x', 'y', 'width', 'height')],
                    region_normalized=normalized,
                    region_area_fraction=(normalized[2]-normalized[0])*(normalized[3]-normalized[1]),
                    selected_UPT_pairs=c['selected_pair_count'], graph_relationship_count=len(region.get('relationships', [])),
                    diagnostic_group=diagnostic_group(region, c['selected_pair_count']),
                    status='MINIMAL_RULE_CANDIDATE_NOT_VERIFIED'))
    assert not ({r['coco_id'] for r in inventory} & (heldout | excluded))
    prior = json.loads((ROOT / 'SAMPLE30_MANIFEST.json').read_text(encoding='utf-8'))
    sample = sample24(inventory, {r['coco_id'] for r in prior})
    assert len(sample) == len({r['coco_id'] for r in sample}) == 24
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'inventory.jsonl').open('w', encoding='utf-8') as stream:
        for row in inventory:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    sample_raw = json.dumps(sample, ensure_ascii=False, indent=2) + '\n'
    (ROOT / 'REGION24_MANIFEST.json').write_text(sample_raw, encoding='utf-8')
    areas = sorted(r['region_area_fraction'] for r in inventory)
    report = dict(status='INVENTORY_AND_SAMPLE_LOCKED_BEFORE_VISUAL_REVIEW', model_or_training_executed=False,
        source_dataset_sha256=DATASET_SHA, cache_receipt_sha256=RECEIPT_SHA,
        source_region_zip_sha256=digest(downloads / 'region_graphs.json.zip'),
        source_metadata_zip_sha256=digest(downloads / 'image_data.json.zip'),
        counts=dict(counts), rejects=dict(rejects), region_candidates=len(inventory),
        unique_images=len({r['coco_id'] for r in inventory}),
        unique_phrases=len({r['phrase'].strip().lower() for r in inventory}),
        no_pair_images=len({r['coco_id'] for r in inventory if r['selected_UPT_pairs']==0}),
        no_pair_regions=sum(r['selected_UPT_pairs']==0 for r in inventory),
        diagnostic_stratum_counts=dict(Counter(r['diagnostic_group'] for r in inventory)),
        median_region_area_fraction=areas[len(areas)//2],
        whole_image_like_regions_ge_075=sum(a >= .75 for a in areas),
        sample_sha256=digest(ROOT / 'REGION24_MANIFEST.json'),
        sample_strata=dict(Counter(r['diagnostic_group'] for r in sample)),
        excluded_prior_audit_images=len(prior), sample_selection='Fixed SHA256 ordering, eight per group, all24 images distinct; previous30 excluded. Paired strata require lexical relation cue; human stratum requires human word. Group tags are not GT.',
        policy='Original nonempty phrase + valid original region + established COCO/cache identity/aspect. No pair match, class/edge-count/phrase-length/lexical/spatial blacklist gate.',
        heldout_policy='All5000COCOval+5000COCOtest plus568known-exposed IDs excluded; train/restval only.',
        constraints='Restricted to existing cache cohort for image identity/selected-pair strata. These are not full-VG counts or validated supervision labels.')
    (ROOT / 'REGION_INVENTORY_RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
