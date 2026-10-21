"""Reuse existing human pair labels and stored captions; no model execution."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ARCHIVE = Path('C:/Users/leduc/Downloads/pair_source_fullframe_ce_review.zip')
FIT = {440092, 368370, 520204, 45949, 540716, 570474, 526325, 181841}
NOTES = {
    200984: ('relation_error_candidate', 'Phone is held in front of the person, not at the ear; inspected full image. Pair truth was previously human verified; whole-image judgment here is assistant only.'),
    503488: ('object_error_candidate', 'People carry skateboards; caption says luggage. Inspected full image. This is not a pure relation error.'),
    521602: ('object_error_candidate', 'Stored caption says fish rather than a person with a board. Do not count this as a relation-only failure.'),
    160647: ('object_error_candidate', 'Stored caption says paddle boat; candidate image contains paddleboarders. Object/category error, not a verified relation-only failure.'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    labels_path = EXPERIMENTS / 'phrase_pair_feasibility/FEASIBLE_GROUPS_21.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    with zipfile.ZipFile(ARCHIVE) as z:
        before_raw = z.read('before_records.json')
        before = {r['group_id']: r for r in json.loads(before_raw)}
        protocol = json.loads(z.read('protocol.json'))
        assert protocol['checkpoint_sha256'] == '5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'
    rows = []
    excluded = []
    for group in labels['groups']:
        if group['same_image']:
            excluded.append(dict(group_id=group['group_id'], reason='Opposite selected-pair targets in one image; cannot silently become opposite whole-image targets.'))
            continue
        off = before[group['group_id']]['conditions']['off']['captions']
        for j, (event, target) in enumerate(zip(group['events'], group['target_description_indices'])):
            iid = event['image_id']
            classification, note = NOTES.get(iid, ('not_adjudicated', 'No automatic omission-to-error conversion.'))
            rows.append(dict(coco_id=iid, group_id=group['group_id'], filename=event['filename'],
                human_pair_target=group['descriptions'][target],
                human_pair_contrast=group['descriptions'][1-target],
                historical_before_off_caption=off[j],
                caption_origin='Frozen-base memory + masked auxiliary slots, before any adaptation; historical proxy, not new exact-original replay.',
                role='basis_shortlist' if iid in FIT else 'unassigned',
                assistant_error_screen=classification, assistant_note=note,
                whole_image_human_verified=False, protected_contrasts=[],
                ready_for_oracle=False))
    assert len(rows) == 34 and len({r['coco_id'] for r in rows}) == 34
    assert len(excluded) == 4
    assert sum(r['role'] == 'basis_shortlist' for r in rows) == 8
    assert not any(r['ready_for_oracle'] for r in rows)
    result = dict(status='DATA_AUDIT_COMPLETE_ORACLE_NOT_READY',
        sources=dict(labels_sha256=sha(labels_path.read_bytes()), archive_sha256=sha(ARCHIVE.read_bytes()), before_records_sha256=sha(before_raw)),
        counts=dict(human_verified_pair_groups=21, excluded_same_image_groups=4,
            cross_image_groups=17, unique_cross_image_candidates=34,
            provisional_basis_images=8, assistant_screened_relation_error_candidates=1,
            additional_object_error_candidates=3, fully_locked_oracle_images=0),
        missing=['Whole-image salience/target confirmation; existing human labels describe selected pairs.',
            'Four eligible relation-error diagnostic images are not established by this audit.',
            'Verified protected contrasts at baseline-correct content positions are not recorded.',
            'Decision prefixes/tokens must use the actual checkpoint tokenizer.',
            'Historical before/off captions are screening evidence; exact-original generation parity is not independently established here.'],
        excluded_groups=excluded, candidates=rows, model_run=False, new_human_labels=False)
    (HERE / 'EXISTING_DATA_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
