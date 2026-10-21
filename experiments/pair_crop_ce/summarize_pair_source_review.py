"""Score the fixed 32-phrase human review without relabeling or model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ALLOWED = {'correct_specific', 'generic', 'wrong_relation_or_fact', 'wrong_instance', 'unclear'}


def summarize(review, key, payload):
    assert review['schema_version'] == 1
    assert review['complete'] is True and review['reviewer_type'] == 'human_self_reported'
    assert isinstance(review['reviewer'], str) and review['reviewer'].strip()
    assert review['review_sha256'] == key['review_sha256'] == payload['review_sha256']
    unsigned = {k: v for k, v in payload.items() if k != 'review_sha256'}
    sha = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert sha == payload['review_sha256'], 'Review payload changed'
    candidates = {c['candidate_id']: (t, c) for t in payload['tasks'] for c in t['candidates']}
    mapping, annotations = key['mapping'], review['annotations']
    assert len(payload['tasks']) == 16 and len(candidates) == len(mapping) == len(annotations) == 32
    assert set(annotations) == set(mapping) == set(candidates)
    for a in annotations.values():
        assert a['status'] in ALLOWED and isinstance(a.get('notes', ''), str)
    inconsistent = []
    for task in payload['tasks']:
        a, b = task['candidates']
        if a['caption'] == b['caption'] and a['truncated'] == b['truncated']:
            if annotations[a['candidate_id']]['status'] != annotations[b['candidate_id']]['status']:
                inconsistent.append(task['task_id'])
    assert not inconsistent, 'Same pair and identical output received different statuses'
    rows = defaultdict(dict)
    counts = defaultdict(Counter)
    for cid, item in mapping.items():
        arm, gid, i = item['arm'], item['group_id'], item['event_index']
        assert arm in ('FULLFRAME', 'PAIRCROP') and i in (0, 1)
        task, candidate = candidates[cid]
        assert (arm, i) not in rows[gid]
        status = annotations[cid]['status']
        counts[arm][status] += 1
        rows[gid][arm, i] = dict(status=status, correct=status == 'correct_specific' and not candidate['truncated'],
            truncated=candidate['truncated'], caption=candidate['caption'], image_id=task['event']['image_id'],
            family=item['family'], notes=annotations[cid].get('notes', ''))
    assert len(rows) == 8 and all(len(items) == 4 for items in rows.values())
    groups, by_family = {}, defaultdict(lambda: Counter(FULLFRAME=0, PAIRCROP=0))
    for gid, items in rows.items():
        family = items['FULLFRAME', 0]['family']
        good = {arm: all(items[arm, i]['correct'] for i in (0, 1)) for arm in ('FULLFRAME', 'PAIRCROP')}
        for arm in good:
            by_family[family][arm] += int(good[arm])
        groups[gid] = dict(family=family, joint_correct=good,
                          events={arm: [items[arm, i] for i in (0, 1)] for arm in good})
    gains = [g for g, v in groups.items() if v['joint_correct']['PAIRCROP'] and not v['joint_correct']['FULLFRAME']]
    regressions = [g for g, v in groups.items() if v['joint_correct']['FULLFRAME'] and not v['joint_correct']['PAIRCROP']]
    joint = {arm: sum(v['joint_correct'][arm] for v in groups.values()) for arm in ('FULLFRAME', 'PAIRCROP')}
    criteria = dict(crop_at_least_6_of_8=joint['PAIRCROP'] >= 6,
                    net_gain_at_least_3=len(gains) - len(regressions) >= 3,
                    regressions_at_most_1=len(regressions) <= 1,
                    gains_in_both_families=all(v['PAIRCROP'] > v['FULLFRAME'] for v in by_family.values()))
    return dict(status='GO' if all(criteria.values()) else 'NO_GO_FOR_FULL_SCALE_PAIRCROP',
                review_sha256=review['review_sha256'], reviewer=review['reviewer'],
                reviewer_type=review['reviewer_type'], complete=True,
                event_status_counts={a: dict(c) for a, c in counts.items()},
                correct_events={a: sum(v['correct'] for items in rows.values() for (arm, _), v in items.items() if arm == a)
                                for a in ('FULLFRAME', 'PAIRCROP')},
                joint_correct_groups=joint, family_joint_correct={k: dict(v) for k, v in by_family.items()},
                gains=gains, regressions=regressions, net_gain=len(gains) - len(regressions),
                criteria=criteria, unclear_count=sum(c['unclear'] for c in counts.values()),
                identical_output_label_conflicts=inconsistent, groups=groups,
                limits=['One AI-assisted human reviewer; identity masking, not independent adjudication.',
                        '8 groups new to adaptation, not a whole-system independent test.',
                        'Phrase-level result; no new full-caption CIDEr or CHAIR measurement.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    raw = args.review.read_bytes()
    key = json.loads((HERE / 'review/pair_source_blind_key.json').read_text(encoding='utf-8'))
    page = (HERE / 'review/pair_source_blind/index.html').read_text(encoding='utf-8')
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))
    result = summarize(json.loads(raw), key, payload)
    result['human_export_sha256'] = hashlib.sha256(raw).hexdigest()
    (HERE / 'PAIR_SOURCE_HUMAN_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['status'], result['correct_events'], result['joint_correct_groups'])
