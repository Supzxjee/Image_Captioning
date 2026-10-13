"""Reparse saved answers offline; preserve raw answers and explicit schema-repair flags."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from captioning.vlm_relation_audit import parse_teacher_answer


def repair(records):
    repaired = []
    for row in records:
        try:
            answer = parse_teacher_answer(row['raw_answer'], repair_status=True)
        except (ValueError, TypeError) as error:
            answer = {'decision': 'parse_error', 'proposed_relations': [], 'parse_error': str(error)}
        # Do not retain stale parsed fields or invent evidence from a failed parse.
        pair_keys = ('pair_index', 'filename', 'slot', 'subject', 'object', 'old_predicate',
                     'subject_box', 'object_box', 'confidence', 'clip_proposal', 'clip_decision',
                     'raw_answer', 'seconds')
        repaired.append({**{k: row[k] for k in pair_keys if k in row}, **answer,
                         'original_decision': row['decision']})
    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Audit ZIP or vlm_records.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == 'vlm_records.json']
            if len(matches) != 1:
                raise ValueError('Expected exactly one vlm_records.json in ZIP.')
            data = archive.read(matches[0])
    else:
        data = source.read_bytes()
    records = repair(json.loads(data))
    indices = [r['pair_index'] for r in records]
    if sorted(indices) != list(range(len(records))):
        raise ValueError('Missing or duplicate pair indices.')
    records.sort(key=lambda r: r['pair_index'])
    output.mkdir(parents=True, exist_ok=True)
    summary = {'count': len(records), 'source_records_sha256': hashlib.sha256(data).hexdigest(),
               'decision_counts': dict(Counter(r['decision'] for r in records)),
               'schema_repairs': [r['pair_index'] for r in records if 'schema_repair' in r],
               'notes': 'Schema repair only; no visual correctness or accuracy claim.'}
    (output / 'repaired_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (output / 'repair_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
