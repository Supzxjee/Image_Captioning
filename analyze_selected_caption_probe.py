"""Recompute caption wording versus score changes from a completed probe ZIP."""
import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path


def analyze(summary, rows):
    if not rows or len(rows) != summary['sample_count']:
        raise ValueError('Record count differs from summary')
    if len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Duplicate filenames')
    result = dict(sample_count=len(rows), valid_relations=sum(
        r['conditions']['on']['valid_relations'] for r in rows), conditions={})
    for condition, flag in [('off', 'off_top_changed'), ('target_swap', 'swap_top_changed')]:
        memory, deltas, top_changed, wording_changed, score_changed = [], [], 0, 0, 0
        for row in rows:
            on, other = row['conditions']['on'], row['conditions'][condition]
            if condition == 'off' and other['valid_relations'] != 0:
                raise ValueError('Off condition still has active relations')
            if condition == 'target_swap' and other['valid_relations'] != on['valid_relations']:
                raise ValueError('Target swap changed relation mask count')
            a, b = on['candidates'], other['candidates']
            if not a or len(a) != len(b):
                raise ValueError('Missing/mismatched candidates')
            changed = a[0]['caption'] != b[0]['caption']
            if row[flag] != changed:
                raise ValueError('Stored caption-change flag disagrees with records')
            top_changed += changed
            wording_changed += [c['caption'] for c in a] != [c['caption'] for c in b]
            score_changed += any(x['logprob'] != y['logprob'] for x, y in zip(a, b))
            memory.append(other['memory_relative_change'])
            deltas.extend(abs(x['logprob'] - y['logprob']) for x, y in zip(a, b))
        if top_changed != summary[flag]:
            raise ValueError('Summary top-change count disagrees with records')
        if not all(math.isfinite(v) and v >= 0 for v in memory + deltas):
            raise ValueError('Invalid memory/score measurements')
        result['conditions'][condition] = dict(top_wording_changed=top_changed,
            candidate_wording_or_order_changed=wording_changed,
            images_with_candidate_logprob_changes=score_changed,
            mean_memory_relative_change=statistics.mean(memory),
            median_memory_relative_change=statistics.median(memory),
            min_memory_relative_change=min(memory), max_memory_relative_change=max(memory),
            mean_abs_candidate_logprob_change=statistics.mean(deltas),
            max_abs_candidate_logprob_change=max(deltas))
    weights = [v for row in rows for v in row['conditions']['on']['evidence_weights']]
    if len(weights) != result['valid_relations'] or not weights:
        raise ValueError('Evidence weights disagree with relation count')
    result['evidence_weight'] = dict(mean=statistics.mean(weights), min=min(weights), max=max(weights))
    result['notes'] = ('Frozen historical checkpoint; selected development images and unknown alternative '
        'targets. Generic V2.1 prompt retained. Nonzero score/memory changes do not imply quality '
        'improvement. No CIDEr/BLEU/CHAIR evaluation; no model training or checkpoint promotion.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.review)
    with zipfile.ZipFile(source) as archive:
        summary = json.loads(archive.read('summary.json'))
        rows = json.loads(archive.read('records.json'))
    result = analyze(summary, rows)
    result['source_review_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['source_summary'] = summary
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
