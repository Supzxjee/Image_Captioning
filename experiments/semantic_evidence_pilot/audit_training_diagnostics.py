"""Summarize saved rank histories without loading or executing checkpoint code."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np


def audit(source, output):
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
        histories = [json.loads(archive.read(f'history_rank{r}.json')) for r in (0, 1)]
        assert all(len(h) == 1250 for h in histories)
        result = {'windows': {}, 'gradients': {}}
        for start, end in ((1,100), (101,200), (201,500), (501,1000), (1001,1250)):
            rows = [x for rank in histories for x in rank[start-1:end]]
            pairs = sum(x['alignment']['pairs'] for x in rows)
            assert pairs > 0
            result['windows'][f'{start}-{end}'] = {
                'loss_mean': float(np.mean([x['alignment']['loss'] for x in rows])),
                'pairs': pairs,
                'pair_weighted_positive_margin_fraction': sum(x['alignment']['pairs']*(x['alignment']['accuracy'] or 0) for x in rows)/pairs,
                'pair_weighted_margin': sum(x['alignment']['pairs']*(x['alignment']['margin'] or 0) for x in rows)/pairs,
            }
        for key in histories[0][0]['submodule_grad_norms']:
            values = [x['submodule_grad_norms'][key] for rank in histories for x in rank]
            assert np.isfinite(values).all()
            result['gradients'][key] = {
                'positive_rank_steps': sum(v > 0 for v in values),
                'rank_steps': len(values), 'median': float(np.median(values)),
            }
        result['validity'] = {'all_alignment_pair_steps_positive': all(x['alignment']['pairs'] > 0 for rank in histories for x in rank)}
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2))
