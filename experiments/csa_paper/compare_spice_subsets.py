"""Compare saved SPICE subsets; no Java, model or scoring execution."""
import argparse
import math
from pathlib import Path
from score_prompt_test_extra import read, save
from score_spice_subsets import CATEGORIES


def paired(left, right):
    a, b = ({r['coco_id']:r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Paired cohort differs or duplicate IDs')
    result = {}
    for category in CATEGORIES:
        delta = []
        for i in sorted(a):
            x, y = [d[i]['scores'].get(category, {}).get('f') for d in (a, b)]
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                delta.append(y-x)
        result[category] = dict(common_finite_images=len(delta),
             prompt_minus_control_x100=100*sum(delta)/len(delta) if delta else None)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    roots={'CONTROL':args.control, 'PROMPT':args.prompt}
    results={a:read(root/'result.json') for a,root in roots.items()}
    for arm,r in results.items():
        if r['status']!='SPICE_SUBSETS_COMPLETE' or r['arm']!=arm or r['images']!=5000:
            raise ValueError('Incomplete arm')
    differences={}
    for policy in ('normal','fixed_rerank'):
        rows=[read(roots[a]/(policy+'_spice_per_image.json')) for a in ('CONTROL','PROMPT')]
        if any(len(r)!=5000 for r in rows):raise ValueError('Incomplete cohort')
        differences[policy]=paired(*rows)
    save(args.output,dict(status='SPICE_SUBSETS_PAIRED_COMPLETE', arms=results,
                         paired_common_denominators=differences))
