"""Paired image bootstrap of saved scalars/counts; never load a model/scorer."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('full', 'zero', 'm2')
METRICS = ('CIDEr_x100', 'CHAIRs_pp', 'CHAIRi_pp')


def statistics(rows):
    # Shape: (..., images, arms, [CIDEr, hallucinated mentions, object mentions]).
    objects = rows[..., 2].sum(axis=-2)
    if np.any(objects == 0):
        raise ValueError('Zero object-mention denominator')
    return np.stack((rows[..., 0].mean(axis=-2),
                     (rows[..., 1] > 0).mean(axis=-2),
                     rows[..., 1].sum(axis=-2)/objects), axis=-1)*100


def resample(rows, replicates, seed, batch_size=64):
    if rows.ndim != 3 or rows.shape[1:] != (3, 3) or len(rows) < 1:
        raise ValueError('Expected images x 3 arms x 3 saved values')
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.empty((replicates, 2, 3), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates-start)
        indices = rng.integers(len(rows), size=(size, len(rows)))
        values = statistics(rows[indices])
        differences[start:start+size] = values[:, :1, :] - values[:, 1:, :]
    return differences


def load_inputs():
    path = HERE/'PAIRED_INPUT.csv'
    manifest = json.loads((HERE/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['input_sha256']
    with path.open(encoding='utf-8', newline='') as stream:
        table = list(csv.DictReader(stream))
    ids = [int(r['coco_id']) for r in table]
    assert len(ids) == len(set(ids)) == manifest['images'] == 5000 and ids == sorted(ids)
    fields = ('CIDEr', 'hallucinated_mentions', 'object_mentions')
    rows = np.array([[[float(r[f'{a}_{f}']) for f in fields] for a in ARMS] for r in table])
    assert np.isfinite(rows).all() and (rows >= 0).all()
    assert np.equal(rows[..., 1:], np.floor(rows[..., 1:])).all()
    assert (rows[..., 1] <= rows[..., 2]).all()
    points = statistics(rows)
    for i, arm in enumerate(ARMS):
        expected = [manifest['sources'][arm]['metrics'][m]*100 for m in ('CIDEr', 'CHAIRs', 'CHAIRi')]
        np.testing.assert_allclose(points[i], expected, atol=1e-10, rtol=0)
    return rows, manifest


def main():
    started = time.monotonic()
    rows, manifest = load_inputs()
    points = statistics(rows)
    draws = resample(rows, replicates=100000, seed=20261124)
    comparisons = {}
    for comparator, arm in enumerate(ARMS[1:], 1):
        metrics = {}
        for j, metric in enumerate(METRICS):
            samples = draws[:, comparator-1, j]
            ci95 = np.quantile(samples, [.025, .975], method='linear').tolist()
            adjusted = np.quantile(samples, [.05/12, 1-.05/12], method='linear').tolist()
            metrics[metric] = dict(full=float(points[0, j]), comparator=float(points[comparator, j]),
                                  difference=float(points[0, j]-points[comparator, j]),
                                  ci95_pointwise=ci95, ci_bonferroni_six=adjusted)
        a, b = rows[:, 0, 1] > 0, rows[:, comparator, 1] > 0
        comparisons[f'full_minus_{arm}'] = dict(metrics=metrics,
            paired_CHAIRs_flags=dict(both_clear=int((~a & ~b).sum()), both_flagged=int((a & b).sum()),
                comparator_flagged_full_clear=int((~a & b).sum()), comparator_clear_full_flagged=int((a & ~b).sum())))
    result = dict(status='PAIRED_TEST_ABLATION_BOOTSTRAP_COMPLETE', date='2026-11-24', images=len(rows),
        replicates=len(draws), seed=20261124, rng='PCG64', interval='percentile, linear quantile',
        family_endpoints=6, adjusted_confidence_percent=100*(1-.05/6), differences='full minus comparator',
        input_sha256=manifest['input_sha256'], protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, comparisons=comparisons,
        cpu_statistics_seconds=time.monotonic()-started, model_inference=False, metric_scorers_run=False,
        limits=['Fixed checkpoint/captions/scorers/reference IDF; image resampling only, not training-seed variance.',
                'Retrospective analysis; prior TEST development exposure. Bonferroni coverage relies on bootstrap approximation.',
                'CHAIR measures object mentions, not action/relation/attribute correctness. No p-values or equivalence test.'])
    (HERE/'RESULT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
