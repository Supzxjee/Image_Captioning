"""Saved VALID-only pilot comparison; NumPy imported only for user-run bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path

POLICY = 'clip_object_fixed'
METRICS = ('BLEU_1', 'BLEU_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'CHAIRs', 'CHAIRi', 'Recall', 'NLL')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def gates(xe, control, kl):
    return dict(CIDEr_vs_XE=kl['CIDEr'] >= xe['CIDEr']+.002,
                CHAIRs_vs_XE=kl['CHAIRs'] <= xe['CHAIRs'],
                CHAIRi_vs_XE=kl['CHAIRi'] <= xe['CHAIRi'],
                CIDEr_retained_vs_control=kl['CIDEr'] >= control['CIDEr']-.002,
                CHAIRs_reduced_vs_control=kl['CHAIRs'] <= control['CHAIRs']-.002,
                CHAIRi_vs_control=kl['CHAIRi'] <= control['CHAIRi'])


def sample_rows(root, name):
    folder = Path(root)/name/'rerank'
    per = {r['coco_id']: r['CIDEr'] for r in read(folder/(POLICY+'_cider_per_image.json'))}
    chair = {r['image_id']: r for r in read(folder/(POLICY+'_chair_output.json'))['sentences']}
    if len(per) != 5000 or set(per) != set(chair):
        raise ValueError('Wrong bootstrap VALID IDs')
    return {i: (per[i], bool(chair[i]['mscoco_hallucinated_words']),
                len(chair[i]['mscoco_hallucinated_words']), len(chair[i]['mscoco_generated_words'])) for i in per}


def bootstrap(left, right, replicates=4000):
    import numpy as np
    if set(left) != set(right):
        raise ValueError('Bootstrap cohorts differ')
    ids = sorted(left)
    a = np.asarray([left[i] for i in ids], dtype=np.float64)
    b = np.asarray([right[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(20261119)
    values = {k: [] for k in ('CIDEr_x100', 'CHAIRs_percentage_points', 'CHAIRi_percentage_points')}
    for start in range(0, replicates, 100):
        take = rng.integers(len(ids), size=(min(100, replicates-start), len(ids)))
        x, y = a[take], b[take]
        values['CIDEr_x100'].extend(100*(x[..., 0]-y[..., 0]).mean(1))
        values['CHAIRs_percentage_points'].extend(100*(x[..., 1]-y[..., 1]).mean(1))
        values['CHAIRi_percentage_points'].extend(100*(x[..., 2].sum(1)/np.maximum(1,x[..., 3].sum(1))-
            y[..., 2].sum(1)/np.maximum(1,y[..., 3].sum(1))))
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}


def run(root):
    root = Path(root)
    results = {name: read(root/name/'result.json') for name in ('xe', 'scst500', 'kl500')}
    if results['kl500']['checkpoint'].get('kl_beta') != 1.0:
        raise ValueError('Pilot beta differs from the locked protocol')
    seals = [hashlib.sha256((root/name/'test_references.json').read_bytes()).hexdigest() for name in results]
    if len(set(seals)) != 1 or any(r['images'] != 5000 or r['test_used'] for r in results.values()):
        raise ValueError('Pilot must use identical VALID references, no TEST')
    metrics = {name: r['metrics'][POLICY] for name, r in results.items()}
    screen = gates(metrics['xe'], metrics['scst500'], metrics['kl500'])
    rows = {name: sample_rows(root, name) for name in results}
    result = dict(status='SCST_KL500_VALID_COMPLETE', seed=43, split='val', images=5000,
        test_used=False, training=True, updates=500, beta=1.0,
        arms=results, display_x100={name: {k: v[k]*(1 if k=='NLL' else 100) for k in METRICS} for name,v in metrics.items()},
        delta_vs_XE={k: metrics['kl500'][k]-metrics['xe'][k] for k in METRICS},
        delta_vs_SCST500={k: metrics['kl500'][k]-metrics['scst500'][k] for k in METRICS},
        gates=screen, screen_passed=all(screen.values()),
        paired_bootstrap_95_ci={name: bootstrap(rows['kl500'], rows[name]) for name in ('xe', 'scst500')},
        bootstrap_note='4000 paired image resamples, fixed corpus CIDEr IDF; conditional on one trained seed and reused VALID, not training-seed uncertainty.',
        no_automatic_extension=True, original_XE_replay=read(root/'xe_replay.json'))
    (root/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(gates=screen, screen_passed=all(screen.values()), metrics=metrics), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
