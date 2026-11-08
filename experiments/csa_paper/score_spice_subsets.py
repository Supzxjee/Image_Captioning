"""User-run CPU SPICE subsets on unchanged, SHA-verified saved captions."""
import argparse
import math
from pathlib import Path

from score_prompt_test_extra import read, save, environment, smoke, tokenize, java_path

HERE = Path(__file__).resolve().parent
CATEGORIES = ('All', 'Objects', 'Relations', 'Attributes')


def summarize(rows):
    result = {}
    for category in CATEGORIES:
        values = [r['scores'].get(category, {}).get('f') for r in rows]
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        result[category] = dict(f_x100=100 * sum(finite) / len(finite) if finite else None,
                                finite_images=len(finite), undefined_images=len(rows)-len(finite))
    return result


def clean(score):
    return {category: {key: float(value) if math.isfinite(float(value)) else None
                       for key, value in part.items()} for category, part in score.items()}


def score(out, arm, manifest):
    from pycocoevalcap.spice.spice import Spice
    receipt = read(out / 'input_receipt.json')
    from score_prompt_test_extra import sha
    if receipt['arm'] != arm or receipt['payload_sha256'] != manifest['arms'][arm]['payload_sha256']:
        raise ValueError('Changed input identity')
    for name, digest in receipt['tokenized_sha256'].items():
        if sha(out / name) != digest:
            raise ValueError('Changed tokenized input: '+name)
    reference = read(out / 'references_ptb.json')
    expected = read(HERE / 'PROMPT_TEST_EXTRA_RESULT.json')['arms'][arm]['metrics']
    points = {}
    for policy in manifest['policies']:
        hyp = read(out / (policy+'_ptb.json'))
        if len(reference) != 5000 or set(reference) != set(hyp):
            raise ValueError('Incomplete TEST cohort')
        print('SPICE subsets:', arm, policy, '5000 saved captions', flush=True)
        scorer = Spice()
        point, scores = scorer.compute_score(reference, hyp)
        # pycocoevalcap returns per-image scores in sorted input-key order.
        if len(scores) != 5000 or not math.isfinite(float(point)):
            raise ValueError('Incomplete/nonfinite SPICE result')
        if abs(float(point)-expected[policy]['SPICE']) > 1e-7:
            raise ValueError('SPICE All historical replay differs; do not rank partial outputs')
        rows = [dict(coco_id=int(i), scores=clean(s)) for i, s in zip(sorted(reference), scores)]
        if any(r['scores'].get('All', {}).get('f') is None for r in rows):
            raise ValueError('Undefined All score')
        for category in CATEGORIES:
            if not any(category in r['scores'] for r in rows):
                raise ValueError('Missing SPICE category: '+category)
        save(out / (policy+'_spice_per_image.json'), rows)
        points[policy] = summarize(rows)
    save(out/'result.json', dict(status='SPICE_SUBSETS_COMPLETE', arm=arm, images=5000,
         subsets=points, source_version=manifest['arms'][arm]['version'], inputs=receipt,
         training=False, decoding=False, reranking=False, test_tuning=False,
         undefined_policy='NaN -> null, excluded per category; denominators reported',
         limitations=['One seed; C10/P9 and adapter capacity differ; previous TEST exposure.',
                      'SPICE subsets measure reference scene-graph overlap, not box grounding.']))
    print('SPICE SUBSETS COMPLETE:', points, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['environment', 'smoke', 'tokenize', 'spice'])
    p.add_argument('--arm', required=True, choices=['CONTROL', 'PROMPT'])
    p.add_argument('--input', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = read(HERE/'PROMPT_TEST_EXTRA_INPUTS.json')
    if a.mode == 'environment':
        environment(a.output)
    else:
        java_path(a.output)
        if a.mode == 'smoke': smoke(a.output)
        elif a.mode == 'tokenize': tokenize(a.output, a.input, a.arm, manifest)
        else: score(a.output, a.arm, manifest)
