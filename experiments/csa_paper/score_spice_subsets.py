"""User-run CPU SPICE subsets on unchanged, SHA-verified saved captions."""
import argparse
import math
from pathlib import Path

from score_prompt_test_extra import read, save, environment, smoke, tokenize, java_path

HERE = Path(__file__).resolve().parent
CATEGORIES = ('All', 'Objects', 'Relations', 'Attributes')
ALIASES = {'Object': 'Objects', 'Relation': 'Relations', 'Attribute': 'Attributes'}


def summarize(rows):
    result = {}
    for category in CATEGORIES:
        values = [r['scores'].get(category, {}).get('f') for r in rows]
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        result[category] = dict(f_x100=100 * sum(finite) / len(finite) if finite else None,
                                finite_images=len(finite), undefined_images=len(rows)-len(finite))
    return result


def clean(score):
    return {ALIASES.get(category, category): {key: float(value) if value is not None and math.isfinite(float(value)) else None
                       for key, value in part.items()} for category, part in score.items()}


def validate_categories(rows):
    for category in CATEGORIES:
        if not any(category in r['scores'] for r in rows):
            observed = sorted({k for r in rows for k in r['scores']})
            raise ValueError(f'Missing SPICE category: {category}; observed={observed}')


def subset_smoke(out):
    smoke(out)
    from pycocoevalcap.spice.spice import Spice
    point, scores = Spice().compute_score(
        {1: ['a red dog is sitting on a couch']}, {1: ['a red dog is sitting on a couch']})
    rows = [dict(coco_id=1, scores=clean(scores[0]))]
    validate_categories(rows)
    if not math.isfinite(float(point)) or point <= 0:
        raise ValueError('Subset smoke failed')
    save(out/'subset_smoke.json', dict(passed=True, categories=list(rows[0]['scores']), rows=rows))


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
        saved_path = out / (policy+'_spice_raw.json')
        if saved_path.is_file():
            raw = read(saved_path)
            if raw['tokenized_sha256'] != receipt['tokenized_sha256'] or raw['arm'] != arm or raw['policy'] != policy:
                raise ValueError('Saved SPICE identity differs')
            point, scores = raw['point'], raw['scores']
            print('Reusing saved SPICE values:', policy, flush=True)
        else:
            scorer = Spice()
            point, scores = scorer.compute_score(reference, hyp)
            # Preserve expensive scorer output before validating its category schema.
            save(saved_path, dict(arm=arm, policy=policy, point=float(point),
                tokenized_sha256=receipt['tokenized_sha256'], scores=[clean(s) for s in scores]))
        # pycocoevalcap returns per-image scores in sorted input-key order.
        if len(scores) != 5000 or not math.isfinite(float(point)):
            raise ValueError('Incomplete/nonfinite SPICE result')
        if abs(float(point)-expected[policy]['SPICE']) > 1e-7:
            raise ValueError('SPICE All historical replay differs; do not rank partial outputs')
        rows = [dict(coco_id=int(i), scores=clean(s)) for i, s in zip(sorted(reference), scores)]
        if any(r['scores'].get('All', {}).get('f') is None for r in rows):
            raise ValueError('Undefined All score')
        validate_categories(rows)
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
        if a.mode == 'smoke': subset_smoke(a.output)
        elif a.mode == 'tokenize': tokenize(a.output, a.input, a.arm, manifest)
        else: score(a.output, a.arm, manifest)
