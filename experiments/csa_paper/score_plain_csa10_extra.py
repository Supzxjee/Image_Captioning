"""USER-RUN baseline captions; reuse the CPU scorers verified at output185/62."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import score_prompt_test_extra as cpu

HERE = Path(__file__).resolve().parent


def load_snapshot(path, manifest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['snapshot_sha256']:
        raise ValueError('Changed frozen caption snapshot')
    raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() != manifest['snapshot_json_sha256']:
        raise ValueError('Changed uncompressed snapshot')
    rows = json.loads(raw)
    ids = [r['coco_id'] for r in rows]
    if len(rows) != manifest['images'] or len(set(ids)) != len(rows) or ids != sorted(ids):
        raise ValueError('Incomplete/duplicate/reordered TEST cohort')
    for r in rows:
        if (type(r['coco_id']) is not int or len(r['references']) < 5
                or any(not isinstance(t, str) or not t.strip() for t in r['references'])
                or any(not isinstance(r[p], str) or not r[p].strip() for p in manifest['policies'])):
            raise ValueError('Invalid original reference/caption structure')
    return rows


def tokenize(out, manifest):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    path = HERE/'plain_csa10_test_snapshot.json.gz'
    rows = load_snapshot(path, manifest)
    tok = PTBTokenizer()
    cpu.save(out/'references_ptb.json', tok.tokenize({r['coco_id']: [dict(caption=t) for t in r['references']] for r in rows}))
    for policy in manifest['policies']:
        cpu.save(out/(policy+'_ptb.json'), tok.tokenize({r['coco_id']: [dict(caption=r[policy])] for r in rows}))
    cpu.save(out/'input_receipt.json', dict(arm='BASELINES', images=len(rows),
        source_version=manifest['arms']['BASELINES']['version'],
        payload_sha256={path.name:cpu.sha(path)}, historical_sources=manifest['sources'],
        tokenized_sha256={p.name:cpu.sha(p) for p in out.glob('*_ptb.json')},
        training=False, decoding=False, reranking=False))


def combine(out, manifest):
    # Same numerical aggregation as the already completed CPU scoring runs.
    cpu.combine(out, 'BASELINES', manifest)
    r = cpu.read(out/'result.json')
    r.update(status='PLAIN_CSA10_TEST_EIGHT_METRICS_COMPLETE',
             historical_sources=manifest['sources'], equal_xe_epochs=10,
             limitations=['Single seed; encoder capacity differs; equal epochs is not compute matching.',
                          'NoCSA has many short captions; lower CHAIRs alone is not higher caption quality.',
                          'CHAIR reused from exact archived-caption audit, not recomputed.'])
    cpu.save(out/'result.json', r)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['environment','smoke','tokenize','basic','meteor','spice','combine'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = cpu.read(HERE/'PLAIN_CSA10_EXTRA_INPUTS.json')
    load_snapshot(HERE/'plain_csa10_test_snapshot.json.gz', m)
    if a.mode == 'environment':
        cpu.environment(a.output)
    else:
        cpu.java_path(a.output)
        if a.mode == 'smoke':
            cpu.smoke(a.output)
        elif a.mode == 'tokenize':
            tokenize(a.output, m)
        elif a.mode == 'combine':
            combine(a.output, m)
        else:
            cpu.metric(a.output, 'BASELINES', m, a.mode)
