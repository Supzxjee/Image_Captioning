"""USER-RUN CPU scorers on exact saved TEST captions; no model imports."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
JRE_URL = ('https://github.com/adoptium/temurin8-binaries/releases/download/'
           'jdk8u472-b08/OpenJDK8U-jre_x64_linux_hotspot_8u472b08.tar.gz')
JRE_SHA = '6f7fb5fd640a0fd00837344b0920cbc4b9b9284b50e66f33789e3b250446a16e'
KEYS = ('Bleu_1', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'CHAIRs', 'CHAIRi')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def download(url, dest):
    print('Downloading:', url, flush=True)
    total = 0
    with urllib.request.urlopen(url, timeout=90) as r, Path(dest).open('wb') as f:
        while chunk := r.read(1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            if total % (25 * 1024 * 1024) == 0:
                print('Downloaded', total // (1024 * 1024), 'MiB', flush=True)


def verify(root, arm, manifest):
    """Accept nesting, never a different payload/version or incomplete cohort."""
    spec = manifest['arms'][arm]
    files = {}
    for name, digest in spec['payload_sha256'].items():
        matches = [p for p in Path(root).rglob(name) if p.is_file() and sha(p) == digest]
        if len(matches) != 1:
            raise ValueError(f'Need one exact {name} from {spec["version"]}; found {len(matches)}')
        files[name] = matches[0]
    result, lock = read(files['result.json']), read(files['test_lock.json'])
    if (result['status'] != 'PROMPT_TEST5000_COMPLETE' or result['arm'] != arm
            or result['images'] != manifest['images'] or result['training']
            or result['checkpoint_sha256'] != spec['checkpoint_sha256']
            or lock['checkpoint_sha256'] != spec['checkpoint_sha256']
            or sha(files['test_references.json']) != lock['references_sha256']
            or read(files['run_status.json']) != dict(complete=True, arm=arm, training=False, test_tuning=False)):
        raise ValueError('Wrong source identity/status')
    refs = read(files['test_references.json'])
    ids = {r['coco_id'] for r in refs}
    if len(refs) != manifest['images'] or len(ids) != len(refs):
        raise ValueError('Duplicate/missing TEST reference IDs')
    if any(type(i) is not int for i in ids) or any(
            not isinstance(r['references'], list) or len(r['references']) < 5
            or any(not isinstance(t, str) or not t.strip() for t in r['references']) for r in refs):
        raise ValueError('Wrong reference structure')
    captions = {}
    for policy in manifest['policies']:
        rows = read(files[policy + '_captions.json'])
        if (len(rows) != len(ids) or {r['coco_id'] for r in rows} != ids
                or any(not isinstance(r['caption'], str) or not r['caption'].strip() for r in rows)
                or result['metrics'][policy] != spec['metrics'][policy]):
            raise ValueError('Changed/incomplete captions or historical metrics: ' + policy)
        captions[policy] = rows
    return refs, captions, files


def environment(out):
    from importlib.metadata import version
    if version('pycocoevalcap') != '1.2':
        raise ValueError('Expected pycocoevalcap==1.2')
    assets = out / 'assets'
    assets.mkdir()
    archive = assets / 'java8.tar.gz'
    download(JRE_URL, archive)
    if sha(archive) != JRE_SHA:
        raise ValueError('Changed pinned Java8 asset')
    with tarfile.open(archive) as t:
        t.extractall(assets / 'jre', filter='data')
    java = list((assets / 'jre').rglob('bin/java'))
    if len(java) != 1:
        raise ValueError('Missing pinned Java executable')
    spice_dir = Path(importlib.util.find_spec('pycocoevalcap.spice.spice').origin).parent
    lib = spice_dir / 'lib'
    lib.mkdir(exist_ok=True)
    archive = assets / 'corenlp.zip'
    download('https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-full-2015-12-09.zip', archive)
    for suffix in ('.jar', '-models.jar'):
        name = 'stanford-corenlp-3.6.0' + suffix
        member = 'stanford-corenlp-full-2015-12-09/' + name
        with zipfile.ZipFile(archive) as z, z.open(member) as src, (lib / name).open('wb') as dst:
            shutil.copyfileobj(src, dst)
    package = spice_dir.parent
    save(out / 'environment.json', dict(pycocoevalcap=version('pycocoevalcap'), java=str(java[0]),
        java_archive_sha256=JRE_SHA, corenlp_archive_sha256=sha(archive),
        corenlp_version='3.6.0', model_digest_pinned_independently=False,
        evaluator_files={str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*'))
                         if p.is_file() and p.suffix in ('.py', '.jar', '.gz')}))


def java_path(out):
    env = read(out / 'environment.json')
    os.environ['PATH'] = str(Path(env['java']).parent) + os.pathsep + os.environ['PATH']
    os.environ['JAVA_HOME'] = str(Path(env['java']).parent.parent)


def checked_point(value, length, scores):
    value = float(value)
    if not math.isfinite(value) or len(scores) != length:
        raise ValueError('Nonfinite metric or incomplete per-image coverage')
    return value


def smoke(out):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    tok = PTBTokenizer()
    refs = tok.tokenize({1: [dict(caption='a person riding a bicycle on a road')]})
    hyp = tok.tokenize({1: [dict(caption='a person riding a bicycle on a road')]})
    points = {'Bleu_1': float(Bleu(4).compute_score(refs, hyp)[0][0])}
    for name, factory in [('METEOR', Meteor), ('SPICE', Spice)]:
        scorer = factory()
        try:
            point, scores = scorer.compute_score(refs, hyp)
            points[name] = checked_point(point, 1, scores)
            if points[name] <= 0:
                raise ValueError('Exact-match smoke has no positive score: ' + name)
        finally:
            del scorer
    save(out / 'smoke.json', dict(passed=True, scores=points))
    print('Actual CPU PTB/BLEU/METEOR/SPICE smoke PASS', points, flush=True)


def tokenize(out, root, arm, manifest):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    refs, captions, files = verify(root, arm, manifest)
    tok = PTBTokenizer()
    reference = tok.tokenize({r['coco_id']: [dict(caption=t) for t in r['references']] for r in refs})
    save(out / 'references_ptb.json', reference)
    for policy, rows in captions.items():
        save(out / (policy + '_ptb.json'), tok.tokenize({r['coco_id']: [dict(caption=r['caption'])] for r in rows}))
    save(out / 'input_receipt.json', dict(arm=arm, source_version=manifest['arms'][arm]['version'],
        images=len(refs), payload_sha256={n: sha(p) for n, p in files.items()},
        tokenized_sha256={p.name: sha(p) for p in out.glob('*_ptb.json')},
        training=False, decoding=False, reranking=False))


def metric(out, arm, manifest, mode):
    receipt = read(out / 'input_receipt.json')
    if receipt['arm'] != arm or receipt['payload_sha256'] != manifest['arms'][arm]['payload_sha256']:
        raise ValueError('Input receipt changed')
    for n, digest in receipt['tokenized_sha256'].items():
        if sha(out / n) != digest:
            raise ValueError('Tokenized captions changed')
    reference = read(out / 'references_ptb.json')
    for policy in manifest['policies']:
        print('CPU scoring:', policy, mode, flush=True)
        hyp = read(out / (policy + '_ptb.json'))
        if len(reference) != manifest['images'] or set(reference) != set(hyp):
            raise ValueError('Metric cohort differs')
        if mode == 'basic':
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.rouge.rouge import Rouge
            b, _ = Bleu(4).compute_score(reference, hyp)
            c, cs = Cider().compute_score(reference, hyp)
            r, rs = Rouge().compute_score(reference, hyp)
            scores = dict(Bleu_1=float(b[0]), Bleu_4=float(b[3]),
                CIDEr=checked_point(c, len(hyp), cs), ROUGE_L=checked_point(r, len(hyp), rs))
            for k in ('Bleu_4', 'CIDEr', 'ROUGE_L'):
                if abs(scores[k] - manifest['arms'][arm]['metrics'][policy][k]) > 1e-7:
                    raise ValueError('Historical metric replay differs: ' + policy + '/' + k)
        else:
            if mode == 'meteor':
                from pycocoevalcap.meteor.meteor import Meteor
                scorer = Meteor()
            elif mode == 'spice':
                from pycocoevalcap.spice.spice import Spice
                scorer = Spice()
            else:
                raise ValueError(mode)
            try:
                point, per_image = scorer.compute_score(reference, hyp)
                scores = {mode.upper(): checked_point(point, len(hyp), per_image)}
            finally:
                del scorer
        if any(not math.isfinite(v) for v in scores.values()):
            raise ValueError('Nonfinite metric')
        save(out / f'{policy}_{mode}.json', dict(images=len(hyp), metrics=scores,
            tokenized_sha256=receipt['tokenized_sha256']))


def combine(out, arm, manifest):
    points = {}
    receipt = read(out / 'input_receipt.json')
    for policy in manifest['policies']:
        row = {}
        for mode in ('basic', 'meteor', 'spice'):
            part = read(out / f'{policy}_{mode}.json')
            if part['images'] != 5000 or part['tokenized_sha256'] != receipt['tokenized_sha256']:
                raise ValueError('Partial/changed metric phase')
            row.update(part['metrics'])
        row.update({k: manifest['arms'][arm]['metrics'][policy][k] for k in ('CHAIRs', 'CHAIRi')})
        if set(row) != set(KEYS) or not all(math.isfinite(v) for v in row.values()):
            raise ValueError('Need all eight metrics')
        points[policy] = row
    result = dict(status='PROMPT_TEST_EIGHT_METRICS_COMPLETE', arm=arm, images=5000,
        source_version=manifest['arms'][arm]['version'], metrics=points,
        display_x100={p: {k: 100*v for k, v in row.items()} for p, row in points.items()},
        chair_provenance='Reused audited scores, exact caption/reference/result SHA verified; not recomputed.',
        inputs=receipt, environment=read(out / 'environment.json'), training=False, decoding=False,
        reranking=False, test_tuning=False, prior_project_test_exposure=True)
    save(out / 'result.json', result)
    with (out / 'metrics_x100.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'policy', *KEYS])
        w.writeheader()
        for policy, row in result['display_x100'].items():
            w.writerow(dict(arm=arm, policy=policy, **row))
    print('COMPLETE — saved captions only:', result['display_x100'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['environment', 'smoke', 'tokenize', 'basic', 'meteor', 'spice', 'combine'])
    p.add_argument('--arm', choices=['CONTROL', 'PROMPT'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = read(HERE / 'PROMPT_TEST_EXTRA_INPUTS.json')
    if args.mode == 'environment':
        environment(args.output)
    else:
        java_path(args.output)
        if args.mode == 'smoke':
            smoke(args.output)
        elif args.mode == 'tokenize':
            tokenize(args.output, args.input, args.arm, manifest)
        elif args.mode == 'combine':
            combine(args.output, args.arm, manifest)
        else:
            metric(args.output, args.arm, manifest, args.mode)
