"""Small data-only helpers. Safe to test without importing any neural library."""
import hashlib
import json
from pathlib import Path
import zipfile

MARKER = 'semantic_response_confirmed15_manifest.json'
BUNDLE = 'semantic_response_confirmed15_assets.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def safe_path(root, name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('Unsafe asset name')
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or Path(name).is_absolute():
        raise ValueError('Asset outside bundle')
    return path


def assets(input_root, temporary, receipt):
    input_root = Path(input_root)
    manifests = list(input_root.rglob(MARKER))
    if len(manifests) > 1:
        raise ValueError('Multiple response15 bundles: set ASSETS_INPUT to one dataset root')
    if not manifests:
        archives = list(input_root.rglob(BUNDLE))
        if len(archives) != 1 or sha(archives[0]) != receipt['sha256']:
            raise ValueError('Add the exact response15 asset ZIP as Kaggle Input')
        root = Path(temporary)
        root.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archives[0]) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or sum(i.file_size for i in z.infolist()) > 12 * 1024**2:
                raise ValueError('Duplicate/oversized asset archive')
            for info in z.infolist():
                target = safe_path(root, info.filename)
                if info.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(z.read(info))
        manifests = [root / MARKER]
    return checked_assets(manifests[0].parent, receipt)


def checked_assets(root, receipt):
    root = Path(root)
    if sha(root / MARKER) != receipt['manifest_sha256']:
        raise ValueError('Wrong response15 manifest')
    m = json.loads((root / MARKER).read_bytes())
    rows = m['pairs']
    if (m['schema'] != 'semantic_response_confirmed15_assets_v1' or len(rows) != 15
            or len({r['vg_id'] for r in rows}) != 15 or len({r['coco_id'] for r in rows}) != 15
            or len(m['files']) != 30 or not all(r['human_confirmed'] for r in rows)
            or m['vocab_sha256'] != receipt['vocab_sha256']):
        raise ValueError('Wrong response15 identity/count/confirmation')
    for name, want in m['files'].items():
        if sha(safe_path(root, name)) != want:
            raise ValueError('Missing or changed image: ' + name)
    for r in rows:
        for side in ('original', 'edited'):
            if m['files'].get(r[side + '_file']) != r[side + '_sha256']:
                raise ValueError('Pair image hash differs from manifest')
    return root, m


def contrast(a, b, vocab):
    """Matched context, separate teacher-forced tails; never align diverged prefixes."""
    aa, bb = a.split(), b.split()
    p = 0
    while p < min(len(aa), len(bb)) and aa[p] == bb[p]:
        p += 1
    if p == 0 or p == min(len(aa), len(bb)):
        raise ValueError('Require shared prefix and two nonempty distinct tails')
    out = []
    for words in (aa, bb):
        if any(w not in vocab or vocab[w] < 4 for w in words) or len(words) + 2 > 30:
            raise ValueError('OOV/overlength fact; do not substitute')
        out.append(dict(ids=[vocab['<bos>']] + [vocab[w] for w in words], start=p,
                        count=len(words)-p))
    return out


def summary(records):
    if not records:
        raise ValueError('No records')
    return dict(pairs=len(records),
        changed_both_ranked_correct=sum(r['changed_d_original'] > 0 and r['changed_d_edited'] < 0 for r in records),
        changed_direction_correct=sum(r['changed_d_edited'] < r['changed_d_original'] for r in records),
        stable_both_ranked_correct=sum(r['stable_d_original'] > 0 and r['stable_d_edited'] > 0 for r in records),
        greedy_caption_text_changed=sum(r['caption_original'] != r['caption_edited'] for r in records),
        mean_abs_stable_delta=sum(abs(r['stable_d_edited']-r['stable_d_original']) for r in records)/len(records),
        caption_semantic_accuracy='NOT_AUTOMATICALLY_SCORED; inspect captions with images',
        interpretation='Descriptive15-pair diagnostic; no CIDEr/CHAIR gain or independent-test claim')
