"""Check exact mounted checkpoint bytes without importing Torch or loading pickle."""
import hashlib
import json
from pathlib import Path


def resolve_mounted(root, expected_sha, report_path=None):
    root = Path(root)
    files = [root] if root.is_file() else sorted(
        (p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.bin','.pt','.pth','.ckpt')
         and p.name != 'cache_metadata_zip.bin'),
        key=lambda p: (p.name != 'best.bin', 'best' not in p.name.lower(), str(p))) if root.is_dir() else []
    report = dict(root=str(root), expected_sha256=expected_sha, model_loaded=False,
                  checked=[], matched_file=None,
                  extracted_torch_directories=[str(p.parent) for p in root.rglob('data.pkl')] if root.is_dir() else [])
    for path in files:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        row = dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
        report['checked'].append(row)
        print('Checkpoint input:', json.dumps(row), flush=True)
        if digest == expected_sha:
            report['matched_file'] = str(path.resolve())
            break
    if report_path:
        Path(report_path).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if report['matched_file'] is None:
        print(json.dumps(report,indent=2),flush=True)
        hint = ' Torch archive appears extracted; attach the intact checkpoint file.' if report['extracted_torch_directories'] else ''
        raise FileNotFoundError('No mounted checkpoint matches the locked source SHA; no model loaded.'+hint)
    return Path(report['matched_file'])
