"""Download annotation ZIPs linked by the official VG page. No images/models."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).parent / 'review' / 'downloads'
BASE = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(name):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / (name + '.json.zip')
    url = BASE + path.name
    if not path.is_file():
        temporary = path.with_suffix('.zip.partial')
        started = time.monotonic()
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            expected = int(response.headers['Content-Length'])
            received, mark = 0, 0
            while block := response.read(1024**2):
                output.write(block); received += len(block)
                if received - mark >= 32 * 1024**2:
                    print(name, received, '/', expected, 'bytes', flush=True); mark = received
        if received != expected:
            raise ValueError('Truncated download: ' + name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC mismatch: ' + name)
        temporary.replace(path)
        print('DOWNLOAD COMPLETE:', name, round(time.monotonic() - started, 1), 'seconds', flush=True)
    result = dict(url=url, bytes=path.stat().st_size, sha256=digest(path),
                  official_link_page='https://homes.cs.washington.edu/~ranjay/visualgenome/api.html')
    with zipfile.ZipFile(path) as archive:
        result['members'] = [dict(name=i.filename, bytes=i.file_size) for i in archive.infolist()]
    (ROOT / (name + '_provenance.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    for name in ['image_data', 'region_graphs']:
        download(name)
