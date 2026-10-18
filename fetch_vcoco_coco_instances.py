"""Fetch official COCO train instance metadata for V-COCO train/val IDs only.

HTTP ranges avoid downloading captions/keypoints/validation members of the ZIP.
Requires ijson for bounded-memory JSON parsing; never downloads image pixels.
"""
import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from audit_vcoco_alignment_data import SOURCE_SHA256, digest


URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
MEMBER = 'annotations/instances_train2014.json'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as response:
            self.length = int(response.headers['Content-Length'])
            self.etag = response.headers['ETag']
        self.position, self.transferred = 0, 0
        self.block_size = 4 * 1024**2
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if whence not in (0, 1, 2) or value < 0:
            raise ValueError('Invalid seek.')
        self.position = value
        return value

    def read(self, size=-1):
        size = max(0, self.length - self.position) if size < 0 else min(size, max(0, self.length - self.position))
        result = bytearray()
        while size:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                end = min(start + self.block_size, self.length) - 1
                request = urllib.request.Request(self.url, headers={
                    'Range': f'bytes={start}-{end}', 'If-Match': self.etag,
                })
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f'bytes {start}-{end}/{self.length}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor the exact HTTP range; refusing full ZIP fallback.')
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise ValueError('Incomplete range response.')
                self.transferred += len(data)
                self.cache[index] = data
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            offset = self.position % self.block_size
            data = self.cache[index][offset:offset + size]
            if not data:
                raise EOFError('Empty cached range.')
            result.extend(data)
            self.position += len(data)
            size -= len(data)
        return bytes(result)


def main():
    import ijson
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sidecar = args.output.with_suffix('.provenance.json')
    temporary = args.output.with_suffix('.raw.partial')
    if any(path.exists() for path in (args.output, sidecar, temporary)):
        raise FileExistsError('Use fresh output paths; no existing metadata is overwritten.')
    ids = set()
    for split in ('train', 'val'):
        source = args.sources / f'vcoco_{split}.json'
        if digest(source) != SOURCE_SHA256['vcoco_' + split]:
            raise ValueError('V-COCO source identity differs.')
        ids.update(iid for action in json.loads(source.read_text()) for iid in action['image_id'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(URL)
    with remote, zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(MEMBER)
        print(f'Official COCO metadata: reading one ZIP member ({info.compress_size/1024**2:.1f} MiB compressed).', flush=True)
        copied, last = 0, time.monotonic()
        with archive.open(info) as reader, temporary.open('xb') as writer:
            while block := reader.read(1024**2):
                writer.write(block)
                copied += len(block)
                if time.monotonic() - last >= 10:
                    print(f'Extracted {copied/1024**2:.1f}/{info.file_size/1024**2:.1f} MiB; network {remote.transferred/1024**2:.1f} MiB', flush=True)
                    last = time.monotonic()
        if copied != info.file_size:
            raise ValueError('ZIP member size differs.')
        # ZipExtFile also verifies the official member CRC when reading to EOF.
    result = {}
    for section in ('images', 'annotations', 'categories'):
        print('Filtering', section, flush=True)
        with temporary.open('rb') as stream:
            items = ijson.items(stream, section + '.item', use_float=True)
            if section == 'annotations':
                result[section] = [{key: row[key] for key in ('id', 'image_id', 'category_id', 'bbox', 'iscrowd')}
                                   for row in items if row['image_id'] in ids]
            elif section == 'images':
                result[section] = [{key: row[key] for key in ('id', 'file_name', 'width', 'height')}
                                   for row in items if row['id'] in ids]
            else:
                result[section] = list(items)
    if {row['id'] for row in result['images']} != ids:
        raise ValueError('Not all V-COCO train/val images are in official COCO train metadata.')
    provenance = {
        'url': URL, 'etag': remote.etag, 'archive_bytes': remote.length,
        'member': MEMBER, 'member_crc32': f'{info.CRC:08x}',
        'member_compressed_bytes': info.compress_size, 'member_uncompressed_bytes': copied,
        'uncompressed_member_sha256': digest(temporary), 'network_bytes': remote.transferred,
        'ijson_version': ijson.__version__, 'image_count': len(result['images']),
        'annotation_count': len(result['annotations']), 'image_pixels_downloaded': False,
    }
    args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    provenance['output_sha256'] = digest(args.output)
    sidecar.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    # Remove only this run's newly created temporary uncompressed metadata file.
    temporary.unlink()
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
