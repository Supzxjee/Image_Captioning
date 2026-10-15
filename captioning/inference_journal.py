"""Resume ordered inference records only under an unchanged input protocol."""
import json
from pathlib import Path


class InferenceJournal:
    def __init__(self, path, protocol, expected_ids):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = self.path.with_suffix('.protocol.json')
        if stamp.exists():
            if json.loads(stamp.read_text(encoding='utf-8')) != protocol:
                raise ValueError('Inference inputs changed; use a fresh output directory.')
        else:
            if self.path.exists(): raise ValueError('Journal has no provenance.')
            stamp.write_text(json.dumps(protocol,sort_keys=True),encoding='utf-8')
        self.expected_ids = list(expected_ids)
        self.records = []
        if self.path.exists():
            raw = self.path.read_bytes(); lines = raw.splitlines(keepends=True)
            good = 0
            for i, line in enumerate(lines):
                try: value = json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    if i != len(lines)-1 or line.endswith(b'\n'): raise ValueError('Corrupt journal.')
                    # A killed process can leave only its final write incomplete.
                    with self.path.open('r+b') as stream: stream.truncate(good)
                    break
                self.records.append(value); good += len(line)
            if self.records and self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
                with self.path.open('ab') as stream: stream.write(b'\n')
        if [r['image_id'] for r in self.records] != self.expected_ids[:len(self.records)]:
            raise ValueError('Journal IDs differ from ordered shard.')

    def append(self, record):
        i = len(self.records)
        if i >= len(self.expected_ids) or record['image_id'] != self.expected_ids[i]:
            raise ValueError('Duplicate, missing or reordered inference record.')
        line = json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'
        with self.path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(line); stream.flush()
        self.records.append(record)
