"""Lazy HDF5 visual-token reader; no feature tensors are preloaded into RAM."""
import os
import re
from pathlib import Path


def coco_image_id(filename):
    """COCO ID from filename, independent of Karpathy row/evaluation IDs."""
    match = re.fullmatch(r'COCO_(?:train|val|test)2014_(\d+)\.jpg', Path(filename).name)
    if match is None:
        raise ValueError(f'Cannot extract COCO image ID from {filename!r}')
    return int(match.group(1))


class VisualCache:
    def __init__(self, paths, id_key="coco_id"):
        if id_key not in {"coco_id", "eval_id", "karpathy_id"}:
            raise ValueError("Invalid visual cache ID key")
        self.id_key = id_key
        import h5py
        self.paths = []
        for value in paths:
            path = Path(value)
            candidates = sorted(path.rglob('*.h5')) if path.is_dir() else [path]
            if not candidates:
                raise FileNotFoundError(f'No .h5 files in {path}')
            for candidate in candidates:
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                resolved = str(candidate.resolve())
                if resolved not in self.paths:
                    self.paths.append(resolved)
        if not self.paths:
            raise ValueError('At least one visual cache file is required.')
        self.index = {}
        self._handles = {}
        self._pid = os.getpid()
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, 'r') as f:
                if 'features' not in f or 'imgids' not in f:
                    raise ValueError(f'{path}: expected features and imgids datasets')
                features, ids = f['features'], f['imgids']
                if features.ndim != 3 or features.shape[1:] != (197, 768):
                    raise ValueError(f'{path}: expected (N, 197, 768), got {features.shape}')
                if features.dtype.kind != 'f' or features.dtype.itemsize not in (2, 4):
                    raise ValueError(f'{path}: features must be float16 or float32')
                if ids.ndim != 1 or len(ids) != len(features) or ids.dtype.kind not in 'iu':
                    raise ValueError(f'{path}: imgids must be an integer (N,) aligned with features')
                # Only the small ID table is loaded; visual features stay on disk.
                for row, value in enumerate(ids[:]):
                    image_id = int(value)
                    if image_id in self.index:
                        raise ValueError(f'Duplicate cache image ID {image_id} across cache rows')
                    self.index[image_id] = (file_index, row)

    def require_ids(self, ids, label):
        missing = sorted(set(int(i) for i in ids) - self.index.keys())
        if missing:
            raise ValueError(f'{label}: {len(missing)} images missing from visual cache; examples: {missing[:10]}')

    def read(self, image_id):
        import h5py
        import numpy as np
        image_id = int(image_id)
        if image_id not in self.index:
            raise KeyError(f'Cache image ID {image_id} is absent from visual cache')
        # Separate handles per DataLoader worker, safe for both fork and spawn.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        file_index, row = self.index[image_id]
        if file_index not in self._handles:
            self._handles[file_index] = h5py.File(self.paths[file_index], 'r')
        array = self._handles[file_index]['features'][row].astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite visual features for cache image ID {image_id}')
        return array

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = {}
        state['_pid'] = None
        return state
