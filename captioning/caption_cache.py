"""Lazy reader for frozen CLIP embeddings of the five reference captions."""
import os
from pathlib import Path


class CaptionEmbeddingCache:
    def __init__(self, path):
        import h5py
        self.path = str(Path(path).resolve())
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)
        self.index = {}
        self._handle = None
        self._pid = os.getpid()
        with h5py.File(self.path, 'r') as handle:
            if not bool(handle.attrs.get('complete', False)):
                raise ValueError(f'{self.path}: caption embedding cache is incomplete')
            if 'features' not in handle or 'imgids' not in handle:
                raise ValueError(f'{self.path}: expected features and imgids')
            features, imgids = handle['features'], handle['imgids']
            if features.ndim != 3 or features.shape[1:] != (5, 512):
                raise ValueError(f'{self.path}: expected (N, 5, 512), got {features.shape}')
            if features.dtype.kind != 'f' or features.dtype.itemsize not in (2, 4):
                raise ValueError(f'{self.path}: caption features must be float16 or float32')
            if imgids.ndim != 1 or len(imgids) != len(features) or imgids.dtype.kind not in 'iu':
                raise ValueError(f'{self.path}: imgids must align with features')
            for row, value in enumerate(imgids[:]):
                image_id = int(value)
                if image_id in self.index:
                    raise ValueError(f'Duplicate caption-cache image ID {image_id}')
                self.index[image_id] = row

    def require_ids(self, ids, label):
        missing = sorted(set(int(value) for value in ids) - self.index.keys())
        if missing:
            raise ValueError(f'{label}: {len(missing)} images missing from caption cache; '
                             f'examples: {missing[:10]}')

    def read(self, image_id):
        import h5py
        import numpy as np
        image_id = int(image_id)
        if image_id not in self.index:
            raise KeyError(f'Caption cache image ID {image_id} is absent')
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._handle is None:
            self._handle = h5py.File(self.path, 'r')
        array = self._handle['features'][self.index[image_id]].astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite caption features for image ID {image_id}')
        return array

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        state['_pid'] = None
        return state
