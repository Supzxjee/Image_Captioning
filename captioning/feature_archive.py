"""Read a feature ZIP or its explicitly selected extracted root, without searching."""
from pathlib import Path
import zipfile


class FeatureDirectory:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def namelist(self):
        names = ['manifest.json', 'diagnostic_summary.json', 'features_0.pt', 'features_1.pt',
                 'data/train_pairs.json', 'data/val_pairs.json', 'data/pair_summary.json',
                 'data/evaluation_pairs.json', 'data/full_candidate_summary.json']
        return [name for name in names if (self.root / name).is_file()]

    def read(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Feature path escapes the supplied root.')
        return path.read_bytes()


def open_feature_archive(path):
    return FeatureDirectory(path) if Path(path).is_dir() else zipfile.ZipFile(path)
