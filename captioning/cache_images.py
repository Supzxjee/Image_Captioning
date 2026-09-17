"""Image dataset defined at module scope for DataLoader worker spawning."""
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset


class CacheImages(Dataset):
    def __init__(self, records, base_path, transform):
        self.records = records
        self.base_path = Path(base_path)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        record = self.records[index]
        path = self.base_path / record['filepath'] / record['filename']
        try:
            with Image.open(path) as image:
                return self.transform(image.convert('RGB'))
        except Exception as error:
            raise RuntimeError(f'Cannot process cache image row {index}: {path}') from error
