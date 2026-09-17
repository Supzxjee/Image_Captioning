"""COCO splits, prompt cache, image preprocessing and training DataLoader."""
import json
import os
from types import SimpleNamespace
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .tokenizer import CaptionTokenizer
from .visual_cache import VisualCache, coco_image_id

NORMALIZATION_STATS = {
    'mean': [0.48145466, 0.4578275, 0.40821073],
    'std': [0.26862954, 0.26130258, 0.27577711],
}

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_STATS['mean'], std=NORMALIZATION_STATS['std']),
])


class CocoPromptDataset(Dataset):
    def __init__(self, dataframe, transform, caption_tokenizer, prompt_cache, visual_cache=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.caption_tokenizer = caption_tokenizer
        self.prompt_cache = prompt_cache
        self.visual_cache = visual_cache

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image']
        if image_path not in self.prompt_cache:
            raise KeyError(f'Missing prompt embedding for: {image_path}')

        image = load_visual_input(row, self.transform, self.visual_cache)
        caption_ids, caption_padding_masks = [], []
        for caption in row['captions']:
            ids, mask = self.caption_tokenizer.encode(caption)
            caption_ids.append(ids)
            caption_padding_masks.append(mask)

        prompt_entry = self.prompt_cache[image_path]
        return (
            image,
            torch.stack(caption_ids),
            torch.stack(caption_padding_masks),
            prompt_entry['tokens'].float(),
            prompt_entry['mask'].long(),
        )

def load_data(config):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for path in (config.dataset_json_path, config.prompt_cache_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with open(config.dataset_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    train_data, val_data, test_data = [], [], []
    for coco_image_id, img in enumerate(coco_data['images']):
        full_image_path = os.path.join(config.base_path, img['filepath'], img['filename'])
        captions = [sent['raw'] for sent in img['sentences']][:5]
        item = {
            'image': full_image_path,
            'captions': captions,
            'eval_id': coco_image_id,
            'filename': img['filename'],
            'coco_id': coco_image_id(img['filename']),
        }
        if img['split'] in ['train', 'restval']:
            train_data.append(item)
        elif img['split'] == 'val':
            val_data.append(item)
        elif img['split'] == 'test':
            test_data.append(item)

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')


    all_train_captions = [caption for item in train_data for caption in item['captions']]
    caption_tokenizer = CaptionTokenizer(all_train_captions)
    vocab_size = len(caption_tokenizer.word2idx)
    print(f'Caption vocabulary: {vocab_size:,}')

    prompt_embedding_bundle = torch.load(config.prompt_cache_path, map_location='cpu', weights_only=False)
    prompt_embedding_cache = prompt_embedding_bundle['data']
    print(f'Prompt embedding entries: {len(prompt_embedding_cache):,}')

    if config.test_after_train and len(test_df) == 0:
        raise ValueError('Test split is empty.')
    visual_cache = VisualCache(config.visual_cache) if config.visual_cache else None
    if visual_cache is not None:
        for label, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            count = sum(int(i) in visual_cache.index for i in df.get('coco_id', []))
            print(f'Visual cache coverage {label}: {count}/{len(df)}', flush=True)
        selected = train_df if config.mode == 'train' else (val_df if config.split == 'val' else test_df)
        if config.mode != 'train' and config.limit:
            selected = selected.head(config.limit)
        visual_cache.require_ids(selected.get('coco_id', []), config.mode)
        if config.test_after_train:
            if len(test_df) == 0:
                raise ValueError('Test split is empty.')
            visual_cache.require_ids(test_df['coco_id'], 'full test after train')
        print('Using cached CLIP tokens; projection, attention, gate and decoder remain trainable.', flush=True)
    return SimpleNamespace(visual_cache=visual_cache, train_df=train_df, val_df=val_df, test_df=test_df,
                           tokenizer=caption_tokenizer, prompt_cache=prompt_embedding_cache,
                           transform=image_transform, vocab_size=vocab_size)


def build_train_loader(config, data):
    dataset = CocoPromptDataset(data.train_df, data.transform, data.tokenizer, data.prompt_cache, data.visual_cache)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        num_workers=config.num_workers, pin_memory=config.device == 'cuda')
    if len(loader) == 0:
        raise ValueError('Training split is too small for the batch size with drop_last=True.')
    print(f'Train batches per epoch: {len(loader):,}')
    return loader


def load_visual_input(row, transform, visual_cache=None):
    if visual_cache is not None:
        return torch.from_numpy(visual_cache.read(row['coco_id']))
    with Image.open(row['image']) as image:
        return transform(image.convert('RGB'))
