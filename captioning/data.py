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
from .caption_cache import CaptionEmbeddingCache
from .semantic_prompts import align_prompt_cache

NORMALIZATION_STATS = {
    'mean': [0.48145466, 0.4578275, 0.40821073],
    'std': [0.26862954, 0.26130258, 0.27577711],
}


def _log(config, message):
    if getattr(config, 'is_main_process', True):
        print(message, flush=True)

def build_image_transform(preprocessing='bilinear'):
    interpolation = (transforms.InterpolationMode.BICUBIC if preprocessing == 'bicubic'
                     else transforms.InterpolationMode.BILINEAR)
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_STATS['mean'], std=NORMALIZATION_STATS['std']),
    ])



class CocoPromptDataset(Dataset):
    def __init__(self, dataframe, transform, caption_tokenizer, prompt_cache, visual_cache=None,
                 region_targets=None, max_regions=10, caption_embedding_cache=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.caption_tokenizer = caption_tokenizer
        self.prompt_cache = prompt_cache
        self.visual_cache = visual_cache
        self.region_targets = region_targets
        self.max_regions = max_regions
        self.caption_embedding_cache = caption_embedding_cache

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
        sample = (
            image,
            torch.stack(caption_ids),
            torch.stack(caption_padding_masks),
            prompt_entry['tokens'].float(),
            prompt_entry['mask'].long(),
        )
        if self.caption_embedding_cache is not None:
            sample += (torch.from_numpy(self.caption_embedding_cache.read(row['coco_id'])),)
        if self.region_targets is None:
            return sample
        target = self.region_targets.get(row['filename'])
        if target is None:
            raise KeyError(f'Missing region targets for: {row["filename"]}')
        boxes = torch.zeros(self.max_regions, 4, dtype=torch.float32)
        labels = torch.full((self.max_regions,), -1, dtype=torch.long)
        confidences = torch.zeros(self.max_regions, dtype=torch.float32)
        count = min(self.max_regions, len(target['labels']))
        if count:
            boxes[:count] = torch.as_tensor(target['boxes'][:count], dtype=torch.float32)
            labels[:count] = torch.as_tensor(target['labels'][:count], dtype=torch.long)
            confidences[:count] = torch.as_tensor(target['confidences'][:count], dtype=torch.float32)
        return sample + (boxes, labels, confidences)

def load_data(config):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for path in (config.dataset_json_path, config.prompt_cache_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with open(config.dataset_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    train_data, val_data, test_data = [], [], []
    for eval_id, img in enumerate(coco_data['images']):
        sentence_ids = {sent['imgid'] for sent in img['sentences'] if 'imgid' in sent}
        karpathy_id = img.get('imgid', next(iter(sentence_ids)) if len(sentence_ids) == 1 else None)
        if len(sentence_ids) > 1 or (sentence_ids and karpathy_id not in sentence_ids):
            raise ValueError(f'Inconsistent Karpathy imgid for {img["filename"]}')
        full_image_path = os.path.join(config.base_path, img['filepath'], img['filename'])
        captions = [sent['raw'] for sent in img['sentences']][:5]
        item = {
            'image': full_image_path,
            'captions': captions,
            'eval_id': eval_id,
            'filename': img['filename'],
            'coco_id': coco_image_id(img['filename']),
            'karpathy_id': karpathy_id,
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

    _log(config, f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')


    all_train_captions = [caption for item in train_data for caption in item['captions']]
    caption_tokenizer = CaptionTokenizer(all_train_captions)
    vocab_size = len(caption_tokenizer.word2idx)
    _log(config, f'Caption vocabulary: {vocab_size:,}')

    prompt_embedding_bundle = torch.load(config.prompt_cache_path, map_location='cpu', weights_only=False)
    prompt_embedding_cache = align_prompt_cache(prompt_embedding_bundle['data'],
        [item['image'] for item in train_data + val_data + test_data])
    _log(config, f'Prompt embedding entries: {len(prompt_embedding_cache):,}')
    selected_prompts = train_data if config.mode == 'train' else (val_data if config.split == 'val' else test_data)
    if config.mode != 'train' and config.limit:
        selected_prompts = selected_prompts[:config.limit]
    if config.test_after_train:
        selected_prompts = selected_prompts + test_data
    missing_prompts = [item['filename'] for item in selected_prompts
                       if item['image'] not in prompt_embedding_cache]
    if missing_prompts:
        raise ValueError(f'Missing {len(missing_prompts)} prompt embeddings; examples: {missing_prompts[:10]}')

    if config.test_after_train and len(test_df) == 0:
        raise ValueError('Test split is empty.')
    image_transform = build_image_transform(config.visual_preprocessing)
    if config.visual_cache and config.visual_cache_id_key == 'karpathy_id':
        for df in (train_df, val_df, test_df):
            if len(df) and df['karpathy_id'].isna().any():
                raise ValueError('Missing Karpathy imgid in JSON; do not substitute row order.')
    visual_cache = VisualCache(config.visual_cache, id_key=config.visual_cache_id_key) if config.visual_cache else None
    if visual_cache is not None:
        for label, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            count = sum(int(i) in visual_cache.index for i in df.get(config.visual_cache_id_key, []))
            _log(config, f'Visual cache coverage {label}: {count}/{len(df)}')
        selected = train_df if config.mode == 'train' else (val_df if config.split == 'val' else test_df)
        if config.mode != 'train' and config.limit:
            selected = selected.head(config.limit)
        visual_cache.require_ids(selected.get(config.visual_cache_id_key, []), config.mode)
        if config.test_after_train and config.test_visual_source == 'same':
            if len(test_df) == 0:
                raise ValueError('Test split is empty.')
            visual_cache.require_ids(test_df[config.visual_cache_id_key], 'full test after train')
        _log(config, 'Using cached CLIP tokens; projection, attention, gate and decoder remain trainable.')
    caption_embedding_cache = None
    if config.mode == 'train' and config.itc_weight > 0:
        caption_embedding_cache = CaptionEmbeddingCache(config.caption_embedding_cache_path)
        caption_embedding_cache.require_ids(train_df['coco_id'], 'ITC training')
        _log(config, f'Caption embedding cache: {len(caption_embedding_cache.index):,} images | '
                     f'ITC weight={config.itc_weight}')

    region_targets, region_metadata, label_prototypes = None, {}, None
    if config.mode == 'train' and config.alignment_weight > 0:
        if not os.path.isfile(config.region_targets_path):
            raise FileNotFoundError(config.region_targets_path)
        region_bundle = torch.load(config.region_targets_path, map_location='cpu', weights_only=False)
        region_targets = region_bundle['data']
        region_metadata = region_bundle['metadata']
        label_prototypes = region_bundle['label_prototypes'].float()
        missing_regions = [item['filename'] for item in train_data if item['filename'] not in region_targets]
        if missing_regions:
            raise ValueError(f'Missing {len(missing_regions)} region targets; examples: {missing_regions[:10]}')
        if label_prototypes.ndim != 2 or label_prototypes.size(1) != 512:
            raise ValueError(f'Expected label prototypes (classes, 512), got {label_prototypes.shape}')
        _log(config, f'Region targets: {len(region_targets):,} images | '
                     f'{label_prototypes.size(0)} labels | lambda={config.alignment_weight}')
    return SimpleNamespace(prompt_metadata=prompt_embedding_bundle.get('metadata', {}),
                           visual_cache=visual_cache, train_df=train_df, val_df=val_df, test_df=test_df,
                           tokenizer=caption_tokenizer, prompt_cache=prompt_embedding_cache,
                           transform=image_transform, vocab_size=vocab_size,
                           caption_embedding_cache=caption_embedding_cache,
                           region_targets=region_targets, region_metadata=region_metadata,
                           label_prototypes=label_prototypes)


def build_train_loader(config, data):
    dataset = CocoPromptDataset(data.train_df, data.transform, data.tokenizer, data.prompt_cache,
                                data.visual_cache, data.region_targets, config.max_regions,
                                data.caption_embedding_cache)
    device_type = getattr(config.device, 'type', config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        num_workers=config.num_workers, pin_memory=device_type == 'cuda')
    if len(loader) == 0:
        raise ValueError('Training split is too small for the batch size with drop_last=True.')
    _log(config, f'Train batches per epoch: {len(loader):,}')
    return loader


def load_visual_input(row, transform, visual_cache=None):
    if visual_cache is not None:
        return torch.from_numpy(visual_cache.read(row[visual_cache.id_key]))
    with Image.open(row['image']) as image:
        return transform(image.convert('RGB'))
