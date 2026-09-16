"""Kaggle CLI for H1.2-G; notebook and script share the same implementation."""
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--checkpoint', default='')
parser.add_argument('--split', choices=['val', 'test'], default='val')
parser.add_argument('--limit', type=int, default=0, help='0 = full split; positive = first N images')
args = parser.parse_args()
os.environ.update(CAPTION_TRAIN=str(int(args.mode == 'train')),
                  CAPTION_EVALUATE=str(int(args.mode == 'evaluate')),
                  CAPTION_EPOCHS=str(args.epochs), CAPTION_LR=str(args.lr),
                  CAPTION_CHECKPOINT=args.checkpoint, CAPTION_EVAL_SPLIT=args.split,
                  CAPTION_EVAL_LIMIT=str(args.limit))


import os
import time
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

MODEL_NAME = 'clip'
EMBED_DIM = 512
MAX_PROMPT_LEN = 20
MAX_CAPTION_LEN = 30
NUM_HEADS = 8
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = float(os.environ.get('CAPTION_LR', '1e-4'))
EPOCHS = int(os.environ.get('CAPTION_EPOCHS', '10'))
RUN_TRAIN = os.environ.get('CAPTION_TRAIN', '1') == '1'
RUN_EVALUATION = os.environ.get('CAPTION_EVALUATE', '0') == '1'
EVAL_SPLIT = os.environ.get('CAPTION_EVAL_SPLIT', 'val')
EVAL_LIMIT = int(os.environ.get('CAPTION_EVAL_LIMIT', '0'))
CHECKPOINT_INPUT = os.environ.get('CAPTION_CHECKPOINT', '')
assert EVAL_SPLIT in {'val', 'test'}
assert EPOCHS > 0 and EVAL_LIMIT >= 0

BASE_PATH = '/kaggle/input/datasets/vuthetam/mscoco-2014/images'
DATASET_JSON_PATH = '/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json'
PROMPT_EMBEDDING_CACHE_PATH = '/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt'
WORK_DIR = Path('/kaggle/working')
EXPERIMENT_NAME = 'h1_2_gated_prompt_to_visual_crossattn'
CHECKPOINT_DIR = WORK_DIR / EXPERIMENT_NAME / 'checkpoints'
EVAL_DIR = WORK_DIR / EXPERIMENT_NAME / 'evaluation'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

assert os.path.isfile(DATASET_JSON_PATH), f'Missing dataset JSON: {DATASET_JSON_PATH}'
assert os.path.isfile(PROMPT_EMBEDDING_CACHE_PATH), f'Missing prompt embedding cache: {PROMPT_EMBEDDING_CACHE_PATH}'

with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

train_data, val_data, test_data = [], [], []
for coco_image_id, img in enumerate(coco_data['images']):
    full_image_path = os.path.join(BASE_PATH, img['filepath'], img['filename'])
    captions = [sent['raw'] for sent in img['sentences']][:5]
    item = {
        'image': full_image_path,
        'captions': captions,
        'eval_id': coco_image_id,
        'filename': img['filename'],
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
assert len(test_df) > 0, 'Test split is empty.'

class CaptionTokenizer:
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self, texts, max_vocab_size=10000):
        self.word2idx = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        for idx, (word, _) in enumerate(counter.most_common(max_vocab_size - 4), start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @property
    def pad_idx(self):
        return self.word2idx[self.PAD]

    @property
    def bos_idx(self):
        return self.word2idx[self.BOS]

    @property
    def eos_idx(self):
        return self.word2idx[self.EOS]

    def encode(self, text, max_len=MAX_CAPTION_LEN):
        words = text.lower().split()
        ids = [self.bos_idx] + [self.word2idx.get(w, self.word2idx[self.UNK]) for w in words] + [self.eos_idx]
        ids = ids[:max_len]
        padding_mask = [False] * len(ids)
        if len(ids) < max_len:
            pad_count = max_len - len(ids)
            ids.extend([self.pad_idx] * pad_count)
            padding_mask.extend([True] * pad_count)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(padding_mask, dtype=torch.bool)

all_train_captions = [caption for item in train_data for caption in item['captions']]
caption_tokenizer = CaptionTokenizer(all_train_captions)
VOCAB_SIZE = len(caption_tokenizer.word2idx)
print(f'Caption vocabulary: {VOCAB_SIZE:,}')

prompt_embedding_bundle = torch.load(PROMPT_EMBEDDING_CACHE_PATH, map_location='cpu')
prompt_embedding_cache = prompt_embedding_bundle['data']
print(f'Prompt embedding entries: {len(prompt_embedding_cache):,}')

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
    def __init__(self, dataframe, transform, caption_tokenizer, prompt_cache):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.caption_tokenizer = caption_tokenizer
        self.prompt_cache = prompt_cache

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image']
        if image_path not in self.prompt_cache:
            raise KeyError(f'Missing prompt embedding for: {image_path}')

        image = self.transform(Image.open(image_path).convert('RGB'))
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

train_dataset = CocoPromptDataset(train_df, image_transform, caption_tokenizer, prompt_embedding_cache)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
print(f'Train batches per epoch: {len(train_loader):,}')

class UniversalVisionEncoder(nn.Module):
    def __init__(self, model_name='clip', embed_dim=EMBED_DIM, num_heads=NUM_HEADS, attn_dropout=0.1):
        super().__init__()
        if model_name.lower() != 'clip':
            raise NotImplementedError('Only CLIP is supported in this notebook.')

        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        self.feature_extractor = clip_model.vision_model
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.vis_projection = nn.Linear(768, embed_dim)
        self.prompt_projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
        )

        # H1.2: Q = prompt tokens, K = V = visual tokens.
        self.prompt_to_visual_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        # Gate sees both the original prompt and retrieved visual information.
        # Initial sigmoid(-2) ~= 0.119: start with a small visual update.
        self.prompt_visual_gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.prompt_visual_gate.weight)
        nn.init.constant_(self.prompt_visual_gate.bias, -2.0)
        self.prompt_attn_norm = nn.LayerNorm(embed_dim)
        self.prompt_attn_dropout = nn.Dropout(attn_dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def forward(self, images, cached_prompt_tokens, prompt_mask):
        with torch.no_grad():
            visual_features = self.feature_extractor(pixel_values=images).last_hidden_state

        vis_features = self.dropout(self.relu(self.vis_projection(visual_features)))
        prompt_features = self.prompt_projection(cached_prompt_tokens)

        attended_prompt, _ = self.prompt_to_visual_attn(
            query=prompt_features,
            key=vis_features,
            value=vis_features,
            need_weights=False,
        )

        gate = torch.sigmoid(self.prompt_visual_gate(
            torch.cat([prompt_features, attended_prompt], dim=-1)
        ))
        grounded_prompt_features = self.prompt_attn_norm(
            prompt_features + self.prompt_attn_dropout(gate * attended_prompt)
        )

        prompt_pad_mask = (prompt_mask == 0)
        grounded_prompt_features = grounded_prompt_features.masked_fill(
            prompt_pad_mask.unsqueeze(-1),
            0.0,
        )

        return vis_features, grounded_prompt_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMBED_DIM, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=NUM_HEADS, num_layers=3, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    @staticmethod
    def causal_mask(size, device):
        return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_embeddings = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        return self.fc_out(self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=self.causal_mask(tgt.size(1), tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ))

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def build_memory(self, images, cached_prompt_tokens, prompt_mask):
        vis_features, grounded_prompt_features = self.encoder(
            images, cached_prompt_tokens, prompt_mask
        )
        memory = torch.cat([grounded_prompt_features, vis_features], dim=1)
        prompt_pad_mask = (prompt_mask == 0)
        visual_pad_mask = torch.zeros(
            images.size(0), vis_features.size(1), dtype=torch.bool, device=images.device
        )
        memory_pad_mask = torch.cat([prompt_pad_mask, visual_pad_mask], dim=1)
        return memory, memory_pad_mask

    def forward(self, images, cached_prompt_tokens, prompt_mask, tgt, tgt_key_padding_mask=None):
        memory, memory_pad_mask = self.build_memory(images, cached_prompt_tokens, prompt_mask)
        batch_size, memory_len, embed_dim = memory.shape
        captions_per_image = tgt.size(0) // batch_size
        if tgt.size(0) != batch_size * captions_per_image:
            raise ValueError('Target batch size must be a multiple of image batch size.')

        memory = memory.unsqueeze(1).expand(batch_size, captions_per_image, memory_len, embed_dim)
        memory = memory.reshape(batch_size * captions_per_image, memory_len, embed_dim)
        memory_pad_mask = memory_pad_mask.unsqueeze(1).expand(batch_size, captions_per_image, memory_len)
        memory_pad_mask = memory_pad_mask.reshape(batch_size * captions_per_image, memory_len)

        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    model.encoder.feature_extractor.eval()
    total_loss = 0.0

    progress = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS}')
    for images, captions, caption_masks, prompt_tokens, prompt_mask in progress:
        images = images.to(DEVICE, non_blocking=True)
        captions = captions.to(DEVICE, non_blocking=True)
        caption_masks = caption_masks.to(DEVICE, non_blocking=True)
        prompt_tokens = prompt_tokens.to(DEVICE, non_blocking=True)
        prompt_mask = prompt_mask.to(DEVICE, non_blocking=True)

        batch_size = images.size(0)
        captions_flat = captions.reshape(batch_size * 5, -1)
        masks_flat = caption_masks.reshape(batch_size * 5, -1)
        tgt_input = captions_flat[:, :-1]
        tgt_expected = captions_flat[:, 1:]
        tgt_padding_mask = masks_flat[:, :-1]

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            images=images,
            cached_prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            tgt=tgt_input,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(loader)

encoder = UniversalVisionEncoder().to(DEVICE)
decoder = CaptionDecoder(vocab_size=VOCAB_SIZE).to(DEVICE)
model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=caption_tokenizer.pad_idx)
optimizer = optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=LEARNING_RATE,
)

# Warm-start model weights only; optimizer is deliberately reset.
if CHECKPOINT_INPUT and RUN_TRAIN:
    initial = torch.load(CHECKPOINT_INPUT, map_location='cpu', weights_only=False)
    if initial.get('caption_word2idx', caption_tokenizer.word2idx) != caption_tokenizer.word2idx:
        raise RuntimeError('Checkpoint caption vocabulary differs from current dataset.')
    result = model.encoder.load_state_dict(initial['encoder_state_dict'], strict=False)
    allowed = {'prompt_visual_gate.weight', 'prompt_visual_gate.bias'}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise RuntimeError(f'Incompatible encoder checkpoint: {result}')
    model.decoder.load_state_dict(initial['decoder_state_dict'])
    print(f'Warm-started from {CHECKPOINT_INPUT}; fresh optimizer; gate initialized if absent.', flush=True)

if RUN_TRAIN:
    history = []
    for epoch in range(1, EPOCHS + 1):
        average_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        history.append({'epoch': epoch, 'train_loss': average_loss})

        checkpoint_path = CHECKPOINT_DIR / f'model_h1_2_crossattn_epoch_{epoch}.pth'
        torch.save({
            'experiment_name': 'H1.2_gated_prompt_to_visual_cross_attention',
            'gate': 'sigmoid(linear(concat(prompt, attended_visual)))',
            'seed': SEED,
            'learning_rate': LEARNING_RATE,
            'caption_word2idx': caption_tokenizer.word2idx,
            'cross_attention': 'query=prompt, key=visual, value=visual',
            'epoch': epoch,
            'loss': average_loss,
            'model_name': MODEL_NAME,
            'embed_dim': EMBED_DIM,
            'max_prompt_len': MAX_PROMPT_LEN,
            'num_attention_heads': NUM_HEADS,
            'vocab_size': VOCAB_SIZE,
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f'✅ Epoch {epoch}/{EPOCHS} | train loss: {average_loss:.4f}')
        print(f'💾 Saved: {checkpoint_path}')

    history_path = WORK_DIR / EXPERIMENT_NAME / 'train_history_h1_2.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f'✅ Training complete. History: {history_path}')

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def generate_caption_beam_search(model, image, cached_prompt_tokens, prompt_mask, tokenizer, max_len=MAX_CAPTION_LEN, beam_size=5):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)
        cached_prompt_tokens = cached_prompt_tokens.unsqueeze(0).to(DEVICE, dtype=torch.float32)
        prompt_mask = prompt_mask.unsqueeze(0).to(DEVICE, dtype=torch.long)
        memory, memory_pad_mask = model.build_memory(image, cached_prompt_tokens, prompt_mask)

        beams = [([tokenizer.bos_idx], 0.0)]
        for _ in range(max_len - 1):
            candidates = []
            for token_ids, score in beams:
                if token_ids[-1] == tokenizer.eos_idx:
                    candidates.append((token_ids, score))
                    continue

                target = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                logits = model.decoder(
                    tgt=target,
                    memory=memory,
                    memory_key_padding_mask=memory_pad_mask,
                )
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                top_log_probs, top_ids = log_probs.topk(beam_size)
                for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                    candidates.append((token_ids + [token_id], score + log_prob))

            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
            if all(token_ids[-1] == tokenizer.eos_idx for token_ids, _ in beams):
                break

    best_tokens = beams[0][0]
    words = [tokenizer.idx2word.get(token_id, tokenizer.UNK) for token_id in best_tokens]
    words = [word for word in words if word not in {tokenizer.PAD, tokenizer.BOS, tokenizer.EOS}]
    return ' '.join(words)

def export_ground_truth(dataframe, output_path):
    images, annotations = [], []
    annotation_id = 0
    for _, row in dataframe.iterrows():
        image_id = int(row['eval_id'])
        images.append({'id': image_id})
        for caption in row['captions']:
            annotations.append({'id': annotation_id, 'image_id': image_id, 'caption': caption})
            annotation_id += 1

    payload = {
        'info': {},
        'licenses': [],
        'type': 'captions',
        'images': images,
        'annotations': annotations,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

def compute_coco_metrics(predictions_path, ground_truth_path):
    coco_gt = COCO(str(ground_truth_path))
    coco_res = coco_gt.loadRes(str(predictions_path))
    image_ids = coco_res.getImgIds()

    gts = {image_id: coco_gt.imgToAnns[image_id] for image_id in image_ids}
    res = {image_id: coco_res.imgToAnns[image_id] for image_id in image_ids}

    ptb_tokenizer = PTBTokenizer()
    gts = ptb_tokenizer.tokenize(gts)
    res = ptb_tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        (Meteor(), 'METEOR'),
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr'),
    ]

    metrics = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(names, list):
            metrics.update({name: float(value) for name, value in zip(names, score)})
        else:
            metrics[names] = float(score)
    return metrics

if RUN_EVALUATION:
    CHECKPOINT_TO_EVALUATE = Path(CHECKPOINT_INPUT) if CHECKPOINT_INPUT else CHECKPOINT_DIR / f'model_h1_2_crossattn_epoch_{EPOCHS}.pth'
    assert CHECKPOINT_TO_EVALUATE.is_file(), f'Missing checkpoint: {CHECKPOINT_TO_EVALUATE}'

    checkpoint = torch.load(CHECKPOINT_TO_EVALUATE, map_location='cpu', weights_only=False)
    if checkpoint.get('caption_word2idx', caption_tokenizer.word2idx) != caption_tokenizer.word2idx:
        raise RuntimeError('Checkpoint caption vocabulary differs from current dataset.')
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    print(f"✅ Loaded H1.2 checkpoint: epoch {checkpoint['epoch']} | loss {checkpoint['loss']:.4f}")

    eval_df = val_df if EVAL_SPLIT == 'val' else test_df
    if EVAL_LIMIT:
        eval_df = eval_df.head(EVAL_LIMIT)
    print(f'Starting {EVAL_SPLIT} inference: {len(eval_df)} images, beam size 5', flush=True)
    started = time.monotonic()
    predictions = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f'Generating {EVAL_SPLIT} captions'):
        image_path = row['image']
        prompt_entry = prompt_embedding_cache[image_path]
        image_tensor = image_transform(Image.open(image_path).convert('RGB'))
        caption = generate_caption_beam_search(
            model=model,
            image=image_tensor,
            cached_prompt_tokens=prompt_entry['tokens'],
            prompt_mask=prompt_entry['mask'],
            tokenizer=caption_tokenizer,
            beam_size=5,
        )
        predictions.append({'image_id': int(row['eval_id']), 'caption': caption})
        if len(predictions) % 50 == 0:
            elapsed = time.monotonic() - started
            rate = elapsed / len(predictions)
            print(f'Generated {len(predictions)}/{len(eval_df)} | {rate:.2f}s/image | ETA {(len(eval_df)-len(predictions))*rate/60:.1f}min', flush=True)
            partial_path = EVAL_DIR / f'{EVAL_SPLIT}_captions_partial.json'
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False)


    predictions_path = EVAL_DIR / f'{EVAL_SPLIT}_{len(eval_df)}_captions_h1_2_gated.json'
    ground_truth_path = EVAL_DIR / f'{EVAL_SPLIT}_{len(eval_df)}_gt_h1_2_gated.json'
    metrics_path = EVAL_DIR / f'{EVAL_SPLIT}_{len(eval_df)}_metrics_h1_2_gated.json'

    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    export_ground_truth(eval_df, ground_truth_path)

    metrics = compute_coco_metrics(predictions_path, ground_truth_path)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n===== H1.2-G — EVALUATION RESULTS =====')
    print(f'Evaluated images: {len(predictions):,}')
    for metric_name, value in metrics.items():
        print(f'{metric_name:<10}: {value:.4f}')
    print(f'\nPredictions: {predictions_path}')
    print(f'Ground truth: {ground_truth_path}')
    print(f'Metrics: {metrics_path}')