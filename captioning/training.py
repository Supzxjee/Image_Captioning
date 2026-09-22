"""Teacher-forced cross-entropy training and per-epoch checkpoint saving."""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from .config import MODEL_NAME, EMBED_DIM, MAX_PROMPT_LEN, NUM_HEADS
from .checkpoints import load_checkpoint

def object_region_loss(visual_features, boxes, labels, confidences, label_prototypes,
                       prompt_projection, temperature):
    """Classify bbox-pooled CLIP patches against CLIP text label prototypes."""
    batch, tokens, dim = visual_features.shape
    if tokens != 197:
        raise ValueError(f'Region alignment expects 197 CLIP tokens, got {tokens}.')
    patches = visual_features[:, 1:].reshape(batch, 14, 14, dim)
    centers = (torch.arange(14, device=visual_features.device, dtype=boxes.dtype) + 0.5) / 14
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')
    region_vectors, targets, weights = [], [], []
    for batch_index in range(batch):
        for region_index in torch.nonzero(labels[batch_index] >= 0, as_tuple=False).flatten().tolist():
            x1, y1, x2, y2 = boxes[batch_index, region_index]
            mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
            if mask.any():
                vector = patches[batch_index][mask].mean(dim=0)
            else:
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                distance = (xx - center_x).square() + (yy - center_y).square()
                flat_index = distance.argmin()
                vector = patches[batch_index].reshape(196, dim)[flat_index]
            region_vectors.append(vector)
            targets.append(labels[batch_index, region_index])
            weights.append(confidences[batch_index, region_index].clamp_min(0.05))
    if not region_vectors:
        return visual_features.sum() * 0
    regions = F.normalize(torch.stack(region_vectors), dim=-1)
    prototypes = F.normalize(prompt_projection(label_prototypes), dim=-1)
    logits = regions @ prototypes.t() / temperature
    target_tensor = torch.stack(targets)
    weight_tensor = torch.stack(weights)
    losses = F.cross_entropy(logits, target_tensor, reduction='none')
    return (losses * weight_tensor).sum() / weight_tensor.sum()


def train_one_epoch(model, loader, optimizer, criterion, epoch, config, label_prototypes=None):
    model.train()
    if model.encoder.feature_extractor is not None:
        model.encoder.feature_extractor.eval()
    totals = {'loss': 0.0, 'caption_loss': 0.0, 'alignment_loss': 0.0}

    progress = tqdm(loader, desc=f'Epoch {epoch}/{config.epochs}')
    for batch in progress:
        images, captions, caption_masks, prompt_tokens, prompt_mask = batch[:5]
        images = images.to(config.device, non_blocking=True)
        captions = captions.to(config.device, non_blocking=True)
        caption_masks = caption_masks.to(config.device, non_blocking=True)
        prompt_tokens = prompt_tokens.to(config.device, non_blocking=True)
        prompt_mask = prompt_mask.to(config.device, non_blocking=True)
        region_batch = batch[5:] if len(batch) > 5 else None

        batch_size = images.size(0)
        captions_flat = captions.reshape(batch_size * 5, -1)
        masks_flat = caption_masks.reshape(batch_size * 5, -1)
        tgt_input = captions_flat[:, :-1]
        tgt_expected = captions_flat[:, 1:]
        tgt_padding_mask = masks_flat[:, :-1]

        optimizer.zero_grad(set_to_none=True)
        output = model(
            images=images,
            cached_prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            tgt=tgt_input,
            tgt_key_padding_mask=tgt_padding_mask,
            return_visual=region_batch is not None,
        )
        if region_batch is not None:
            logits, visual_features = output
        else:
            logits, visual_features = output, None
        caption_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
        alignment_loss = caption_loss.new_zeros(())
        if region_batch is not None:
            boxes, labels, confidences = [value.to(config.device, non_blocking=True)
                                          for value in region_batch]
            alignment_loss = object_region_loss(
                visual_features, boxes, labels, confidences, label_prototypes,
                model.encoder.prompt_projection, config.alignment_temperature)
        loss = caption_loss + config.alignment_weight * alignment_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals['loss'] += loss.item()
        totals['caption_loss'] += caption_loss.item()
        totals['alignment_loss'] += alignment_loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}', cap=f'{caption_loss.item():.4f}',
                             align=f'{alignment_loss.item():.4f}')

    return {key: value / len(loader) for key, value in totals.items()}

def train_model(model, train_loader, data, config):
    criterion = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_idx)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    if config.checkpoint:
        load_checkpoint(model, config.checkpoint, data.tokenizer, warm_start=True)
        print(f'Warm-started from {config.checkpoint}; fresh optimizer.', flush=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    label_prototypes = (data.label_prototypes.to(config.device)
                        if data.label_prototypes is not None else None)
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        losses = train_one_epoch(model, train_loader, optimizer, criterion, epoch, config,
                                 label_prototypes)
        elapsed = time.monotonic() - started
        history.append({'epoch': epoch, 'train_loss': losses['loss'],
                        'caption_loss': losses['caption_loss'],
                        'alignment_loss': losses['alignment_loss'], 'train_seconds': elapsed})
        print(f'Epoch training time: {elapsed:.1f}s | {elapsed / len(train_loader):.3f}s/batch', flush=True)

        checkpoint_path = config.checkpoint_dir / f'model_h1_2_crossattn_epoch_{epoch}.pth'
        torch.save({
            'experiment_name': 'H1.2_gated_prompt_to_visual_cross_attention',
            'gate': 'sigmoid(linear(concat(prompt, attended_visual)))',
            'seed': config.seed,
            'prompt_cache_path': config.prompt_cache_path,
            'prompt_metadata': getattr(data, 'prompt_metadata', {}),
            'visual_preprocessing': config.visual_preprocessing,
            'visual_precision': config.visual_precision,
            'visual_cache_id_key': config.visual_cache_id_key,
            'visual_cache_files': data.visual_cache.paths if data.visual_cache else [],
            'learning_rate': config.lr,
            'alignment_weight': config.alignment_weight,
            'alignment_temperature': config.alignment_temperature,
            'region_targets_path': config.region_targets_path,
            'region_metadata': data.region_metadata,
            'caption_word2idx': data.tokenizer.word2idx,
            'cross_attention': 'query=prompt, key=visual, value=visual',
            'epoch': epoch,
            'loss': losses['loss'],
            'caption_loss': losses['caption_loss'],
            'alignment_loss': losses['alignment_loss'],
            'model_name': MODEL_NAME,
            'embed_dim': EMBED_DIM,
            'max_prompt_len': MAX_PROMPT_LEN,
            'num_attention_heads': NUM_HEADS,
            'vocab_size': data.vocab_size,
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f'✅ Epoch {epoch}/{config.epochs} | total: {losses["loss"]:.4f} | '
              f'caption: {losses["caption_loss"]:.4f} | alignment: {losses["alignment_loss"]:.4f}')
        print(f'💾 Saved: {checkpoint_path}')

    history_path = config.work_dir / config.experiment_name / 'train_history_h1_2.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f'✅ Training complete. History: {history_path}')
