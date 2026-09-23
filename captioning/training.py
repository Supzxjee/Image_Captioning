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
    patches = visual_features[:, 1:].reshape(batch, 196, dim)
    centers = (torch.arange(14, device=visual_features.device, dtype=boxes.dtype) + 0.5) / 14
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')
    patch_x = xx.reshape(1, 1, 196)
    patch_y = yy.reshape(1, 1, 196)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    inside = ((patch_x >= x1.unsqueeze(-1)) & (patch_x <= x2.unsqueeze(-1)) &
              (patch_y >= y1.unsqueeze(-1)) & (patch_y <= y2.unsqueeze(-1)))
    valid = labels >= 0
    inside = inside & valid.unsqueeze(-1)

    # One batched matrix multiplication replaces hundreds of small GPU launches
    # and Python/GPU synchronizations per batch.
    counts = inside.sum(dim=-1)
    pooled = torch.einsum('brp,bpd->brd', inside.to(patches.dtype), patches)
    pooled = pooled / counts.clamp_min(1).unsqueeze(-1).to(pooled.dtype)

    # Very small boxes can contain no patch center. Select the patch nearest to
    # the bbox centre for all such regions in one vectorized gather.
    center_x = ((x1 + x2) / 2).unsqueeze(-1)
    center_y = ((y1 + y2) / 2).unsqueeze(-1)
    distances = (patch_x - center_x).square() + (patch_y - center_y).square()
    nearest_index = distances.argmin(dim=-1)
    nearest = patches.gather(
        1, nearest_index.unsqueeze(-1).expand(-1, -1, dim)
    )
    pooled = torch.where((counts > 0).unsqueeze(-1), pooled, nearest)

    if not valid.any():
        return visual_features.sum() * 0
    regions = F.normalize(pooled[valid], dim=-1)
    prototypes = F.normalize(prompt_projection(label_prototypes), dim=-1)
    logits = regions @ prototypes.t() / temperature
    target_tensor = labels[valid]
    weight_tensor = confidences[valid].clamp_min(0.05)
    losses = F.cross_entropy(logits, target_tensor, reduction='none')
    return (losses * weight_tensor).sum() / weight_tensor.sum()


def train_one_epoch(model, loader, optimizer, criterion, epoch, config, label_prototypes=None):
    model.train()
    if model.encoder.feature_extractor is not None:
        model.encoder.feature_extractor.eval()
    totals = {'loss': 0.0, 'caption_loss': 0.0, 'alignment_loss': 0.0}

    progress = tqdm(loader, desc=f'Epoch {epoch}/{config.epochs}')
    processed_batches = 0
    epoch_started = time.monotonic()
    print(f'Epoch {epoch}: waiting for first training batch...', flush=True)
    for batch_number, batch in enumerate(progress, 1):
        if config.max_train_batches and batch_number > config.max_train_batches:
            break
        if batch_number == 1:
            print(f'Epoch {epoch}: first batch loaded; starting GPU forward/backward.', flush=True)
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
        processed_batches += 1
        progress.set_postfix(loss=f'{loss.item():.4f}', cap=f'{caption_loss.item():.4f}',
                             align=f'{alignment_loss.item():.4f}')
        if batch_number == 1 or batch_number % 100 == 0:
            elapsed = time.monotonic() - epoch_started
            total_batches = min(len(loader), config.max_train_batches or len(loader))
            eta_minutes = max(total_batches - batch_number, 0) * elapsed / batch_number / 60
            print(f'Epoch {epoch}: batch {batch_number}/{total_batches} | '
                  f'{elapsed / batch_number:.2f}s/batch | ETA {eta_minutes:.1f}min | '
                  f'total={loss.item():.4f} caption={caption_loss.item():.4f} '
                  f'alignment={alignment_loss.item():.4f}', flush=True)

    if processed_batches == 0:
        raise ValueError('No training batches were processed.')
    return {key: value / processed_batches for key, value in totals.items()}

def train_model(model, train_loader, data, config):
    criterion = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_idx)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    if config.checkpoint:
        load_checkpoint(model, config.checkpoint, data.tokenizer, warm_start=True)
        print(f'Warm-started from {config.checkpoint}; fresh optimizer.', flush=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.max_train_batches:
        print(f'SMOKE MODE: stopping each epoch after {config.max_train_batches} batches.', flush=True)
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
            'max_train_batches': config.max_train_batches,
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
