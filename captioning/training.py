"""Teacher-forced cross-entropy training and per-epoch checkpoint saving."""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .config import MODEL_NAME, EMBED_DIM, MAX_PROMPT_LEN, NUM_HEADS
from .checkpoints import load_checkpoint

def train_one_epoch(model, loader, optimizer, criterion, epoch, config):
    model.train()
    if model.encoder.feature_extractor is not None:
        model.encoder.feature_extractor.eval()
    total_loss = 0.0

    progress = tqdm(loader, desc=f'Epoch {epoch}/{config.epochs}')
    for images, captions, caption_masks, prompt_tokens, prompt_mask in progress:
        images = images.to(config.device, non_blocking=True)
        captions = captions.to(config.device, non_blocking=True)
        caption_masks = caption_masks.to(config.device, non_blocking=True)
        prompt_tokens = prompt_tokens.to(config.device, non_blocking=True)
        prompt_mask = prompt_mask.to(config.device, non_blocking=True)

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

def train_model(model, train_loader, data, config):
    criterion = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_idx)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    if config.checkpoint:
        load_checkpoint(model, config.checkpoint, data.tokenizer, warm_start=True)
        print(f'Warm-started from {config.checkpoint}; fresh optimizer.', flush=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        average_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, config)
        elapsed = time.monotonic() - started
        history.append({'epoch': epoch, 'train_loss': average_loss, 'train_seconds': elapsed})
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
            'caption_word2idx': data.tokenizer.word2idx,
            'cross_attention': 'query=prompt, key=visual, value=visual',
            'epoch': epoch,
            'loss': average_loss,
            'model_name': MODEL_NAME,
            'embed_dim': EMBED_DIM,
            'max_prompt_len': MAX_PROMPT_LEN,
            'num_attention_heads': NUM_HEADS,
            'vocab_size': data.vocab_size,
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f'✅ Epoch {epoch}/{config.epochs} | train loss: {average_loss:.4f}')
        print(f'💾 Saved: {checkpoint_path}')

    history_path = config.work_dir / config.experiment_name / 'train_history_h1_2.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f'✅ Training complete. History: {history_path}')
