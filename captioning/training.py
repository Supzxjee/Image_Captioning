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
from .grounded_relations import counterfactual_relation_loss, appearance_relation_loss


def _gather_with_grad(tensor):
    """Gather equal local batches while preserving gradients for global ITC negatives."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(tensor.contiguous()), dim=0)


def image_text_contrastive_loss(projected_queries, caption_embeddings, temperature):
    """BLIP-2-style max-query image/text contrastive loss with batch negatives."""
    queries = F.normalize(projected_queries, dim=-1)
    captions = F.normalize(caption_embeddings, dim=-1)
    captions = F.normalize(captions.mean(dim=1), dim=-1)
    queries = _gather_with_grad(queries)
    captions = _gather_with_grad(captions)
    scores = torch.einsum('bqd,nd->bqn', queries, captions).max(dim=1).values
    scores = scores / temperature
    targets = torch.arange(scores.size(0), device=scores.device)
    return (F.cross_entropy(scores, targets) +
            F.cross_entropy(scores.transpose(0, 1), targets)) / 2

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


def train_one_epoch(model, base_model, loader, optimizer, criterion, epoch, config,
                    accelerator, label_prototypes=None):
    model.train()
    if base_model.encoder.feature_extractor is not None:
        base_model.encoder.feature_extractor.eval()
    if config.grounded_relation_version == 'verified_v3':
        base_model.encoder.dropout.eval()
        base_model.encoder.relation_alignment.appearance_projection.eval()
    totals = {'loss': 0.0, 'caption_loss': 0.0, 'alignment_loss': 0.0,
              'itc_loss': 0.0, 'relation_loss': 0.0}
    if config.grounded_relation_version == 'appearance_v2':
        totals.update(relation_predicate_loss=0.0, relation_wrong_image_loss=0.0,
                      relation_actions_per_batch=0.0, relation_wrong_images_per_batch=0.0)

    total_batches = min(len(loader), config.max_train_batches or len(loader))
    progress = tqdm(loader, total=total_batches, desc=f'Epoch {epoch}/{config.epochs}',
                    disable=not accelerator.is_local_main_process)
    processed_batches = 0
    epoch_started = time.monotonic()
    accelerator.print(f'Epoch {epoch}: waiting for first training batch...')
    for batch_number, batch in enumerate(progress, 1):
        if config.max_train_batches and batch_number > config.max_train_batches:
            break
        if batch_number == 1:
            accelerator.print(f'Epoch {epoch}: first batch loaded; starting GPU forward/backward.')
        if ((config.object_semantic_alignment or config.cascade_semantic_alignment or
             config.grounded_relation_alignment) and
                (batch_number == 1 or batch_number % 100 == 1)):
            print(f'[rank {accelerator.process_index}] epoch {epoch} batch {batch_number}: '
                  'loaded', flush=True)
        images, captions, caption_masks, prompt_tokens, prompt_mask = batch[:5]
        images = images.to(config.device, non_blocking=True)
        captions = captions.to(config.device, non_blocking=True)
        caption_masks = caption_masks.to(config.device, non_blocking=True)
        prompt_tokens = prompt_tokens.to(config.device, non_blocking=True)
        prompt_mask = prompt_mask.to(config.device, non_blocking=True)
        offset = 5
        object_prompt_tokens = None
        object_prompt_mask = None
        if config.object_semantic_alignment or config.cascade_semantic_alignment:
            object_prompt_tokens = batch[offset].to(config.device, non_blocking=True)
            object_prompt_mask = batch[offset + 1].to(config.device, non_blocking=True)
            offset += 2
        relation_batch = None
        relation_action_mask = None
        if config.grounded_relation_alignment:
            relation_batch = tuple(value.to(config.device, non_blocking=True)
                                   for value in batch[offset:offset + 6])
            offset += 6
            if config.grounded_relation_version in {'appearance_v2', 'verified_v3'}:
                relation_action_mask = batch[offset].to(config.device, non_blocking=True)
                offset += 1
        caption_embeddings = None
        if config.itc_weight > 0:
            caption_embeddings = batch[offset].to(config.device, non_blocking=True)
            offset += 1
        region_batch = batch[offset:] if config.alignment_weight > 0 else None

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
            return_itc=caption_embeddings is not None,
            object_prompt_tokens=object_prompt_tokens,
            object_prompt_mask=object_prompt_mask,
            relation_tokens=(relation_batch[0] if relation_batch else None),
            negative_relation_tokens=(relation_batch[1] if relation_batch else None),
            subject_boxes=(relation_batch[2] if relation_batch else None),
            object_boxes=(relation_batch[3] if relation_batch else None),
            relation_confidences=(relation_batch[4] if relation_batch else None),
            relation_mask=(relation_batch[5] if relation_batch else None),
            return_relation_diagnostics=relation_batch is not None,
        )
        if ((config.object_semantic_alignment or config.cascade_semantic_alignment or
             config.grounded_relation_alignment) and
                (batch_number == 1 or batch_number % 100 == 1)):
            print(f'[rank {accelerator.process_index}] epoch {epoch} batch {batch_number}: '
                  'forward complete', flush=True)
        if relation_batch is not None:
            logits, relation_diagnostics = output
            visual_features = None
        elif region_batch is not None or caption_embeddings is not None:
            logits, visual_features = output
            relation_diagnostics = None
        else:
            logits, visual_features = output, None
            relation_diagnostics = None
        caption_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
        alignment_loss = caption_loss.new_zeros(())
        if region_batch is not None:
            boxes, labels, confidences = [value.to(config.device, non_blocking=True)
                                          for value in region_batch]
            alignment_loss = object_region_loss(
                visual_features, boxes, labels, confidences, label_prototypes,
                base_model.encoder.prompt_projection, config.alignment_temperature)
        itc_loss = caption_loss.new_zeros(())
        if caption_embeddings is not None:
            itc_loss = image_text_contrastive_loss(
                visual_features, caption_embeddings, config.itc_temperature)
        relation_loss = caption_loss.new_zeros(())
        if relation_diagnostics is not None:
            if config.grounded_relation_version == 'verified_v3':
                if batch_number == 1 or batch_number % 100 == 0:
                    accelerator.print(f'Verified V3: accepted action relations='
                                      f'{relation_diagnostics["accepted_mask"].sum().item()}')
            elif config.grounded_relation_version == 'appearance_v2':
                relation_loss, relation_terms = appearance_relation_loss(
                    relation_diagnostics, relation_action_mask, config.relation_alignment_margin,
                    config.relation_wrong_image_weight)
                if batch_number == 1 or batch_number % 100 == 0:
                    accelerator.print(
                        f'Appearance V2: actions={relation_terms["actions"]} | '
                        f'eligible wrong images={relation_terms["eligible_wrong_images"]} | '
                        f'predicate={relation_terms["predicate_loss"].detach().item():.4f} | '
                        f'wrong image={relation_terms["wrong_image_loss"].detach().item():.4f}')
            else:
                relation_loss = counterfactual_relation_loss(
                    relation_diagnostics, config.relation_alignment_margin)
        loss = (caption_loss + config.alignment_weight * alignment_loss +
                config.itc_weight * itc_loss +
                config.relation_alignment_weight * relation_loss)
        accelerator.backward(loss)
        if ((config.object_semantic_alignment or config.cascade_semantic_alignment or
             config.grounded_relation_alignment) and
                (batch_number == 1 or batch_number % 100 == 1)):
            print(f'[rank {accelerator.process_index}] epoch {epoch} batch {batch_number}: '
                  'backward complete', flush=True)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ((config.object_semantic_alignment or config.cascade_semantic_alignment or
             config.grounded_relation_alignment) and
                (batch_number == 1 or batch_number % 100 == 1)):
            print(f'[rank {accelerator.process_index}] epoch {epoch} batch {batch_number}: '
                  'optimizer complete', flush=True)

        # DDP already synchronizes model gradients during backward. A second
        # per-batch collective here was only used to average display values and
        # intermittently deadlocked on Kaggle T4x2 after both ranks had finished
        # optimizer.step(). Track rank-local losses instead; they do not affect
        # training, checkpoint weights or validation metrics.
        reduced_loss = loss.detach().item()
        reduced_caption = caption_loss.detach().item()
        reduced_alignment = alignment_loss.detach().item()
        reduced_itc = itc_loss.detach().item()
        reduced_relation = relation_loss.detach().item()
        totals['loss'] += reduced_loss
        totals['caption_loss'] += reduced_caption
        totals['alignment_loss'] += reduced_alignment
        totals['itc_loss'] += reduced_itc
        totals['relation_loss'] += reduced_relation
        if config.grounded_relation_version == 'appearance_v2':
            totals['relation_predicate_loss'] += relation_terms['predicate_loss'].detach().item()
            totals['relation_wrong_image_loss'] += relation_terms['wrong_image_loss'].detach().item()
            totals['relation_actions_per_batch'] += relation_terms['actions']
            totals['relation_wrong_images_per_batch'] += relation_terms['eligible_wrong_images']
        processed_batches += 1
        progress.set_postfix(loss=f'{reduced_loss:.4f}', cap=f'{reduced_caption:.4f}',
                             rel=f'{reduced_relation:.4f}', itc=f'{reduced_itc:.4f}')
        if batch_number == 1 or batch_number % 100 == 0:
            elapsed = time.monotonic() - epoch_started
            eta_minutes = max(total_batches - batch_number, 0) * elapsed / batch_number / 60
            accelerator.print(f'Epoch {epoch}: batch {batch_number}/{total_batches} | '
                              f'{elapsed / batch_number:.2f}s/batch | ETA {eta_minutes:.1f}min | '
                              f'total={reduced_loss:.4f} caption={reduced_caption:.4f} '
                              f'alignment={reduced_alignment:.4f} relation={reduced_relation:.4f} '
                              f'itc={reduced_itc:.4f}')

    if processed_batches == 0:
        raise ValueError('No training batches were processed.')
    return {key: value / processed_batches for key, value in totals.items()}

def train_model(model, train_loader, data, config, accelerator=None):
    if accelerator is None:
        from accelerate import Accelerator, DataLoaderConfiguration
        accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=True))
    criterion = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_idx)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    if config.checkpoint:
        load_checkpoint(model, config.checkpoint, data.tokenizer, warm_start=True)
        accelerator.print(f'Warm-started from {config.checkpoint}; fresh optimizer.')
    if accelerator.is_main_process:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.max_train_batches:
        accelerator.print(f'SMOKE MODE: stopping each epoch after '
                          f'{config.max_train_batches} batches.')
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    base_model = accelerator.unwrap_model(model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + accelerator.process_index)
    history = []
    label_prototypes = (data.label_prototypes.to(config.device)
                        if data.label_prototypes is not None else None)
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        losses = train_one_epoch(model, base_model, train_loader, optimizer, criterion, epoch,
                                 config, accelerator, label_prototypes)
        elapsed = time.monotonic() - started
        history.append({'epoch': epoch, 'train_loss': losses['loss'],
                        'caption_loss': losses['caption_loss'],
                        'alignment_loss': losses['alignment_loss'],
                        'itc_loss': losses['itc_loss'],
                        'relation_loss': losses['relation_loss'],
                        'train_seconds': elapsed})
        if config.grounded_relation_version == 'appearance_v2':
            history[-1].update({key: value for key, value in losses.items()
                               if key.startswith('relation_')})
        completed_batches = min(len(train_loader), config.max_train_batches or len(train_loader))
        accelerator.print(f'Epoch training time: {elapsed:.1f}s | '
                          f'{elapsed / completed_batches:.3f}s/batch')

        checkpoint_path = config.checkpoint_dir / f'model_h1_2_crossattn_epoch_{epoch}.pth'
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save({
            'experiment_name': 'H1.2_gated_prompt_to_visual_cross_attention',
            'gate': 'sigmoid(linear(concat(prompt, attended_visual)))',
            'seed': config.seed,
            'prompt_cache_path': config.prompt_cache_path,
            'prompt_metadata': getattr(data, 'prompt_metadata', {}),
            'visual_preprocessing': config.visual_preprocessing,
            'visual_precision': config.visual_precision,
            'clip_model_path': config.clip_model_path,
            'visual_cache_id_key': config.visual_cache_id_key,
            'visual_cache_files': data.visual_cache.paths if data.visual_cache else [],
            'learning_rate': config.lr,
            'alignment_weight': config.alignment_weight,
            'alignment_temperature': config.alignment_temperature,
            'region_targets_path': config.region_targets_path,
            'region_metadata': data.region_metadata,
            'max_train_batches': config.max_train_batches,
            'visual_adapter': config.visual_adapter,
            'num_visual_queries': config.num_visual_queries,
            'qformer_layers': config.qformer_layers,
            'prompt_conditioned_qformer': config.prompt_conditioned_qformer,
            'object_semantic_alignment': config.object_semantic_alignment,
            'cascade_semantic_alignment': config.cascade_semantic_alignment,
            'cascade_selector_mode': config.cascade_selector_mode,
            'object_prompt_cache_path': config.object_prompt_cache_path,
            'object_prompt_metadata': data.object_prompt_metadata,
            'grounded_relation_alignment': config.grounded_relation_alignment,
            'grounded_relation_cache_path': config.grounded_relation_cache_path,
            'grounded_relation_metadata': data.grounded_relation_metadata,
            'relation_alignment_weight': config.relation_alignment_weight,
            'relation_alignment_margin': config.relation_alignment_margin,
            'grounded_relation_version': config.grounded_relation_version,
            'relation_wrong_image_weight': config.relation_wrong_image_weight,
            'relation_verified_scale': config.relation_verified_scale,
            'relation_verifier_frozen': config.grounded_relation_version == 'verified_v3',
            'verified_visual_projection_frozen': config.grounded_relation_version == 'verified_v3',
            'caption_embedding_cache_path': config.caption_embedding_cache_path,
            'itc_weight': config.itc_weight,
            'itc_temperature': config.itc_temperature,
            'caption_word2idx': data.tokenizer.word2idx,
            'cross_attention': 'query=prompt, key=visual, value=visual',
            'epoch': epoch,
            'loss': losses['loss'],
            'caption_loss': losses['caption_loss'],
            'alignment_loss': losses['alignment_loss'],
            'itc_loss': losses['itc_loss'],
            'relation_loss': losses['relation_loss'],
            'model_name': MODEL_NAME,
            'embed_dim': EMBED_DIM,
            'max_prompt_len': MAX_PROMPT_LEN,
            'num_attention_heads': NUM_HEADS,
            'vocab_size': data.vocab_size,
            'encoder_state_dict': base_model.encoder.state_dict(),
            'decoder_state_dict': base_model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        accelerator.print(f'✅ Epoch {epoch}/{config.epochs} | total: {losses["loss"]:.4f} | '
                          f'caption: {losses["caption_loss"]:.4f} | '
                          f'alignment: {losses["alignment_loss"]:.4f} | '
                          f'relation: {losses["relation_loss"]:.4f} | '
                          f'itc: {losses["itc_loss"]:.4f}')
        accelerator.print(f'💾 Saved: {checkpoint_path}')

    history_path = config.work_dir / config.experiment_name / 'train_history_h1_2.json'
    if accelerator.is_main_process:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    accelerator.wait_for_everyone()
    accelerator.print(f'✅ Training complete. History: {history_path}')
    is_main_process = accelerator.is_main_process
    accelerator.end_training()
    return is_main_process
