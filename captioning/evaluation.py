"""Validation/test inference, progress snapshots and COCO metrics."""
import json
import time
from pathlib import Path
from tqdm.auto import tqdm
from .data import load_visual_input
from .checkpoints import load_checkpoint
from .inference import generate_caption_beam_search
from .metrics import export_ground_truth, score_files

def evaluate_model(model, data, config):
    path = Path(config.checkpoint) if config.checkpoint else config.checkpoint_dir / f'model_h1_2_crossattn_epoch_{config.epochs}.pth'
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = load_checkpoint(model, path, data.tokenizer)
    model.eval()
    print(f"Loaded H1.2-G checkpoint: epoch {checkpoint['epoch']} | loss {checkpoint['loss']:.4f}", flush=True)
    config.eval_dir.mkdir(parents=True, exist_ok=True)
    eval_df = data.val_df if config.split == 'val' else data.test_df
    if config.limit:
        eval_df = eval_df.head(config.limit)
    if len(eval_df) == 0:
        raise ValueError('Evaluation split is empty.')
    print(f'Starting {config.split} inference: {len(eval_df)} images, beam size 5', flush=True)
    started = time.monotonic()
    predictions = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f'Generating {config.split} captions'):
        image_path = row['image']
        prompt_entry = data.prompt_cache[image_path]
        image_tensor = load_visual_input(row, data.transform, data.visual_cache)
        caption = generate_caption_beam_search(
            model=model,
            image=image_tensor,
            cached_prompt_tokens=prompt_entry['tokens'],
            prompt_mask=prompt_entry['mask'],
            tokenizer=data.tokenizer,
            beam_size=5,
            device=config.device,
        )
        predictions.append({'image_id': int(row['eval_id']), 'caption': caption})
        if len(predictions) % 50 == 0:
            elapsed = time.monotonic() - started
            rate = elapsed / len(predictions)
            print(f'Generated {len(predictions)}/{len(eval_df)} | {rate:.2f}s/image | ETA {(len(eval_df)-len(predictions))*rate/60:.1f}min', flush=True)
            partial_path = config.eval_dir / f'{config.split}_captions_partial.json'
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False)


    predictions_path = config.eval_dir / f'{config.split}_{len(eval_df)}_captions_h1_2_gated.json'
    ground_truth_path = config.eval_dir / f'{config.split}_{len(eval_df)}_gt_h1_2_gated.json'
    metrics_path = config.eval_dir / f'{config.split}_{len(eval_df)}_metrics_h1_2_gated.json'

    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    export_ground_truth(eval_df, ground_truth_path)

    print(f'Predictions saved: {predictions_path}', flush=True)
    print(f'Ground truth saved: {ground_truth_path}', flush=True)
    if config.mode == 'predict':
        return predictions_path, ground_truth_path
    try:
        score_files(predictions_path, ground_truth_path, metrics_path)
    except Exception:
        print('Scoring failed; captions and ground truth are already saved. Retry with --mode metrics; do not regenerate captions.', flush=True)
        raise
    return predictions_path, ground_truth_path
