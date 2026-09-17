"""Compare a deterministic sample of cached tokens to the current frozen CLIP."""
import torch
from .data import load_visual_input


def verify_cache(encoder, data, config):
    df = data.val_df if config.split == 'val' else data.test_df
    # Default small sample, not full split; spread checks across selected images.
    count = min(config.limit or 10, len(df))
    if not count:
        raise ValueError('Verification split is empty.')
    selected = df.head(config.limit) if config.limit else df
    positions = torch.linspace(0, len(selected)-1, count).long().tolist()
    encoder.eval()
    bad = []
    for position in positions:
        row = selected.iloc[position]
        cached = load_visual_input(row, data.transform, data.visual_cache).to(config.device)
        pixels = load_visual_input(row, data.transform).unsqueeze(0).to(config.device)
        with torch.no_grad():
            direct = encoder.extract_visual_features(pixels)[0].half().float()
        delta = (direct - cached).abs()
        # Allow float16 quantization; preprocessing/model mismatches should fail.
        matches = torch.allclose(direct, cached, atol=0.01, rtol=0.005)
        print(f"COCO {row['coco_id']}: mean_abs={delta.mean().item():.6f}, max_abs={delta.max().item():.6f}, pass={matches}", flush=True)
        if not matches:
            bad.append(row['coco_id'])
    if bad:
        raise RuntimeError(f'Visual cache differs from current CLIP/preprocessing: {bad}. Do not use as an equivalent speed optimization.')
    print(f'PASS: {count} sampled images match within FP16 tolerance. This does not certify every cache row.', flush=True)
