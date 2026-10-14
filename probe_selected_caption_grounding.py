"""Frozen-checkpoint intervention: on/off/alternate target, same generic prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captioning.checkpoints import load_checkpoint
from captioning.grounded_relations import align_grounded_relation_cache
from captioning.inference import generate_caption_candidates
from captioning.tokenizer import CaptionTokenizer
from captioning.visual_cache import VisualCache, coco_image_id


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    return intersection / max(area(a) + area(b) - intersection, 1e-8)


def alternate_target(selected, entry):
    """Deterministic low-overlap same-class target; correctness is UNKNOWN."""
    for slot, claim in enumerate(entry['relations']):
        if not entry['mask'][slot]:
            continue
        options = sorted(selected['objects'], key=lambda obj: obj['instance_id'])
        for obj in options:
            if (obj['label'] == claim['object'] and obj['instance_id'] not in
                    {claim['subject_instance'], claim['object_instance']} and
                    iou(obj['bbox'], entry['object_boxes'][slot].tolist()) < .5):
                swapped = dict(entry)
                swapped['object_boxes'] = entry['object_boxes'].clone()
                swapped['object_boxes'][slot] = torch.tensor(obj['bbox'])
                return swapped, dict(slot=slot, original_target=claim['object_instance'],
                                     alternate_target=obj['instance_id'], alternate_box=obj['bbox'],
                                     status='unknown; intervention only, not negative label')
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--selected', required=True)
    parser.add_argument('--grounded-cache', required=True)
    parser.add_argument('--prompt-cache', required=True)
    parser.add_argument('--visual-cache', required=True, action='append')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-size', type=int, default=12)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--candidate-count', type=int, default=3)
    args = parser.parse_args()
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    if args.sample_size < 1 or not 1 <= args.candidate_count <= args.beam_size:
        raise ValueError('Invalid sample/candidate count')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory to preserve prior results')
    output.mkdir(parents=True)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (not saved.get('grounded_relation_alignment') or
            saved.get('object_semantic_alignment') or saved.get('cascade_semantic_alignment') or
            saved.get('visual_adapter') != 'qformer'):
        raise ValueError('Require a pair-grounded Q-Former caption checkpoint')
    vocabulary = saved.get('caption_word2idx')
    if not vocabulary or sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError('Checkpoint must contain contiguous caption_word2idx')
    tokenizer = CaptionTokenizer([])
    for key, value in tokenizer.word2idx.items():
        if vocabulary.get(key) != value:
            raise ValueError('Special vocabulary IDs differ')
    tokenizer.word2idx = vocabulary
    tokenizer.idx2word = {value: key for key, value in vocabulary.items()}
    encoder = UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
        num_visual_queries=saved['num_visual_queries'], qformer_layers=saved['qformer_layers'],
        prompt_conditioned_qformer=saved.get('prompt_conditioned_qformer', False),
        use_itc=saved.get('itc_weight', 0) > 0, grounded_relation_alignment=True,
        grounded_relation_version=saved.get('grounded_relation_version', 'v1'),
        relation_verified_scale=saved.get('relation_verified_scale', .1))
    del saved
    model = ImageCaptioningModel(encoder, CaptionDecoder(len(vocabulary))).to(device).eval()
    load_checkpoint(model, args.checkpoint, tokenizer)  # Strict; no warm start or random new heads.
    selected = json.loads(Path(args.selected).read_text(encoding='utf-8'))
    grounded = torch.load(args.grounded_cache, map_location='cpu', weights_only=False)
    if grounded['metadata'].get('source_sha256') != hashlib.sha256(Path(args.selected).read_bytes()).hexdigest():
        raise ValueError('Grounded cache does not match selected JSON')
    grounded = align_grounded_relation_cache(grounded['data'], list(selected))
    prompts = torch.load(args.prompt_cache, map_location='cpu', weights_only=False)['data']
    indexed = {}
    for key, value in prompts.items():
        name = str(key).replace('\\', '/').split('/')[-1]
        if name in indexed:
            raise ValueError(f'Ambiguous prompt filename: {name}')
        indexed[name] = value
    eligible = []
    for name in sorted(selected):
        swapped, metadata = alternate_target(selected[name], grounded[name])
        if swapped is not None:
            eligible.append((name, swapped, metadata))
    if not eligible:
        raise ValueError('No same-class alternative targets available')
    cohort = eligible[:args.sample_size]
    visual_cache = VisualCache(args.visual_cache, id_key='coco_id')
    records = []
    try:
        visual_cache.require_ids([coco_image_id(name) for name, _, _ in cohort], 'probe')
        for name, swapped, intervention in cohort:
            if name not in indexed:
                raise ValueError(f'Missing original generic prompt: {name}')
            original = grounded[name]
            off = dict(original); off['mask'] = torch.zeros_like(original['mask'])
            image, prompt = torch.from_numpy(visual_cache.read(coco_image_id(name))), indexed[name]
            outcomes = {}; baseline_memory = None
            for condition, entry in [('on', original), ('off', off), ('target_swap', swapped)]:
                with torch.inference_mode():
                    kwargs = dict(relation_tokens=entry['tokens'][None].float().to(device),
                        negative_relation_tokens=entry['negative_tokens'][None].float().to(device),
                        subject_boxes=entry['subject_boxes'][None].to(device),
                        object_boxes=entry['object_boxes'][None].to(device),
                        relation_confidences=entry['confidences'][None].to(device),
                        relation_mask=entry['mask'][None].to(device), return_relation_diagnostics=True)
                    memory, _, diagnostics = model.build_memory(image[None].to(device),
                        prompt['tokens'][None].float().to(device), prompt['mask'][None].to(device), **kwargs)
                    if baseline_memory is None:
                        baseline_memory = memory.clone()
                    delta = float((memory - baseline_memory).norm() / baseline_memory.norm().clamp_min(1e-8))
                    candidates = generate_caption_candidates(model, image, prompt['tokens'],
                        prompt['mask'], tokenizer, beam_size=args.beam_size,
                        candidate_count=args.candidate_count, device=device, relation_entry=entry)
                    valid = diagnostics['mask'].bool()
                    outcomes[condition] = dict(candidates=candidates, memory_relative_change=delta,
                        valid_relations=int(valid.sum()),
                        accepted_relations=int(diagnostics['accepted_mask'].sum())
                            if 'accepted_mask' in diagnostics else None,
                        evidence_weights=diagnostics['evidence_gate'][valid].cpu().tolist())
            top = lambda key: outcomes[key]['candidates'][0]['caption']
            records.append(dict(filename=name, intervention=intervention, conditions=outcomes,
                off_top_changed=top('on') != top('off'), swap_top_changed=top('on') != top('target_swap'),
                off_candidates_changed=outcomes['on']['candidates'] != outcomes['off']['candidates'],
                swap_candidates_changed=outcomes['on']['candidates'] != outcomes['target_swap']['candidates']))
            (output / 'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            print(f'{len(records)}/{len(cohort)} {name}: off={records[-1]["off_top_changed"]}, '
                  f'swap={records[-1]["swap_top_changed"]}', flush=True)
    finally:
        visual_cache.close()
    summary = dict(sample_count=len(records), eligible_images=len(eligible),
        selection='first lexicographic filenames with same-class low-overlap alternative; inspected development',
        off_top_changed=sum(r['off_top_changed'] for r in records),
        swap_top_changed=sum(r['swap_top_changed'] for r in records),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        grounded_sha256=hashlib.sha256(Path(args.grounded_cache).read_bytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(Path(args.prompt_cache).read_bytes()).hexdigest(),
        beam_size=args.beam_size, candidate_count=args.candidate_count,
        notes='Inference only. Generic relation prompt retained in all conditions. Alternate targets '
              'are unknown, not negatives. Caption change is sensitivity, not quality. '
              'Candidates_changed includes logprob changes, not only changed wording. '
              'Historical checkpoint and new selected relations may differ in distribution.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
