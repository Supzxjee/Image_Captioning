"""Diagnostic bridge: retain detector boxes without double normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captioning.selected_grounding_bridge import prepare_selected, pack_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clip-model-path', required=True)
    parser.add_argument('--revision', default='57c216476eefef5ab752ec549e440a49ae4ae5f3')
    parser.add_argument('--max-relations', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    source = Path(args.selected)
    selected = json.loads(source.read_text(encoding='utf-8'))
    prepared = {name: prepare_selected(entry, args.max_relations) for name, entry in selected.items()}
    phrases = sorted({item[key] for items in prepared.values() for item in items
                      for key in ('positive', 'negative')})
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    from transformers import CLIPTextModel, CLIPTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path, revision=args.revision)
    model = CLIPTextModel.from_pretrained(args.clip_model_path, revision=args.revision).float().to(device).eval()
    embeddings = {}
    for start in range(0, len(phrases), args.batch_size):
        batch = phrases[start:start + args.batch_size]
        encoded = tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(device)
        if encoded['input_ids'].shape[1] > model.config.max_position_embeddings:
            raise ValueError('Relation phrase too long; refusing silent truncation')
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            eos = encoded['attention_mask'].sum(1) - 1
            vectors = hidden[torch.arange(len(batch), device=device), eos].cpu().half()
        embeddings.update(zip(batch, vectors))
        print(f'Embedded {len(embeddings)}/{len(phrases)} phrases', flush=True)
    data = {name: pack_selected(items, embeddings, args.max_relations) for name, items in prepared.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(variant='selected_detector_grounded_diagnostic_v1', complete=True,
        coverage_scope='selected JSON only, not complete COCO', count=len(data),
        coordinate_format='normalized_xyxy', text_encoder=args.clip_model_path,
        text_encoder_revision=args.revision,
        feature='CLIPTextModel.last_hidden_state at EOS',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        confidence_policy='min subject/object detector confidence; relation scores are not probabilities',
        counterfactual_strategy='contrast phrases may coexist; not verified negative supervision',
        inference_only=True)
    torch.save(dict(data=data, metadata=metadata), destination)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
