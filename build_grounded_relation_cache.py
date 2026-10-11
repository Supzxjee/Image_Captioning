"""Build instance-pair relation embeddings and counterfactual predicates for training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch
from PIL import Image

from captioning.detection_audit import dataset_records


COUNTERFACTUAL_PREDICATE = {
    'left_of': 'right_of', 'right_of': 'left_of',
    'above': 'below', 'below': 'above',
    'inside': 'on', 'on': 'inside',
    'holding': 'wearing', 'wearing': 'holding',
    'carrying': 'wearing', 'riding': 'holding',
    'overlapping': 'near', 'near': 'inside',
}


def _index_by_filename(mapping, label):
    indexed = {}
    for key, value in mapping.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous {label} filename: {filename}')
        indexed[filename] = value
    return indexed


def _phrase(subject, predicate, obj):
    return f'a photo showing {subject} {predicate.replace("_", " ")} {obj}'


def _embed_phrases(phrases, model_name, batch_size):
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPTextModel.from_pretrained(model_name).float().to(device).eval()
    result = {}
    ordered = sorted(phrases)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            eos = encoded['attention_mask'].sum(dim=1) - 1
            vectors = hidden[torch.arange(len(batch), device=device), eos].cpu().half()
        result.update(zip(batch, vectors))
        print(f'Embedded relation phrases: {min(start + len(batch), len(ordered))}/'
              f'{len(ordered)}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relations', required=True,
                        help='V2.1 relation JSON containing objects and triplets.')
    parser.add_argument('--dataset-json-path', required=True)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-relations', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-name', default='openai/clip-vit-base-patch16')
    args = parser.parse_args(argv)
    if args.max_relations <= 0 or args.batch_size <= 0:
        parser.error('max-relations and batch-size must be positive.')

    relation_path = Path(args.relations)
    dataset_path = Path(args.dataset_json_path)
    relations = json.loads(relation_path.read_text(encoding='utf-8'))
    relations = relations.get('data', relations) if isinstance(relations, dict) else relations
    if not isinstance(relations, dict):
        raise ValueError('Relation JSON must be a mapping.')
    relation_index = _index_by_filename(relations, 'relation')
    records = dataset_records(dataset_path)
    base_path = Path(args.base_path)

    prepared, phrases = {}, set()
    predicate_counts = {}
    for index, (filename, record) in enumerate(records.items(), 1):
        entry = relation_index.get(filename)
        if entry is None:
            raise ValueError(f'Missing relation entry for {filename}')
        objects = {item['instance_id']: item for item in entry.get('objects', [])}
        image_path = base_path / record.get('filepath', '') / filename
        with Image.open(image_path) as image:
            width, height = image.size
        items = []
        for triplet in entry.get('triplets', [])[:args.max_relations]:
            subject = objects.get(triplet.get('subject_id'))
            obj = objects.get(triplet.get('object_id'))
            predicate = triplet.get('relation')
            if subject is None or obj is None or predicate not in COUNTERFACTUAL_PREDICATE:
                continue
            subject_box = [subject['bbox'][0] / width, subject['bbox'][1] / height,
                           subject['bbox'][2] / width, subject['bbox'][3] / height]
            object_box = [obj['bbox'][0] / width, obj['bbox'][1] / height,
                          obj['bbox'][2] / width, obj['bbox'][3] / height]
            positive = _phrase(triplet['subject'], predicate, triplet['object'])
            negative_predicate = COUNTERFACTUAL_PREDICATE[predicate]
            negative = _phrase(triplet['subject'], negative_predicate, triplet['object'])
            phrases.update((positive, negative))
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
            items.append({
                'positive': positive,
                'negative': negative,
                'subject_box': subject_box,
                'object_box': object_box,
                'confidence': min(float(subject['conf']), float(obj['conf'])),
                'triplet': triplet,
                'negative_predicate': negative_predicate,
            })
        prepared[filename] = items
        if index == 1 or index % 10000 == 0:
            print(f'Prepared grounded relations: {index}/{len(records)}', flush=True)

    embeddings = _embed_phrases(phrases, args.model_name, args.batch_size)
    data = {}
    for filename, items in prepared.items():
        tokens = torch.zeros(args.max_relations, 512, dtype=torch.float16)
        negative_tokens = torch.zeros_like(tokens)
        subject_boxes = torch.zeros(args.max_relations, 4)
        object_boxes = torch.zeros(args.max_relations, 4)
        confidences = torch.zeros(args.max_relations)
        mask = torch.zeros(args.max_relations, dtype=torch.long)
        metadata = []
        for relation_index_value, item in enumerate(items):
            tokens[relation_index_value] = embeddings[item['positive']]
            negative_tokens[relation_index_value] = embeddings[item['negative']]
            subject_boxes[relation_index_value] = torch.tensor(item['subject_box'])
            object_boxes[relation_index_value] = torch.tensor(item['object_box'])
            confidences[relation_index_value] = item['confidence']
            mask[relation_index_value] = 1
            metadata.append({
                **item['triplet'],
                'counterfactual_relation': item['negative_predicate'],
            })
        data[filename] = {
            'tokens': tokens, 'negative_tokens': negative_tokens,
            'subject_boxes': subject_boxes, 'object_boxes': object_boxes,
            'confidences': confidences, 'mask': mask, 'relations': metadata,
        }

    metadata = {
        'variant': 'instance_pair_grounded_relation_v1',
        'complete': len(data) == len(records),
        'count': len(data),
        'images_with_relations': sum(bool(items) for items in prepared.values()),
        'total_relations': sum(len(items) for items in prepared.values()),
        'predicate_distribution': predicate_counts,
        'max_relations': args.max_relations,
        'coordinate_format': 'normalized_xyxy',
        'text_encoder': args.model_name,
        'feature': 'CLIPTextModel.last_hidden_state at EOS',
        'counterfactual_strategy': 'predicate swap with fixed subject/object instances',
        'relation_source_sha256': hashlib.sha256(relation_path.read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    torch.save({'data': data, 'metadata': metadata}, temporary)
    temporary.replace(destination)
    destination.with_suffix('.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
