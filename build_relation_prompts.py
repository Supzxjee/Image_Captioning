"""Build semantic, instance-aware object-relation prompts from saved YOLO boxes."""
import argparse
import json
from collections import Counter
from pathlib import Path

from captioning.relation_prompts import build_relation_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--bbox-key', default='bbox')
    parser.add_argument('--confidence-key', default='conf')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--max-objects', type=int, default=10)
    parser.add_argument('--max-relations', type=int, default=3)
    parser.add_argument('--duplicate-iou', type=float, default=0.8)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; 0 processes the full cache.')
    args = parser.parse_args(argv)
    if not 0 <= args.min_confidence <= 1:
        parser.error('--min-confidence must be in [0, 1].')
    if args.max_objects < 1 or args.max_relations < 0 or args.limit < 0:
        parser.error('--max-objects must be positive; max-relations/limit nonnegative.')
    if not 0 < args.duplicate_iou <= 1:
        parser.error('--duplicate-iou must be in (0, 1].')

    source_path = Path(args.source)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if args.limit:
        payload = source.get('data', source) if isinstance(source, dict) else source
        if not isinstance(payload, dict):
            raise ValueError('Source must be a mapping.')
        source = dict(list(payload.items())[:args.limit])
    cache = build_relation_cache(
        source,
        objects_field=args.objects_field,
        name_key=args.name_key,
        bbox_key=args.bbox_key,
        confidence_key=args.confidence_key,
        min_confidence=args.min_confidence,
        max_objects=args.max_objects,
        max_relations=args.max_relations,
        duplicate_iou=args.duplicate_iou,
    )
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    temporary.replace(destination)

    relation_counts = Counter(
        item['relation'] for entry in cache.values() for item in entry['triplets'])
    summary = {
        'heuristic_version': 'H1_2_semantic_instance_aware_v2',
        'images': len(cache),
        'images_with_relations': sum(bool(entry['triplets']) for entry in cache.values()),
        'total_relations': sum(relation_counts.values()),
        'relation_distribution': dict(relation_counts.most_common()),
        'configuration': {
            'min_confidence': args.min_confidence,
            'max_objects': args.max_objects,
            'max_relations': args.max_relations,
            'duplicate_iou': args.duplicate_iou,
        },
    }
    destination.with_suffix('.summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Saved:', destination, flush=True)


if __name__ == '__main__':
    main()
