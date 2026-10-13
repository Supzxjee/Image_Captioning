"""Build every ordered person-object pair within a retained detector inventory."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image, ImageDraw
from captioning.relation_prompts import select_detections
from captioning.detector_pair_matching import normalized_detection_box
from captioning.detector_candidates import person_object_pairs
from captioning.selected_relation_prompts import entity_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-list', required=True)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--coordinate-format', choices=['pixel_xyxy', 'normalized_xyxy'], default='pixel_xyxy')
    args = parser.parse_args()
    selection = json.loads(Path(args.image_list).read_text())
    source_path = Path(args.detections)
    source = json.loads(source_path.read_text())
    if not isinstance(source, dict):
        raise ValueError('Expected detection filename mapping.')
    payload = source.get('data', source)
    if not isinstance(payload, dict):
        raise ValueError('Expected detection filename mapping.')
    indexed = {}
    for key, value in payload.items():
        name = PurePosixPath(str(key).replace('\\', '/')).name
        if name in indexed:
            raise ValueError('Duplicate detection filename.')
        indexed[name] = value
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inventory, pairs, issues = {}, [], Counter()
    for record in selection['images']:
        filename = record['filename']
        if filename not in indexed:
            raise ValueError(f'Missing detector cache record for {filename}; not an empty detection list.')
        with Image.open(Path(args.images) / record['filepath'] / filename) as source_image:
            full = source_image.convert('RGB')
            width, height = full.size
            kept = select_detections(indexed[filename], min_confidence=.5, max_objects=10, duplicate_iou=.8)
            objects = []
            for item in kept:
                if not 0 <= item.confidence <= 1:
                    issues['confidence_out_of_range'] += 1
                    continue
                try:
                    box = normalized_detection_box(item.box, width, height, args.coordinate_format)
                except ValueError:
                    issues['invalid_box'] += 1
                    continue
                objects.append({'label': item.label, 'bbox': box, 'confidence': item.confidence})
            inventory[filename] = objects
            pairs += person_object_pairs(record, objects)
            marked = full.copy()
            draw = ImageDraw.Draw(marked)
            for obj in objects:
                box = tuple(v*(width if i%2==0 else height) for i, v in enumerate(obj['bbox']))
                draw.rectangle(box, outline='red' if obj['label']=='person' else 'blue', width=2)
                draw.text((box[0]+2, box[1]+2), obj['label']+' '+entity_id(filename,obj['label'],obj['bbox']), fill='yellow')
            # Original resolution retained for reviewing contact/identity evidence.
            full.save(output / (Path(filename).stem+'_full.jpg'), quality=95)
            marked.save(output / (Path(filename).stem+'_marked.jpg'), quality=95)
    pair_file = output / 'evaluation_pairs.json'
    pair_file.write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    (output / 'object_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    summary = {'variant': 'detector_full_candidate_audit_20_v1', 'images': len(inventory),
               'candidate_pairs': len(pairs), 'retained_objects': sum(len(x) for x in inventory.values()),
               'images_without_person': sum(not any(o['label']=='person' for o in x) for x in inventory.values()),
               'issues': dict(issues), 'min_confidence': .5, 'max_objects': 10, 'duplicate_iou': .8,
               'coordinate_format': args.coordinate_format,
               'image_list_sha256': hashlib.sha256(Path(args.image_list).read_bytes()).hexdigest(),
               'pair_json_sha256': hashlib.sha256(pair_file.read_bytes()).hexdigest(),
               'notes': 'All person-object pairs within top10 retained detections; no GT matching/labels. '
                        '20 images from previously evaluated cohort, qualitative audit not new heldout benchmark.'}
    digest = hashlib.sha256()
    with source_path.open('rb') as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    summary['detection_cache_sha256'] = digest.hexdigest()
    (output / 'full_candidate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
