"""Compare matched baseline and OCC outputs produced by a CHAIR evaluator."""
import argparse
import json
from pathlib import Path


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _index(payload, label):
    sentences = payload.get('sentences')
    metrics = payload.get('overall_metrics')
    if not isinstance(sentences, list) or not sentences or not isinstance(metrics, dict):
        raise ValueError(f'{label}: invalid CHAIR output.')
    indexed = {}
    for sentence in sentences:
        image_id = int(sentence['image_id'])
        if image_id in indexed:
            raise ValueError(f'{label}: duplicate image id {image_id}.')
        indexed[image_id] = sentence
    return metrics, indexed


def summarize(baseline_path, occ_path):
    baseline_metrics, baseline = _index(_read(baseline_path), 'baseline')
    occ_metrics, occ = _index(_read(occ_path), 'OCC')
    if set(baseline) != set(occ):
        raise ValueError('Baseline and OCC CHAIR outputs cover different images.')

    transitions = {
        'both_clean': 0,
        'improved_to_clean': 0,
        'regressed_to_hallucinated': 0,
        'both_hallucinated': 0,
    }
    changed_captions = 0
    examples = []
    for image_id in sorted(baseline):
        before, after = baseline[image_id], occ[image_id]
        before_h = bool(before['metrics']['CHAIRs'])
        after_h = bool(after['metrics']['CHAIRs'])
        if not before_h and not after_h:
            key = 'both_clean'
        elif before_h and not after_h:
            key = 'improved_to_clean'
        elif not before_h and after_h:
            key = 'regressed_to_hallucinated'
        else:
            key = 'both_hallucinated'
        transitions[key] += 1
        if before.get('caption') != after.get('caption'):
            changed_captions += 1
            examples.append({
                'image_id': image_id,
                'transition': key,
                'baseline_caption': before.get('caption'),
                'occ_caption': after.get('caption'),
                'baseline_hallucinated_words': before.get('mscoco_hallucinated_words', []),
                'occ_hallucinated_words': after.get('mscoco_hallucinated_words', []),
            })

    keys = ('CHAIRs', 'CHAIRi', 'Recall')
    return {
        'count': len(baseline),
        'changed_captions': changed_captions,
        'baseline': {key: float(baseline_metrics[key]) for key in keys},
        'occ': {key: float(occ_metrics[key]) for key in keys},
        'delta_occ_minus_baseline': {
            key: float(occ_metrics[key]) - float(baseline_metrics[key]) for key in keys
        },
        'interpretation': 'Lower CHAIRs/CHAIRi is better; higher Recall is better.',
        'sentence_transitions': transitions,
        'changed_examples': examples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-chair', required=True)
    parser.add_argument('--occ-chair', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = summarize(args.baseline_chair, args.occ_chair)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'changed_examples'}, indent=2), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
