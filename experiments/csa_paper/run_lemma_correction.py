"""USER-RUN CPU subset recount; preserve original words, captions and ROI126."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import traceback

from lemma_correction import corrected_lemma, corrected_bank

DATA_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
BANK_SHA = '450fc969e0842ca3270d1f899b7e8fe594639260644ad5815a69430fe48264b1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(dataset_path, out, report):
    import spacy
    here = Path(__file__).parent
    baseline = here/'original_bank_words.json'
    if sha(dataset_path) != DATA_SHA or sha(baseline) != BANK_SHA:
        raise ValueError('Original dataset/bank checksum differs')
    original = json.loads(baseline.read_text(encoding='utf-8'))
    environment = json.loads((here/'source_environment.json').read_text(encoding='utf-8'))
    versions = {n: importlib.metadata.version(n) for n in ('spacy', 'thinc', 'numpy', 'en-core-web-sm')}
    if versions != environment['versions']:
        raise ValueError('POS versions differ from126')
    spacy.require_cpu()
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    if set(nlp.pipe_names) != {'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'}:
        raise ValueError('POS pipeline differs')
    model_hashes = {str(p.relative_to(nlp.path)): sha(p) for p in sorted(Path(nlp.path).rglob('*')) if p.is_file()}
    if model_hashes != environment['pos_model_files_sha256']:
        raise ValueError('POS model data differs from126')
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8'))
    pattern = re.compile(r'\b(?:riding|rides)\b', re.I)
    report['train_caption_events_scanned'] = 0
    def selected():
        for image in dataset['images']:
            if image['split'] not in ('train', 'restval'):
                continue
            for sentence_index, sentence in enumerate(image['sentences']):
                report['train_caption_events_scanned'] += 1
                if pattern.search(sentence['raw']):
                    yield sentence['raw'], dict(coco_id=image['cocoid'], filename=image['filename'],
                        source_split=image['split'], sentence_index=sentence_index, caption=sentence['raw'])
    moved = 0
    documents = 0
    by_surface = Counter()
    example_counts = Counter()
    examples = []
    ledger = hashlib.sha256()
    for number, (doc, context) in enumerate(nlp.pipe(selected(), as_tuples=True, batch_size=256, n_process=1), 1):
        documents = number
        for token_index, token in enumerate(doc):
            new = corrected_lemma(token.text, token.pos_, token.lemma_)
            if new == token.lemma_:
                continue
            if token.pos_ != 'VERB' or token.text.lower() not in ('riding', 'rides') or token.lemma_.lower() != 'rid' or new != 'ride':
                raise ValueError('Change outside the approved narrow rule')
            moved += 1
            by_surface[token.text.lower()] += 1
            record = dict(context, token_index=token_index, surface=token.text, pos=token.pos_,
                          tag=token.tag_, original_lemma=token.lemma_, corrected_lemma=new)
            ledger.update((json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'))+'\n').encode())
            if example_counts[token.text.lower()] < 10:
                examples.append(record)
                example_counts[token.text.lower()] += 1
        if number % 1000 == 0:
            print('Affected TRAIN captions processed:', number, '; redirected tokens:', moved, flush=True)
    if report['train_caption_events_scanned'] != 566747:
        raise ValueError('TRAIN caption coverage differs')
    corrected, delta = corrected_bank(original, moved)
    if corrected['noun'] != original['noun'] or sha(baseline) != BANK_SHA:
        raise ValueError('Original nouns or bank changed')
    corrected['lemma_amendment']['source_bank_sha256'] = BANK_SHA
    save(out/'bank_words_corrected.json', corrected)
    save(out/'correction_examples.json', examples)
    report.update(status='BANK_LEMMA_CORRECTION_COMPLETE', affected_documents_pos_processed=documents,
        delta=delta, redirected_by_surface=dict(by_surface), full_change_ledger_sha256=ledger.hexdigest(),
        versions=versions, model_files_matched=len(model_hashes), noun_words=len(corrected['noun']),
        verb_words=len(corrected['verb']), examples_saved=len(examples))
    save(out/'correction_receipt.json', dict(status=report['status'],
        amendment='VERB + riding/rides + original rid -> ride; everything else unchanged',
        source_dataset_sha256=DATA_SHA, source_bank_sha256=BANK_SHA, preserved_roi_source_version=126,
        corrected_bank_sha256=sha(out/'bank_words_corrected.json'), delta=delta,
        nouns_unchanged=True, caption_targets_changed=False, roi_modified=False,
        semantic_embeddings_created=False, caption_model_executed=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset-json', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = dict(status='INCOMPLETE', original_bank_modified=False, roi_modified=False,
                  caption_targets_changed=False, semantic_embeddings_created=False,
                  caption_model_executed=False, gpu_used=False, dataset_created=False)
    start = time.monotonic()
    try:
        run(Path(args.dataset_json), out, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['seconds'] = time.monotonic()-start
        save(out/'summary.json', report)


if __name__ == '__main__':
    main()
