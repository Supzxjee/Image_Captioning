"""USER-RUN CPU POS + actual ROI preparation; no CLIP/caption model or optimizer."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import traceback
import unicodedata
import re

import numpy as np

from audit_assets import DATA_SHA, DETECTION_SHA, coco_id, detection_index, sha, text
from data import bank_words, encode, roi_pool, vocabulary

VOCAB_SHA = '924be210aa0daa8b0968a6d69ad2c66b3de603f49c35f4aa67247c75d6173fc7'
MODEL_WHEEL_SHA = '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def select_pilot(dataset):
    def ordered(splits, count):
        rows = [r for r in dataset['images'] if r['split'] in splits]
        rows.sort(key=lambda r: hashlib.sha256(('csa-paper-pilot-v1/42/'+r['filename']).encode()).hexdigest())
        if len(rows) < count:
            raise ValueError('Insufficient pilot images')
        return rows[:count]
    train, val = ordered(('train', 'restval'), 256), ordered(('val',), 64)
    ids = [coco_id(r['filename']) for r in train + val]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate/overlapping pilot images')
    return train, val


def raw_train_text(dataset):
    for image in dataset['images']:
        if image['split'] in ('train', 'restval'):
            for sentence in image['sentences']:
                raw = sentence['raw']
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError('Missing original TRAIN raw caption')
                yield raw


def make_word_lists(dataset, nlp, stats, progress=5000):
    corpus_hash = hashlib.sha256()
    def texts():
        for raw in raw_train_text(dataset):
            payload = raw.encode('utf-8')
            corpus_hash.update(len(payload).to_bytes(8, 'little')); corpus_hash.update(payload)
            yield raw
    rejected = Counter()
    stats['caption_documents'] = 0
    def tagged():
        for number, doc in enumerate(nlp.pipe(texts(), batch_size=256, n_process=1), 1):
            stats['caption_documents'] = number
            if progress and number % progress == 0:
                print('TRAIN POS documents:', number, '/566747', flush=True)
            for token in doc:
                word = unicodedata.normalize('NFKC', token.lemma_).lower()
                if token.pos_ not in ('NOUN', 'PROPN', 'VERB'):
                    rejected['other_pos_including_aux'] += 1
                elif not re.fullmatch('[a-z]+', word):
                    rejected['not_ascii_alphabetic_lemma'] += 1
                yield token.lemma_, token.pos_
    words = bank_words(tagged(), minimum=5)
    if not words['noun'] or not words['verb']:
        raise ValueError('Empty semantic word list')
    stats['corpus_sha256_length_prefixed_utf8'] = corpus_hash.hexdigest()
    stats['rejected_token_counts'] = dict(rejected)
    return words


def pack_regions(records):
    if not records:
        raise ValueError('Empty ROI dataset')
    offsets = [0]
    for feature, boxes, scores, identity in records:
        if feature.ndim != 2 or feature.shape[1] != 768 or not 1 <= len(feature) <= 150:
            raise ValueError('Invalid raw768 ROI count/shape')
        if boxes.shape != (len(feature), 4) or scores.shape != (len(feature),):
            raise ValueError('ROI metadata lengths differ')
        if not all(np.isfinite(a).all() for a in (feature, boxes, scores)):
            raise ValueError('Nonfinite ROI payload')
        offsets.append(offsets[-1] + len(feature))
    packed = dict(features=np.concatenate([r[0] for r in records]).astype(np.float16),
                  boxes=np.concatenate([r[1] for r in records]).astype(np.float32),
                  scores=np.concatenate([r[2] for r in records]).astype(np.float32),
                  offsets=np.asarray(offsets, dtype=np.int64), ids=np.asarray([r[3] for r in records], dtype=np.int64))
    if not np.isfinite(packed['features']).all() or len(set(packed['ids'].tolist())) != len(records):
        raise ValueError('FP16 overflow or duplicate ROI images')
    return packed


def visual_index(root, selected):
    import h5py
    paths = sorted(Path(root).rglob('*.h5'))
    index, profiles = {}, []
    for path in paths:
        with h5py.File(path, 'r') as f:
            if not bool(f.attrs.get('complete', False)) or text(f.attrs.get('id_key', '')) != 'coco_id':
                raise ValueError('H5 completion/ID declaration differs')
            meta = json.loads(text(f.attrs['metadata_json']))
            required = dict(model_id='openai/clip-vit-base-patch16', output='vision_model.last_hidden_state',
                            image_size=[224, 224], preprocessing='bilinear', antialias=True, dataset_json_sha256=DATA_SHA)
            if any(meta.get(k) != v for k, v in required.items()):
                raise ValueError('Incompatible H5 extraction profile')
            for k, values in [('mean', [.48145466, .4578275, .40821073]), ('std', [.26862954, .26130258, .27577711])]:
                np.testing.assert_allclose(meta[k], values, atol=1e-8, rtol=0)
            profile = {k: meta.get(k) for k in (*required, 'mean', 'std', 'model_revision', 'compute_dtype', 'storage_dtype', 'autocast')}
            if profiles and profiles[0] != profile:
                raise ValueError('Mixed CLIP cache profiles')
            profiles.append(profile)
            ids, names, features = f['imgids'], f['filename'], f['features']
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or features.shape != (len(ids), 197, 768) or len(names) != len(ids):
                raise ValueError('H5 tables/layout differ')
            if features.dtype.kind != 'f' or features.dtype.itemsize not in (2, 4):
                raise ValueError('Expected FP16/FP32 raw CLIP values')
            for row, value in enumerate(ids[:]):
                identity = int(value)
                if identity in index:
                    raise ValueError('Duplicate visual ID')
                index[identity] = (path, row)
    for image in selected:
        identity = coco_id(image['filename'])
        if identity != int(image['cocoid']) or identity not in index:
            raise ValueError('Selected COCO image missing/mismatched')
        path, row = index[identity]
        with h5py.File(path, 'r') as f:
            if text(f['filename'][row]) != image['filename']:
                raise ValueError('Selected feature filename mismatch')
    if not profiles:
        raise ValueError('No visual shards')
    return index, profiles[0]


def make_regions(dataset, detections, config, out, words):
    import h5py
    from PIL import Image
    train, val = select_pilot(dataset)
    selected = train + val
    index, profile = visual_index(config['visual_cache'], selected)
    rows, features, handles = [], [], {}
    clipped_boxes = fallback_images = 0
    try:
        for number, image in enumerate(selected, 1):
            name = image['filename']; identity = coco_id(name)
            directory = image.get('filepath', 'train2014' if 'train2014' in name else 'val2014')
            root = Path(config['image_root'])
            candidates = [root/directory/name, root/name]
            existing = [p for p in candidates if p.is_file()]
            if not existing:
                raise FileNotFoundError('RGB missing: '+name)
            jpeg = existing[0].read_bytes()
            with Image.open(io.BytesIO(jpeg)) as rgb:
                width, height = rgb.size
            objects = detections[name]
            clipped = sum(any(v < 0 for v in obj['bbox'][:2]) or obj['bbox'][2] > width or obj['bbox'][3] > height for obj in objects)
            clipped_boxes += clipped
            path, row = index[identity]
            if path not in handles:
                handles[path] = h5py.File(path, 'r')
            raw = handles[path]['features'][row]
            if raw.shape != (197, 768) or not np.isfinite(raw).all():
                raise ValueError('Selected raw feature row is malformed/nonfinite')
            raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            patches = raw.astype(np.float32)[1:].reshape(14, 14, 768)
            pooled, boxes, kinds = roi_pool(patches, objects, width, height)
            chosen = sorted(enumerate(objects), key=lambda x: (-float(x[1]['conf']), x[0]))[:150]
            scores = np.asarray([float(obj['conf']) for _, obj in chosen] if chosen else [0.], dtype=np.float32)
            fallback_images += int(not objects)
            features.append((pooled, boxes, scores, identity))
            rows.append(dict(coco_id=identity, filename=name, source_split=image['split'],
                group='train' if number <= 256 else 'val', width=width, height=height,
                image_sha256=hashlib.sha256(jpeg).hexdigest(), raw_selected_row_sha256=raw_hash,
                raw_selected_row_dtype=str(raw.dtype), region_kinds=kinds, source_detections=len(objects),
                boxes_needing_boundary_clip=clipped,
                captions=[s['raw'] for s in image['sentences']],
                sequences=[encode(s['tokens'], words, maximum=54) for s in image['sentences']]))
            if number % 32 == 0:
                print('Actual RGB dimensions/raw CLIP ROI:', number, '/320', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    packed = pack_regions(features)
    np.savez_compressed(out/'regions.npz', **packed)
    save(out/'pilot_metadata.json', dict(selection='csa-paper-pilot-v1/42/ + filename SHA256', images=rows,
        train_images=256, val_images=64, test_images=0, feature_profile=profile,
        detector_coordinate_interpretation='pixel xyxy; producer provenance unknown',
        clipped_boxes=clipped_boxes, full_frame_fallback_images=fallback_images))
    return dict(images=320, train_images=256, val_images=64, test_images=0,
                total_regions=int(packed['offsets'][-1]), max_regions=max(np.diff(packed['offsets']).tolist()),
                clipped_boxes=clipped_boxes, fallback_images=fallback_images, feature_profile=profile)


def prepare(config, out, report):
    import spacy
    if spacy.__version__ != '3.8.11':
        raise ValueError('Pinned spaCy3.8.11 required')
    spacy.require_cpu()
    for key, expected in [('dataset_json', DATA_SHA), ('detections', DETECTION_SHA)]:
        if sha(config[key]) != expected:
            raise ValueError('Pinned source differs: '+key)
    dataset = json.loads(Path(config['dataset_json']).read_text(encoding='utf-8'))
    words = vocabulary(dataset)
    # Match inventory's exact JSON serialization before saving pretty metadata elsewhere.
    (out/'vocabulary.json').write_text(json.dumps(words, ensure_ascii=False)+'\n', encoding='utf-8')
    if sha(out/'vocabulary.json') != VOCAB_SHA:
        raise ValueError('TRAIN vocabulary differs from verified inventory')
    report['regions'] = make_regions(dataset, detection_index(json.loads(Path(config['detections']).read_text(encoding='utf-8'))), config, out, words)
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    if nlp.meta['version'] != '3.8.0' or set(nlp.pipe_names) != {'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'}:
        raise ValueError('POS model/components differ')
    report['pos_model_files_sha256'] = {str(p.relative_to(nlp.path)): sha(p) for p in sorted(Path(nlp.path).rglob('*')) if p.is_file()}
    report['pos_statistics'] = {}
    print('Starting full TRAIN POS,566747 captions; CPU only', flush=True)
    lists = make_word_lists(dataset, nlp, report['pos_statistics'])
    if report['pos_statistics']['caption_documents'] != 566747:
        raise ValueError('TRAIN POS caption coverage differs')
    report['bank_word_counts'] = {k: len(v) for k, v in lists.items()}
    save(out/'bank_words.json', dict(source='TRAIN only, train+restval', pos_library='spacy3.8.11',
        model='en_core_web_sm3.8.0', model_wheel_sha256=MODEL_WHEEL_SHA, minimum_count=5,
        noun=[dict(word=w, count=n) for w, n in lists['noun']],
        verb=[dict(word=w, count=n) for w, n in lists['verb']],
        noun_template='a photo of a {lemma}', verb_template='a photo contains the action of {lemma}'))
    report['versions'] = {name: importlib.metadata.version(name) for name in ('spacy', 'en-core-web-sm', 'thinc', 'numpy', 'h5py', 'Pillow')}
    report['versions']['python'] = platform.python_version()
    report['status'] = 'CPU_PREPARATION_COMPLETE'
    save(out/'receipt.json', dict(status=report['status'], schema='csa_paper_words_roi_v1',
        inputs=dict(dataset_json=DATA_SHA, detections=DETECTION_SHA, vocabulary=VOCAB_SHA),
        payload_sha256={name: sha(out/name) for name in ('vocabulary.json', 'bank_words.json', 'pilot_metadata.json', 'regions.npz')},
        semantic_embeddings_created=False, training_ready=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    report = dict(status='INCOMPLETE', caption_model_executed=False, gpu_used=False,
                  training_ready=False, semantic_embeddings_created=False, optimizer_updates=0,
                  pos_model_execution='User-run CPU spaCy, not agent local', full_h5_payload_hashes_checked=False)
    start = time.monotonic()
    try:
        prepare(config, out, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['seconds'] = time.monotonic()-start
        save(out/'summary.json', report)


if __name__ == '__main__':
    main()
