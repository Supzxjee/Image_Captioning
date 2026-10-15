"""Label-blind full-frame fallback for missing ROI poses; keep accepted ROI poses."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from captioning.pose_evidence import match_subject


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import mediapipe as mp
    source, package, output = map(Path, (args.source, args.package, args.output))
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source/'private_subject_poses.json').read_text())
    summary = json.loads((source/'extraction_summary.json').read_text())
    original_protocol = json.loads((source/'extraction_protocol.json').read_text())
    if (sha(source/'private_subject_poses.json') != summary['pose_cache_sha256'] or
            sha(source/'extraction_protocol.json') != summary['protocol_sha256'] or
            sha(args.model) != original_protocol['model_sha256']):
        raise ValueError('Original pose cache/model changed.')
    provenance = {r['filename']: r for r in json.loads((package/'image_provenance.json').read_text())}
    if sha(package/'image_provenance.json') != original_protocol['image_provenance_sha256']:
        raise ValueError('Original images changed.')
    protocol = {**original_protocol, 'variant': 'subject_pose_context_fallback_v2',
        'original_roi_protocol_sha256': sha(source/'extraction_protocol.json'),
        'original_roi_cache_sha256': summary['pose_cache_sha256'], 'fallback_num_poses': 8,
        'fallback_rule': 'Full image only for no_pose/no_confident_subject. Preserve accepted ROI poses and ambiguous cases. '
                         'Same confidence/matching thresholds; no relation labels. Conflicting fallback pose reuse is ambiguous.'}
    path = output/'extraction_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Changed fallback protocol.')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    result = {k: {**v, 'pose_origin': 'roi' if v['status'] == 'matched' else 'unavailable'} for k, v in original.items()}
    groups = defaultdict(list)
    for key, row in original.items():
        if row['status'] in ('no_pose', 'no_confident_subject'):
            groups[row['filename']].append(key)
    options = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=8, min_pose_detection_confidence=.5,
        min_pose_presence_confidence=.5, output_segmentation_masks=False)
    promoted = 0
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        for filename, keys in sorted(groups.items()):
            image_path = package/'images'/filename
            if sha(image_path) != provenance[filename]['sha256']: raise ValueError('Annotation image changed.')
            with Image.open(image_path) as source_image:
                image = source_image.convert('RGB'); w, h = image.size
                prediction = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image).copy()))
            poses = [[{'x': float(p.x), 'y': float(p.y), 'visibility': float(p.visibility), 'presence': float(p.presence)}
                      for p in pose] for pose in prediction.pose_landmarks]
            matched = {key: match_subject(poses, original[key]['subject_box']) for key in keys}
            used = Counter(r['candidates'][0]['pose_index'] for r in matched.values() if r['status'] == 'matched')
            for key, found in matched.items():
                if found['status'] == 'matched' and used[found['candidates'][0]['pose_index']] > 1:
                    found = {'status': 'ambiguous', 'landmarks': None, 'candidates': found['candidates']}
                if found['status'] == 'matched':
                    result[key] = {**original[key], **found, 'pose_origin': 'full_frame_fallback',
                                   'detected_poses': len(poses), 'image_size': [w,h],
                                   'original_roi_status': original[key]['status'], 'crop_pixels': [0,0,w,h]}
                    promoted += 1
                else:
                    result[key]['fallback_status'] = found['status']
                    result[key]['fallback_candidates'] = found['candidates']
            print(f'Context pose fallback {filename}: total recovered {promoted}', flush=True)
    for key, row in original.items():
        if row['status'] == 'matched':
            if result[key]['landmarks'] != row['landmarks']: raise ValueError('Accepted ROI pose was changed.')
    cache = output/'private_subject_poses.json'; cache.write_text(json.dumps(result, indent=2), encoding='utf-8')
    final = {'variant': protocol['variant'], 'protocol_sha256': sha(path), 'pose_cache_sha256': sha(cache),
        'subjects': len(result), 'images': summary['images'], 'statuses': dict(Counter(r['status'] for r in result.values())),
        'context_images_processed': len(groups), 'newly_matched_subjects': promoted, 'accepted_roi_poses_unchanged': True,
        'relation_labels_generated': 0, 'dependencies': summary['dependencies']}
    (output/'extraction_summary.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'package', 'model', 'output'): parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
