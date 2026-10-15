"""Extract frozen CPU poses from the existing61-image annotation package."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from captioning.pose_evidence import subject_key, match_subject


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import mediapipe as mp
    package, output = Path(args.package), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path('audit/interaction_region_train_pairs.json').read_text())
    mapping = json.loads((package/'private_mapping.json').read_text())
    if sha(package/'cases.json') != mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json') != mapping['source_pairs_sha256']:
        raise ValueError('Pose extraction source/manifest changed.')
    images = json.loads((package/'image_provenance.json').read_text())
    lookup = {x['filename']: x for x in images}
    subjects = {}
    for row in rows:
        if row['filename'] in lookup:
            key = subject_key(row['filename'], row['subject_box'])
            subjects[key] = {'filename': row['filename'], 'subject_box': row['subject_box']}
    protocol = {'variant': 'subject_pose_landmarker_lite_v1', 'mediapipe': mp.__version__,
        'model_sha256': sha(args.model), 'model_url': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        'source_pairs_sha256': mapping['source_pairs_sha256'], 'manifest_sha256': mapping['manifest_sha256'],
        'image_provenance_sha256': sha(package/'image_provenance.json'), 'subjects': len(subjects),
        'num_poses': 4, 'padding_fraction': .15, 'visibility_presence_minimum': .5,
        'minimum_visible_torso': 3, 'minimum_match_iou': .15, 'minimum_match_score_gap': .1,
        'notes': 'CPU image mode; padded subject crop; no temporal tracking, GT keypoints, human relation labels or action-label inference.'}
    path = output/'extraction_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Changed extraction protocol; use a new output.')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    options = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=4,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, output_segmentation_masks=False)
    result = {}; verified_images = set()
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        for key, row in sorted(subjects.items()):
            image_path = package/'images'/row['filename']
            if row['filename'] not in verified_images:
                if sha(image_path) != lookup[row['filename']]['sha256']:
                    raise ValueError('Local annotation image changed: '+row['filename'])
                verified_images.add(row['filename'])
            with Image.open(image_path) as source:
                image = source.convert('RGB'); w, h = image.size
                if list(image.size) != lookup[row['filename']]['image_size']:
                    raise ValueError('Image dimensions changed.')
                a, b, c, d = row['subject_box']; pw, ph = .15*(c-a), .15*(d-b)
                crop = [max(0, math.floor((a-pw)*w)), max(0, math.floor((b-ph)*h)),
                        min(w, math.ceil((c+pw)*w)), min(h, math.ceil((d+ph)*h))]
                cropped = image.crop(crop)
                if max(cropped.size) > 512:
                    cropped.thumbnail((512, 512), Image.Resampling.BICUBIC)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cropped).copy())
                prediction = model.detect(mp_image)
            poses = [[{'x': (p.x*(crop[2]-crop[0])+crop[0])/w, 'y': (p.y*(crop[3]-crop[1])+crop[1])/h,
                       'visibility': float(p.visibility), 'presence': float(p.presence)} for p in pose]
                     for pose in prediction.pose_landmarks]
            matched = match_subject(poses, row['subject_box'])
            result[key] = {**row, 'image_size': [w, h], 'crop_pixels': crop, 'detected_poses': len(poses), **matched}
            if len(result) % 5 == 0 or len(result) == len(subjects):
                print(f'Pose subjects {len(result)}/{len(subjects)}: {dict(Counter(x["status"] for x in result.values()))}', flush=True)
    (output/'private_subject_poses.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    summary = {'variant': protocol['variant'], 'protocol_sha256': sha(path), 'subjects': len(subjects),
        'images': len(verified_images), 'statuses': dict(Counter(x['status'] for x in result.values())),
        'pose_cache_sha256': sha(output/'private_subject_poses.json'), 'relation_labels_generated': 0,
        'dependencies': {name: importlib.metadata.version(name) for name in ('mediapipe', 'numpy', 'opencv-python-headless', 'matplotlib')}}
    (output/'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('package', 'model', 'output'): parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
