"""Conservative pose/subject matching and target-local anatomical evidence.

Coordinates support a learned relation score; they do not certify contact or
action truth. Missing/ambiguous pose always has an explicit unavailable mask.
"""
import hashlib
import json
import math

TORSO = (11, 12, 23, 24)
HIPS, KNEES, ANKLES = (23, 24), (25, 26), (27, 28)
FEATURE_NAMES = ('hip_dx', 'hip_dy', 'hip_distance', 'knee_distance', 'ankle_distance',
                 'hip_inside_fraction', 'knee_angle', 'hip_visibility', 'knee_visibility',
                 'ankle_visibility', 'angle_visibility', 'subject_match_iou')


def subject_key(filename, box):
    return hashlib.sha256(json.dumps([filename, box], separators=(',', ':')).encode()).hexdigest()[:20]


def visible(point, threshold=.5):
    values = [point.get(k) for k in ('x', 'y', 'visibility', 'presence')]
    return (all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
            and point['visibility'] >= threshold and point['presence'] >= threshold)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / area if area else 0.


def match_subject(poses, box, threshold=.5):
    candidates = []
    width, height = box[2]-box[0], box[3]-box[1]
    if width <= 0 or height <= 0:
        raise ValueError('Empty subject box.')
    for index, landmarks in enumerate(poses):
        if len(landmarks) != 33:
            raise ValueError('Expected33 MediaPipe landmarks.')
        torso = [landmarks[k] for k in TORSO if visible(landmarks[k], threshold)]
        hips = [landmarks[k] for k in HIPS if visible(landmarks[k], threshold)]
        if len(torso) < 3 or not hips:
            continue
        inside = lambda p: box[0]-.05*width <= p['x'] <= box[2]+.05*width and box[1]-.05*height <= p['y'] <= box[3]+.05*height
        fraction = sum(inside(p) for p in torso) / len(torso)
        if fraction < .75 or not all(inside(p) for p in hips):
            continue
        points = [p for p in landmarks if visible(p, threshold)]
        pose_box = [min(p['x'] for p in points), min(p['y'] for p in points),
                    max(p['x'] for p in points), max(p['y'] for p in points)]
        overlap = iou(pose_box, box)
        if overlap < .15:
            continue
        candidates.append({'pose_index': index, 'iou': overlap, 'score': overlap + .25*fraction})
    candidates.sort(key=lambda x: -x['score'])
    if not candidates:
        return {'status': 'no_pose' if not poses else 'no_confident_subject', 'landmarks': None, 'candidates': candidates}
    if len(candidates) > 1 and candidates[0]['score'] - candidates[1]['score'] < .1:
        return {'status': 'ambiguous', 'landmarks': None, 'candidates': candidates}
    best = candidates[0]
    return {'status': 'matched', 'landmarks': poses[best['pose_index']], 'match_iou': best['iou'], 'candidates': candidates}


def pair_features(entry, subject, target, proxy=False):
    if entry['status'] != 'matched':
        return [0.] * len(FEATURE_NAMES), False
    points = [dict(p) for p in entry['landmarks']]
    width, height = subject[2]-subject[0], subject[3]-subject[1]
    if min(width, height, target[2]-target[0], target[3]-target[1]) <= 0:
        raise ValueError('Invalid subject/target geometry.')
    if proxy:
        # Matched ablation: same availability/confidences/matchIoU, only joint
        # positions are replaced by fixed detector-box anchors.
        for left, right, y in ((23, 24, .58), (25, 26, .78), (27, 28, .95)):
            for index, x in ((left, .4), (right, .6)):
                points[index]['x'], points[index]['y'] = subject[0]+x*width, subject[1]+y*height
    groups = [[points[k] for k in indices if visible(points[k])] for indices in (HIPS, KNEES, ANKLES)]
    hips, knees, ankles = groups
    if not hips:
        return [0.] * len(FEATURE_NAMES), False
    image_width, image_height = entry.get('image_size', [1, 1])
    aspect = image_height/image_width
    diagonal = math.hypot(width, height*aspect)
    cx, cy = sum(p['x'] for p in hips)/len(hips), sum(p['y'] for p in hips)/len(hips)
    clip = lambda x: max(-2., min(2., x))/2
    def distance(p):
        dx = max(target[0]-p['x'], 0., p['x']-target[2])
        dy = max(target[1]-p['y'], 0., p['y']-target[3])
        return min(2., math.hypot(dx, dy*aspect)/diagonal)/2
    distances = [min(distance(p) for p in group) if group else 0. for group in groups]
    inside = sum(target[0] <= p['x'] <= target[2] and target[1] <= p['y'] <= target[3] for p in hips)/len(hips)
    angles = []
    for ids in ((23, 25, 27), (24, 26, 28)):
        if not all(visible(points[k]) for k in ids):
            continue
        a, b, c = [points[k] for k in ids]
        u, v = (a['x']-b['x'], (a['y']-b['y'])*aspect), (c['x']-b['x'], (c['y']-b['y'])*aspect)
        norm = math.hypot(*u)*math.hypot(*v)
        if norm > 1e-10:
            angles.append(math.acos(max(-1., min(1., (u[0]*v[0]+u[1]*v[1])/norm)))/math.pi)
    return [clip(((target[0]+target[2])/2-cx)/diagonal),
            clip(((target[1]+target[3])/2-cy)*aspect/diagonal), *distances, inside,
            sum(angles)/len(angles) if angles else 0., len(hips)/2, len(knees)/2, len(ankles)/2,
            len(angles)/2, entry['match_iou']], True
