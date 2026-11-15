from copy import deepcopy
import re

def augmented(row, case):
    roi = case['roi']
    if len(row['objects']) >= 8 or any(obj['roi'] == roi['roi'] for obj in row['objects']):
        raise ValueError('Expected one unused detector region and one empty object slot')
    edited = deepcopy(row)
    edited['objects'].append(dict(instance_id='verified_yolo_' + str(case['coco_id']),
                                 roi=roi['roi'], text=case['text'],
                                 bbox=roi['bbox'], roi_bbox=roi['bbox']))
    return edited

def mentions(caption, label):
    suffix = r's?'
    return bool(re.search(r'\b' + re.escape(label) + suffix + r'\b', caption.lower()))

def verify_slot_change(before, after, slot):
    modified = {'object_text_id','object_box','object_weights','object_valid'}
    for name in before:
        if name in modified:
            np.testing.assert_array_equal(np.delete(before[name],slot,axis=0),
                                          np.delete(after[name],slot,axis=0))
        else:
            np.testing.assert_array_equal(before[name],after[name])
    if before['object_valid'][slot] or not after['object_valid'][slot]:
        raise ValueError('Expected exactly one newly active object slot')
