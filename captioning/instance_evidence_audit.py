"""Development diagnostics, not new calibrated inference rules."""
from captioning.selected_relation_prompts import entity_id


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    return overlap / max(area(a)+area(b)-overlap, 1e-12)


def ambiguous_regions(inventory, minimum_iou=.9):
    if not 0 < minimum_iou <= 1:
        raise ValueError('Invalid IoU threshold.')
    result = []
    for filename, objects in inventory.items():
        for i, a in enumerate(objects):
            for b in objects[i+1:]:
                if a['label'] == b['label']:
                    continue
                iou = box_iou(a['bbox'], b['bbox'])
                if iou >= minimum_iou:
                    result.append({'filename': filename, 'iou': iou,
                        'a': entity_id(filename, a['label'], a['bbox']), 'a_label': a['label'],
                        'b': entity_id(filename, b['label'], b['bbox']), 'b_label': b['label']})
    return result


def subject_swap_indices(rows):
    """Same exact object and image, other actor. No semantic negative label implied."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['filename'], row['object_id']), []).append(i)
    return [(i, j) for indexes in groups.values() for i in indexes for j in indexes
            if rows[i]['subject_id'] != rows[j]['subject_id']]
