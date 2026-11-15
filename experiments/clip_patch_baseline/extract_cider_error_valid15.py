from pathlib import Path
import hashlib
import json

root = Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')
paths = list(root.rglob('prompt_links.jsonl'))
if len(paths) != 1:
    raise ValueError(f'Attach csa-prompt-full-karpathy-v175. Found {len(paths)} prompt link files.')
wanted = {331475, 379156, 72024, 241888, 239894, 436349, 337439, 136915,
          29358, 139749, 487438, 422464, 389308, 468858, 82174}
digest = hashlib.sha256()
rows = {}
with paths[0].open('rb') as stream:
    for line in stream:
        digest.update(line)
        row = json.loads(line)
        image_id = row['coco_id']
        if image_id in wanted and row['group'] == 'val':
            if image_id in rows:
                raise ValueError(f'Duplicate VALID image {image_id}')
            rows[image_id] = row
expected = 'e90c79b3b04540630b4bd2d279a4070ef05da7f4b65970c952851c37e2423bf0'
if digest.hexdigest() != expected or set(rows) != wanted:
    raise ValueError(f'Unexpected v175 hash or incomplete VALID coverage. Found {len(rows)}/15.')
output = Path('/kaggle/working/cider_error_valid15_prompt_links.json')
output.write_text(json.dumps({'source_sha256': expected, 'rows': list(rows.values())},
                             ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(output)
