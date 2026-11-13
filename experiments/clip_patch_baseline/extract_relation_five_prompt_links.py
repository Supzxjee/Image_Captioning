from pathlib import Path
import hashlib, json
root = Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')
paths = list(root.rglob('prompt_links.jsonl'))
if len(paths) != 1:
    raise ValueError('Attach csa-prompt-full-karpathy-v175')
with paths[0].open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
if digest != 'e90c79b3b04540630b4bd2d279a4070ef05da7f4b65970c952851c37e2423bf0':
    raise ValueError('Unexpected prompt cache payload')
wanted = {195862, 217827, 546160, 199017, 147386}
rows = []
with paths[0].open(encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        if row['group'] == 'val' and row['coco_id'] in wanted:
            rows.append(row)
assert len(rows) == 5 and {row['coco_id'] for row in rows} == wanted
output = Path('/kaggle/working/csa_relation_five_prompt_links.json')
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(output)
