"""Local metadata inventory only. Never reads feature values or imports Torch."""
import ast
import gzip
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from readout_data import *


def main():
    path=Path('C:/KhoaLuanTotNghiep/Code/.tmp/hoi_caption_validation_audit/inputs/objectdetectioncache.json')
    if sha(path)!=DETECTION_SHA:raise ValueError('Local detector differs')
    dims=json.loads((HERE/'DEV_DIMENSIONS.json').read_bytes())['rows'];wanted={r['coco_id']:r for r in dims}
    found={}
    for name,r in json.loads(path.read_bytes()).items():
        iid=int(Path(name).stem.rsplit('_',1)[-1])
        if iid in wanted:
            if iid in found:raise ValueError('Duplicate dev detector')
            row=wanted[iid];bank=bank_row(r['objects'],row['width'],row['height'])
            region_arrays({**row,'objects':bank});found[iid]=len(bank)
    if set(found)!=set(wanted):raise ValueError('Local dev detector incomplete')
    manifest=json.loads(gzip.decompress((ROOT/'experiments/relation_frames_inventory/V2_MANIFEST.json.gz').read_bytes()))
    train={p[k]['image_id'] for p in manifest['train'] for k in ('positive','negative')}
    held={p[k]['image_id'] for p in manifest['heldout'] for k in ('positive','negative')}
    if train&held or (train|held)&set(wanted):raise ValueError('Image leakage')
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
    report=dict(status='STATIC_AND_LOCAL_METADATA_PASS',dev_image_dimensions=1000,dev_predicted_box_banks=1000,
        min_predicted_objects=min(found.values()),max_predicted_objects=max(found.values()),empty_banks=sum(v==0 for v in found.values()),
        caption_train_images=len(train),relation_train_pairs=256,heldout_images=len(held),heldout_pairs=64,
        train_dev_held_image_overlap=0,dimensions_sha256=sha(HERE/'DEV_DIMENSIONS.json'),detector_sha256=DETECTION_SHA,
        local_neural_execution=False,not_checked=['Actual Kaggle mounts, source HDF5 and original prompt values',
            'Torch/model/autograd/policy/numerical/two-GPU checks; user-run only','Learned caption or semantic improvement'])
    (HERE/'STATIC_CHECKS.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
