"""Verify feature coverage, pooling geometry and masks; does not train a head."""
import argparse,json
from collections import Counter
from pathlib import Path
import torch
from captioning.interaction_regions import boxes_in_crop,region_weights,REGIONS
from evaluate_frozen_residual import sha,newline_hashes


def validate(output):
    root=Path(output);protocol_path=Path('audit/interaction_region_protocol.json')
    protocol=json.loads(protocol_path.read_text());manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['configuration']
    if config['training'] is not False or config['model_revision']!=protocol['revision'] or config['protocol_sha256'] not in newline_hashes(protocol_path):
        raise ValueError('Manifest/model/protocol changed.')
    rows=[]
    for split in ('train','val','transfer','full100'):
        path=root/'data'/f'{split}_pairs.json';pairs=json.loads(path.read_text())
        if protocol['splits'][split]['pair_json_sha256'] not in newline_hashes(path) or len(pairs)!=protocol['splits'][split]['pairs']:
            raise ValueError('Output pair metadata changed.')
        rows.extend((split,i,row) for i,row in enumerate(pairs))
    training_ids={r['image_id'] for split,_,r in rows if split=='train'}
    for split in ('val','transfer','full100'):
        if training_ids&{r['image_id'] for s,_,r in rows if s==split}:raise ValueError('Train/evaluation overlap.')
    records=[];file_hashes={}
    for shard in range(2):
        path=root/f'regions_{shard}.pt';part=torch.load(path,map_location='cpu',weights_only=True)
        if any(r['index']%2!=shard for r in part):raise ValueError('Shard assignment changed.')
        records.extend(part);file_hashes[path.name]=sha(path)
    records.sort(key=lambda r:r['index'])
    if [r['index'] for r in records]!=list(range(len(rows))):raise ValueError('Missing/duplicate feature coverage.')
    empty=Counter()
    for record,(split,index,row) in zip(records,rows):
        if (record['split'],record['split_index'])!=(split,index):raise ValueError('Split/index ordering changed.')
        f,v,m=record['regions'],record['valid'],record['mass']
        if f.shape!=(6,768) or v.shape!=(6,) or m.shape!=(6,) or v.dtype!=torch.bool:raise ValueError('Feature/mask shape changed.')
        if not torch.isfinite(f).all() or not torch.isfinite(m).all() or not ((m>=0)&(m<=1+1e-6)).all():raise ValueError('Invalid feature values.')
        if not torch.allclose(f[v].norm(dim=-1),torch.ones(int(v.sum())),atol=1e-3) or f[~v].abs().sum()!=0:
            raise ValueError('Valid regions must normalize; empty regions must be zero.')
        width,height=record['image_size']
        boxes=boxes_in_crop(row['subject_box'],row['object_box'],record['crop_xyxy_pixels'],width,height)
        if not torch.allclose(boxes,record['boxes_in_crop'],atol=1e-6):raise ValueError('ROI/crop geometry changed.')
        weights=region_weights(boxes,protocol['grid']);expected_mass=weights.sum(-1)
        if not torch.equal(v,torch.cat([expected_mass>1e-8,torch.ones(1,dtype=torch.bool)])):
            raise ValueError('Empty-region flags disagree with geometric support.')
        if not torch.allclose(m,torch.cat([expected_mass/196,torch.ones(1)]),atol=1e-6):raise ValueError('Region support changed.')
        for name,valid in zip(REGIONS,v):
            if not valid:empty[(split,name)]+=1
    result={'complete':True,'training_performed':False,'pairs':len(rows),'split_counts':dict(Counter(s for s,_,_ in rows)),
            'feature_files_sha256':file_hashes,'empty_regions':{f'{s}/{name}':n for (s,name),n in empty.items()},
            'train_evaluation_overlap':0,'feature_shape':[6,768],'region_order':REGIONS,
            'notes':'Coverage/geometry/finite/normalization checks only; not relation accuracy or contact segmentation.'}
    (root/'extraction_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);validate(parser.parse_args().output)
