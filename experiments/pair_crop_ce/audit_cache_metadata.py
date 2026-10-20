"""Audit the user's metadata export only; never claim to have read source tensors."""
import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def audit(report):
    assert report['status']=='METADATA_ONLY_NO_MODEL_NO_TRAINING'
    shards=report['shards'];assert len(shards)==3
    assert report['overlapping_ids']==[] and report['unique_ids']==123287
    assert report['feature_values_read'] is False
    total=0;parts=set();receipts=[]
    for s in shards:
        a=s['attributes'];m=s['metadata'];d=s['datasets'];ids=s['id_summary']
        assert a['complete'] is True and a['id_key']=='coco_id'
        assert json.loads(a['metadata_json'])==m
        n=m['image_count'];total+=n;parts.add(m['part'])
        assert m['parts']==3 and not m['smoke'] and m.get('start_row',0)==0
        assert a['rows_written']==n==ids['count'] and ids['key']=='imgids'
        assert d['features']['shape']==[n,197,768] and d['features']['dtype']=='float16'
        assert all(d[k]['shape']==[n] for k in ('filename','filepath','split','imgids','eval_ids'))
        assert m['model_id']=='openai/clip-vit-base-patch16' and m['output']=='vision_model.last_hidden_state'
        assert m['dataset_json_sha256']==DATA_SHA
        assert m['preprocessing']=='bilinear' and m['image_size']==[224,224] and m['antialias'] is True
        assert m['mean']==[.48145466,.4578275,.40821073] and m['std']==[.26862954,.26130258,.27577711]
        assert m['compute_dtype']=='float32' and m['storage_dtype']=='float16' and m['autocast'] is False
        receipts.append(dict(filename=s['filename'],image_count=n,model_revision=m['model_revision'],
            original_runtime={k:m[k] for k in ('torch','torchvision','transformers')},id_array_sha256=ids['sha256']))
    assert parts=={1,2,3} and total==report['unique_ids']
    return dict(status='REPORTED_HEADER_SCHEMA_PASS_REVISION_UNKNOWN' if any(r['model_revision'] is None for r in receipts) else 'REPORTED_HEADER_SCHEMA_PASS',
        shards=receipts,reported_unique_ids=total,reported_duplicate_ids=False,dataset_json_sha256=DATA_SHA,
        preprocessing_metadata_matches_protocol=True,encoder_revision_verified=False,feature_compatibility_verified=False,training_approved=False,
        limits=['Exported headers/ID summaries validated; actual HDF5 files and feature values were not independently read locally',
                'Bilinear224 metadata plus source code supports planned transform, not proof of every historical preprocessing operation',
                'Null revision is unknown provenance, not evidence of corrupt tensors or proof of a model bug',
                'Original cache records transformers5.0.0; library/attention/runtime differences must be logged and numeric compatibility checked'],
        next='Pinned re-encoding of fullframe and crop; compare fullframe with corresponding old rows before any training; same recomputed fullframe G for both arms')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.input.read_bytes();result=audit(json.loads(raw));result['source_export_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(result['status'])
