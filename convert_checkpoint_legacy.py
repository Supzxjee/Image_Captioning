"""Convert an identified local PyTorch checkpoint to non-ZIP serialization."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import torch


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compare(left,right,path='checkpoint'):
    if type(left) is not type(right):raise ValueError(f'Type changed at {path}')
    if isinstance(left,torch.Tensor):
        if left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):
            raise ValueError(f'Tensor changed at {path}')
        return 1
    if isinstance(left,dict):
        if list(left)!=list(right):raise ValueError(f'Keys/order changed at {path}')
        return sum(compare(left[k],right[k],path+'.'+str(k)) for k in left)
    if isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError(f'Length changed at {path}')
        return sum(compare(a,b,path+f'[{i}]') for i,(a,b) in enumerate(zip(left,right)))
    if left!=right:raise ValueError(f'Metadata changed at {path}')
    return 0


def convert(source,output,expected_sha):
    source,output=Path(source),Path(output)
    manifest=output.with_suffix('.manifest.json')
    temporary=output.with_suffix(output.suffix+'.partial')
    if any(p.exists() for p in (output,manifest,temporary)):
        raise FileExistsError('Output/manifest/temporary path already exists; choose a fresh name.')
    original_sha=digest(source)
    if original_sha!=expected_sha:raise ValueError('Source SHA256 mismatch; conversion stopped.')
    payload=torch.load(source,map_location='cpu',weights_only=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,temporary,_use_new_zipfile_serialization=False)
    if zipfile.is_zipfile(temporary):raise ValueError('Unexpected ZIP output.')
    restored=torch.load(temporary,map_location='cpu',weights_only=True)
    tensors=compare(payload,restored)
    temporary.replace(output)
    record={'status':'CONVERTED_AND_RELOADED_VERIFIED','source_file':source.name,'source_sha256':original_sha,
            'source_bytes':source.stat().st_size,'output_file':output.name,'output_sha256':digest(output),
            'output_bytes':output.stat().st_size,'output_is_zip':False,'serialization':'torch.save legacy non-ZIP',
            'load_check':'torch.load map_location=cpu weights_only=True','tensor_entries_exactly_equal':tensors,
            'all_metadata_and_vocab_preserved':True,'no_training_or_model_inference':True,
            'selected_metadata':{k:restored.get(k) for k in ('optimization','visual_adapter','num_visual_queries',
                'qformer_layers','scst_weight','ce_weight','object_reward_weight','object_reward_parser','object_reward_penalty')},
            'note':'File hash changes with serialization; tensor and metadata equality verified. Original source hash remains provenance.'}
    manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(record,indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();convert(a.source,a.output,a.expected_sha256)
