"""Byte/SHA and per-image metadata checks only. No neural or NPZ imports."""
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def check_shard_metadata(root,manifest,vocab_size):
    root=Path(root);seen=set();groups={'train':0,'val':0};regions=captions=fallback=clipped=maximum=0
    for entry in manifest['shards']:
        group=entry['group']
        if group not in groups:raise ValueError('Unexpected shard group')
        metadata_names=[n for n in entry['files'] if n.endswith('.json')]
        if len(metadata_names)!=1:raise ValueError('Expected one metadataJSON/shard')
        data=read(root/metadata_names[0]);images=data['images'];ids=[m['coco_id'] for m in images]
        if data['schema']!='csa_full_roi_shard_v1' or data['group']!=group or len(images)!=entry['images']:raise ValueError('Shard metadata count/group differs')
        if not ids or ids[0]!=entry['first_id'] or ids[-1]!=entry['last_id'] or hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=entry['id_order_sha256']:
            raise ValueError('Shard ID order differs')
        shard_regions=0
        for m in images:
            identity=m['coco_id']
            if type(identity)!=int or identity in seen or m['group']!=group:raise ValueError('Duplicate/invalid COCO ID or group')
            seen.add(identity)
            if m['source_split'] not in (('train','restval') if group=='train' else ('val',)):raise ValueError('Split leakage')
            n=len(m['region_kinds'])
            if not 1<=n<=150 or m['width']<=0 or m['height']<=0 or any(k not in ('predicted_roi','full_frame_fallback') for k in m['region_kinds']):raise ValueError('Invalid ROI metadata')
            if m['source_detections']==0 and m['region_kinds']!=['full_frame_fallback']:raise ValueError('Invalid fallback metadata')
            if len(m['captions'])!=len(m['sequences']) or not m['sequences']:raise ValueError('Caption/sequence count differs')
            for seq in m['sequences']:
                if len(seq)!=54 or seq[0]!=1 or seq.count(2)!=1 or any(type(t)!=int or not 0<=t<vocab_size for t in seq):raise ValueError('Invalid caption IDs')
                eos=seq.index(2)
                if 0 in seq[:eos] or any(seq[eos+1:]):raise ValueError('Invalid EOS/PAD placement')
            groups[group]+=1;shard_regions+=n;captions+=len(m['sequences']);maximum=max(maximum,n)
            fallback+=int(m['source_detections']==0);clipped+=m['boxes_needing_boundary_clip']
        if shard_regions!=entry['regions']:raise ValueError('Shard region counts differ')
        regions+=shard_regions
    if groups!={'train':manifest['train_images'],'val':manifest['val_images']} or len(seen)!=manifest['images'] or regions!=manifest['regions'] or captions!=manifest['caption_events']:
        raise ValueError('Full metadata counts differ')
    if (fallback,clipped,maximum)!=(manifest['fallback_images'],manifest['clipped_boxes'],manifest['max_regions']):raise ValueError('ROI summary differs')
    return dict(unique_images=len(seen),groups=groups,regions=regions,caption_events=captions,test_images=0)

def package(source,destination,inventory):
    source,destination=Path(source),Path(destination)
    if inventory['kernel_version']!='ducanh2403/2-vi-cai-kia-qua-nhieu/134':raise ValueError('Expected exact Output134')
    if destination.exists():raise ValueError('Preserve existing destination; use a fresh path')
    for number,(name,h) in enumerate(inventory['files'].items(),1):
        if Path(name).name!=name or not (source/name).is_file() or sha(source/name)!=h:raise ValueError('Missing/changed output: '+name)
        if number%25==0:print('Verified output SHA',number,'/',len(inventory['files']),flush=True)
    manifest=read(source/'manifest.json');receipt=read(source/'receipt.json');vocab=read(source/'vocabulary.json')
    if manifest['status']!='FULL_ROI_PREPARATION_COMPLETE' or receipt['status']!='FULL_ROI_PREPARATION_COMPLETE':raise ValueError('Incomplete source')
    if (manifest['images'],manifest['train_images'],manifest['val_images'],manifest['test_images'],len(manifest['shards']))!=(118287,113287,5000,0,119):raise ValueError('Full coverage differs')
    if receipt['manifest_sha256']!=inventory['files']['manifest.json'] or receipt['input_receipt_sha256']!=inventory['files']['input_receipt.json'] or receipt['vocabulary_sha256']!=inventory['files']['vocabulary.json']:raise ValueError('Global receipt SHA differs')
    for e in manifest['shards']:
        for n,h in e['files'].items():
            if inventory['files'].get(n)!=h:raise ValueError('Shard manifest SHA differs')
    if len(vocab)!=10000 or vocab[:4]!=['<pad>','<bos>','<eos>','<unk>']:raise ValueError('Vocabulary differs')
    metadata=check_shard_metadata(source,manifest,len(vocab))
    destination.mkdir(parents=True)
    for n,h in inventory['files'].items():
        shutil.copyfile(source/n,destination/n)
        if sha(destination/n)!=h:raise ValueError('Copied bytes differ: '+n)
    report=dict(status='FULL_ROI_V134_TRANSPORT_COMPLETE',source_kernel_version=inventory['kernel_version'],
        source_review_zip_sha256=inventory['source_review_zip_sha256'],files_sha256=inventory['files'],
        original_bytes_preserved=True,actual_per_image_metadata_verified=metadata,npz_arrays_loaded=False,model_executed=False,
        also_requires_dataset='ducanh2403/csa-semantic-banks-v130')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
