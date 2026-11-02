"""Audit cached review metadata only; never read feature shards or run a model."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent

def audit():
    cache=HERE/'full_roi_review_text';read=lambda n:json.loads((cache/n).read_text(encoding='utf-8'))
    source=read('source_receipt.json');manifest=read('manifest.json');receipt=read('receipt.json');inputs=read('input_receipt.json')
    checks=0
    def check(condition,message):
        nonlocal checks
        if not condition:raise AssertionError(message)
        checks+=1
    for name,h in source['sha256'].items():
        blob=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=HERE.parents[1])
        check(hashlib.sha256(blob).hexdigest()==h,'Git source '+name)
    for name,key in [('manifest.json','manifest_sha256'),('input_receipt.json','input_receipt_sha256')]:
        check(hashlib.sha256((cache/name).read_bytes()).hexdigest()==receipt[key],name+' bytes')
    check((cache/'progress.json').read_bytes()==(cache/'manifest.json').read_bytes(),'progress/manifest')
    check(inputs['source']==source and inputs['feature_profile']==manifest['feature_profile'],'input/profile')
    check(inputs['dataset_json_sha256']=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d','dataset identity')
    check(inputs['detections_sha256']=='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c','detector identity')
    check(inputs['vocabulary_sha256']==receipt['vocabulary_sha256']==manifest['vocabulary_sha256']=='924be210aa0daa8b0968a6d69ad2c66b3de603f49c35f4aa67247c75d6173fc7','vocabulary')
    check(not inputs['full_h5_payload_hashed'] and not inputs['bank_reencoded'] and not inputs['test_selected'],'scope')
    check(read('run_status.json')==dict(complete=True,training_executed=False),'run status')
    check(receipt['status']==manifest['status']=='FULL_ROI_PREPARATION_COMPLETE','completion')
    check((manifest['images'],manifest['train_images'],manifest['val_images'],manifest['test_images'])==(118287,113287,5000,0),'image totals')
    shards=manifest['shards'];check(len(shards)==119,'119shards')
    check(Counter(e['group'] for e in shards)==dict(train=114,val=5),'group counts')
    check(sum(e['regions'] for e in shards)==manifest['regions'],'region sum')
    names=[]
    for group in ('train','val'):
        entries=[e for e in shards if e['group']==group]
        check(sum(e['images'] for e in entries)==manifest[group+'_images'],'group image sum')
        for i,e in enumerate(entries):
            prefix=f'{group}_{i:04d}'
            check(set(e['files'])=={prefix+'.npz',prefix+'.json'},'shard filenames')
            check(e['images']==(287 if group=='train' and i==113 else 1000),'tail count')
            check(e['images']<=e['regions']<=e['images']*manifest['max_regions'],'region range')
            check(e['first_id']!=e['last_id'] and isinstance(e['first_id'],int) and isinstance(e['last_id'],int),'endpoint IDs')
            for h in list(e['files'].values())+[e['id_order_sha256']]:check(bool(re.fullmatch('[0-9a-f]{64}',h)),'SHA format')
            names.extend(e['files'])
    check(len(set(names))==238,'unique payload names')
    check(manifest['clipped_boxes']==0 and manifest['max_regions']==87 and manifest['fallback_images']==1004 and manifest['regions']==833473,'region summary')
    check(0<manifest['worker_seconds']<35400 and manifest['payload_bytes']==1749398178,'runtime/size')
    throughput=read('throughput.json')
    check(math.isclose(throughput['projected_remaining_seconds'],throughput['processing_seconds']/1000*117287,rel_tol=1e-12),'throughput projection')
    log=(cache/'notebook.log').read_text(encoding='utf-8')
    check(log.count('SHARD COMPLETE ')==119 and 'FULL ROI COMPLETE.' in log,'log completion')
    check(not (cache/'failure.txt').exists(),'no failure artifact')
    result=dict(status='REVIEWED_FULL_ROI_PREPARATION_COMPLETE',zip_sha256='a7749fde0ee8ee2ccfb78661c8121d892179dcc81d842864f969b1569bb0ea41',
        source_commit=source['commit'],metadata_assertions=checks,git_blob_hashes_verified=7,
        train_images=113287,val_images=5000,test_images=0,total_images=118287,regions=833473,shards=119,shard_payload_files=238,
        caption_events=manifest['caption_events'],fallback_images=1004,max_regions=87,clipped_boxes=0,
        worker_seconds=manifest['worker_seconds'],payload_bytes=manifest['payload_bytes'],manifest_sha256=receipt['manifest_sha256'],
        feature_revision_unknown=manifest['feature_profile']['model_revision'] is None,
        shard_binaries_and_per_image_metadata_in_review=False,all_image_ids_independently_recomputed=False,
        shard_sha_checks_pending_transport=True,local_feature_or_model_execution=False,
        next_step='Save exact output as Dataset with all238shard fullSHA checks; obtain version. Decode131 remains input-blocked; full training not yet scheduled.')
    (HERE/'FULL_ROI_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
