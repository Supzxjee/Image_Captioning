"""Audit cached benchmark receipts without importing torch or rerunning extraction."""
import hashlib
import json
from pathlib import Path
import zipfile
from tile_data import estimates,select_sample,ARCHIVE_SHA

HERE=Path(__file__).resolve().parent


def sha(value):return hashlib.sha256(value).hexdigest()


def audit(loaded):
    files=loaded['files'];objects={n:json.loads(v) for n,v in files.items() if n.endswith('.json')}
    cfg=objects['protocol.json'];prep=objects['preparation.json'];summary=objects['benchmark_summary.json']
    assert summary['status']=='COMPLETE' and summary['benchmark_only'] and not summary['trained'] and not summary['caption_evaluated']
    assert cfg['protocol']=='native_detail_cache_benchmark_v1' and cfg['benchmark_only']
    assert prep['status']=='COMPLETE' and prep['source_files_checked']==15000
    assert prep['checkpoint_sha256']=='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
    assert prep['dataset_sha256']=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    assert prep['groups_archive_sha256']==ARCHIVE_SHA and prep['sample_sha256']==sha(files['sample.json'].encode())
    for name,expected in cfg['implementation_sha256'].items():
        source=files['source/'+name];local=HERE/name if '/' not in name else HERE.parents[1]/name
        assert sha(source.encode())==expected,name
        assert source==local.read_text(encoding='utf-8'),name
    archive=HERE.parent/'noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'
    assert sha(archive.read_bytes())==ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:groups=json.loads(z.read('groups.json'))
    expected=select_sample(groups['train']);sample=objects['sample.json']
    for key,rows in expected.items():
        actual=[{k:v for k,v in row.items() if k!='source_path'} for row in sample[key]]
        assert actual==rows
    assert not {r['coco_id'] for r in sample['measure']}&{r['coco_id'] for r in groups['val']}
    workers=[objects[f'complete_rank{r}.json'] for r in range(2)]
    clock=objects['release.json']['monotonic_ns']
    for rank,worker in enumerate(workers):
        ready=objects[f'ready_rank{rank}.json'];progress=objects[f'progress_rank{rank}.json'];runtime=objects[f'runtime_rank{rank}.json']
        assert worker['status']=='COMPLETE' and worker['measured_images']==128 and progress['images']==128
        assert worker['measure_start_ns']>=clock and worker['measure_end_ns']>worker['measure_start_ns']
        assert ready['preflight']=='PASS' and ready['model_sha256']==worker['model_sha256']
        assert ready['compatibility_tolerance']==dict(atol=.01,rtol=.005)
        assert [r['coco_id'] for r in ready['compatibility']]==[r['coco_id'] for r in sample['warmup'][rank::2][:4]]
        assert len(worker['resolutions'])==128 and all(w>0 and h>0 for w,h in worker['resolutions'])
        assert runtime['gpu']['name']=='Tesla T4' and runtime['deterministic_algorithms'] and not runtime['mha_fastpath']
        assert runtime['packages']['transformers']=='4.57.1'
        log=files[f'worker_rank{rank}.log']
        assert all(f'GPU {rank}: {i}/128 measured images' in log for i in range(4,129,4))
        assert 'Traceback' not in log
    assert workers[0]['model_sha256']==workers[1]['model_sha256']
    assert objects['runtime_rank0.json']==objects['runtime_rank1.json']
    derived=estimates(workers)
    for key,value in derived.items():assert summary[key]==value,key
    assert summary['observed_hdf5_bytes']==sum(w['hdf5_bytes'] for w in workers)
    metadata=objects['old_visual_cache_metadata.json']
    for part in metadata:
        m=json.loads(part['attrs']['metadata_json'])
        assert part['attrs']['complete']=='True' and m['model_id']=='openai/clip-vit-base-patch16'
        assert m['preprocessing']=='bilinear' and m['image_size']==[224,224] and m['dataset_json_sha256']==prep['dataset_sha256']
    return dict(status='AUDIT_PASS',zip_sha256=loaded['zip_sha256'],implementation_sha256=cfg['implementation_sha256'],
        checked=['source hashes and local approved source','input identities and locked disjoint sample','both rank receipts and progress logs',
                 'release clocks and concurrent timing arithmetic','matched runtimes and frozen weight digests','global compatibility receipts and old-cache metadata'],
        measured=dict(images=256,views=2048,two_gpu_wall_seconds=derived['measured_wall_seconds'],
            preparation_seconds=prep['preparation_seconds'],parent_total_seconds=summary['total_run_seconds'],
            peak_allocated_bytes_per_gpu=max(w['peak_allocated_bytes'] for w in workers),
            peak_reserved_bytes_per_gpu=max(w['peak_reserved_bytes'] for w in workers),
            cpu_decode_preprocess_share_per_worker=[w['phase_seconds']['cpu_decode_preprocess']/sum(w['phase_seconds'].values()) for w in workers],
            hdf5_bytes=summary['observed_hdf5_bytes'],free_disk_bytes=summary['free_disk_bytes']),
        projections=dict(cache_loop_minutes=60*derived['cache_only_linear_wall_hours'],tensor_payload_bytes=derived['full_tensor_payload_bytes'],
            cache_loop_plus_observed_preparation_minutes=(derived['cache_only_linear_wall_hours']*3600+prep['preparation_seconds'])/60,
            qualification='Linear resource estimate, not ETA. Excludes full A staging, training, evaluation and changed setup/filesystem conditions.'),
        compatibility=dict(checked_images=8,max_absolute_error=max(x['max_absolute_error'] for r in range(2) for x in objects[f'ready_rank{r}.json']['compatibility']),
            passed_tolerance=dict(atol=.01,rtol=.005),old_revision_recorded=False,bitwise_identity_claim=False),
        decision='Cache extraction is feasible on the measured configuration. Next: prepare resumable full H/L cache for the locked 15000 images; retain A. Training launch requires a separate runtime/quota check.',
        limits=['Review ZIP omits actual HDF5 feature bytes; readback/finite/frozen checks are runtime receipts, not independent re-verification.',
                'No training, caption-quality result, or full-pilot budget measurement. No local model execution.',
                'Eight global compatibility examples do not prove compatibility for every image or identity of unrecorded old weights.'])


if __name__=='__main__':
    result=audit(json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8')))
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
