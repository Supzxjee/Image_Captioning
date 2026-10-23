"""Audit a cached full H/L review ZIP once, without model or feature execution."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'native_detail_benchmark'))
from cache_data import identity,plan,spec_for,EXPECTED_MODELS
from tile_data import ARCHIVE_SHA,REVISION,select_sample
from run_benchmark import A_SHA,DATA_SHA


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def audit(loaded):
    files=loaded['files'];objects={n:json.loads(v) for n,v in files.items() if n.endswith('.json')}
    receipt=objects['cache_receipt.json'];spec=objects['cache_spec.json'];manifest=objects['manifest.json']
    attempts=[n.rsplit('/',1)[0] for n,v in objects.items() if n.endswith('/run_summary.json') and v['status']=='COMPLETE']
    if len(attempts)!=1:raise ValueError('This audit expects the supplied single completed attempt; inspect history before extending it.')
    prefix=attempts[0]+'/';cfg=objects[prefix+'protocol.json'];prep=objects[prefix+'preparation.json'];summary=objects[prefix+'run_summary.json']
    assert receipt['status']=='CACHE_COMPLETE' and receipt['images']==15000 and receipt['train_images']==10000 and receipt['val_images']==5000
    assert not receipt['benchmark_only'] and not receipt['trained'] and not receipt['caption_evaluated']
    assert receipt['model_sha256']==EXPECTED_MODELS and receipt['spec_sha256']==identity(spec) and receipt['manifest_sha256']==identity(manifest)
    assert receipt['schema']==spec['schema']=='native_detail_HL_15k_v1'
    assert spec==spec_for(manifest,cfg['implementation_sha256'],ARCHIVE_SHA,REVISION)
    assert prep['status']=='COMPLETE' and prep['checkpoint_sha256']==A_SHA and prep['dataset_sha256']==DATA_SHA
    assert prep['groups_archive_sha256']==ARCHIVE_SHA and prep['source_files_checked']==15000
    archive=HERE.parent/'noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:groups=json.loads(z.read('groups.json'))
    # The public COCO filename identifies its physical train2014/val2014 directory; group is the locked split.
    rows=groups['train']+groups['val'];data=[None]*(max(r['image_id'] for r in rows)+1)
    for row in rows:data[row['image_id']]=dict(filename=row['filename'],filepath=row['filename'].split('_')[1])
    assert manifest==plan(groups,data)
    sample=objects[prefix+'sample.json'];locked=select_sample(groups['train'])
    assert prep['sample_sha256']==sha(files[prefix+'sample.json'])
    for key in ('warmup','measure'):
        assert [{k:v for k,v in r.items() if k!='source_path'} for r in sample[key]]==locked[key]
    assert len(manifest)==30 and len(receipt['shards'])==30
    records=[]
    for shard in manifest:
        record=objects[shard['name']+'.json'];assert record['file']==shard['name']+'.h5' and record['images']==500
        assert record['rows_sha256']==identity(shard['rows']) and record['spec_sha256']==identity(spec)
        assert record['model_sha256']==EXPECTED_MODELS and record['runtime_sha256']==receipt['runtime_sha256']
        assert len(record['sha256'])==64 and all(c in '0123456789abcdef' for c in record['sha256'])
        assert record['bytes']>=500*2*196*512*2
        records.append(record)
    assert records==receipt['shards']
    workers=[]
    for rank in range(2):
        worker=objects[prefix+f'complete_rank{rank}.json'];runtime=objects[prefix+f'runtime_rank{rank}.json'];preflight=objects[prefix+f'preflight_rank{rank}.json']
        assert worker['status']=='COMPLETE' and worker['total_images']==7500 and worker['new_images']==7500 and worker['reused_images']==0
        assert worker['model_sha256']==EXPECTED_MODELS and worker['runtime_sha256']==receipt['runtime_sha256']==identity(runtime)
        assert runtime['gpu']['name']=='Tesla T4' and runtime['deterministic_algorithms'] and not runtime['mha_fastpath']
        assert runtime['packages']['transformers']=='4.57.1'
        assert preflight['status']=='PASS' and preflight['model_sha256']==EXPECTED_MODELS and preflight['atol']==.01 and preflight['rtol']==.005
        assert [r['coco_id'] for r in preflight['compatibility']]==[r['coco_id'] for r in locked['warmup'][rank::2][:4]]
        log=files[prefix+f'worker_rank{rank}.log']
        assert 'Traceback' not in log
        mine=[s['name'] for s in manifest if s['rank']==rank]
        for i,name in enumerate(mine):assert f'GPU {rank}: committed {name}; {(i+1)*500}/7500 images' in log
        workers.append(worker)
    assert objects[prefix+'runtime_rank0.json']==objects[prefix+'runtime_rank1.json'] and summary['workers']==workers
    assert summary['seconds']>max(w['seconds'] for w in workers)+prep['preparation_seconds']
    for name,expected in cfg['implementation_sha256'].items():
        candidates=[v for n,v in files.items() if n.startswith('notebook_attempts/') and n.endswith('/'+name) and sha(v)==expected]
        assert candidates,name
        path=HERE/name
        if name.startswith('captioning/'):path=HERE.parents[1]/name
        elif not path.exists():path=HERE.parent/'native_detail_benchmark'/name
        assert candidates[0]==path.read_text(encoding='utf-8'),name
    logs=[v for n,v in files.items() if n.endswith('/notebook.log')]
    assert any('Ran 10 tests' in v and '\nOK\n' in v for v in logs)
    return dict(status='RECEIPT_AUDIT_PASS',zip_sha256=loaded['zip_sha256'],spec_sha256=identity(spec),manifest_sha256=identity(manifest),
        images=dict(train=10000,val=5000,total=15000),shards=30,hdf5_bytes=sum(r['bytes'] for r in records),
        parent_wall_seconds=summary['seconds'],preparation_seconds=prep['preparation_seconds'],worker_seconds=[w['seconds'] for w in workers],
        peak_gpu_allocated_bytes=max(w['peak_allocated_bytes'] for w in workers),peak_gpu_reserved_bytes=max(w['peak_reserved_bytes'] for w in workers),
        model_sha256=EXPECTED_MODELS,runtime_sha256=receipt['runtime_sha256'],fresh_extraction=True,
        checks=['30 per-shard receipts agree with final receipt and exact locked split/order','all approved source hashes and saved source match',
                'frozen model identities match passed benchmark and both workers','two runtime digests and complete per-shard progress logs agree',
                'all10 notebook tests passed, including4 HDF5 checks','eight global compatibility receipts passed original mixed tolerance'],
        full_payload_independently_rehashed_locally=False,trained=False,caption_evaluated=False,
        decision='Preserve the entire native_detail_HL_15k folder as a Kaggle dataset. Obtain its input mount path, then prepare the shared-adapter H/L pilot and measure training/evaluation cost before the full5000-update run.',
        limits=['The review ZIP omits HDF5 payloads. Hash/readback/finite/frozen assertions are consistent runtime evidence; the consuming notebook must rehash actual input files.',
                'Fresh full extraction succeeded. Real interrupted full-cache resume has not been demonstrated; its synthetic HDF5 tests passed.',
                'This establishes cache readiness at receipt level, not caption improvement, novel alignment or full training-budget feasibility.'])


if __name__=='__main__':
    result=audit(json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8')))
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
