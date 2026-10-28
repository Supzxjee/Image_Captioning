"""Audit cached review texts once loaded; no checkpoint/model/neural execution."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from compare_frames import compare
from frame_data import CHECKPOINT_SHA, FILES, META_SHA, RELATIONS, CONTACT, save

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit():
    package=json.loads((HERE/'NOTEBOOK_PACKAGE.json').read_text())
    manifest=json.loads(gzip.decompress((ROOT/'experiments/relation_frames_inventory/V2_MANIFEST.json.gz').read_bytes()))
    frozen={p['pair_id']:p for p in manifest['heldout']}
    reports={};results={}
    for arm in ('RelPos','RelContrast'):
        loaded=json.loads((HERE/'review'/f'{arm}_loaded.json').read_text(encoding='utf-8'));texts=loaded['texts']
        def read(n):return json.loads(texts[n])
        status=read('execution_status.json');p=read('protocol.json');pre=read('preflight.json');r=read('result.json');prep=read('preparation.json')
        if not status['success'] or status['arm']!=arm or status['source_commit']!=package['source_commit']:raise ValueError('Incomplete/wrong source')
        if p!=read('notebook_config.json') or p!=r['protocol']:raise ValueError('Protocol lineage differs')
        if p['source_assets']!=package['assets'] or p['source_commit']!=package['source_commit']:raise ValueError('Source package differs')
        for name,entry in package['assets'].items():
            if name.endswith('.py'):
                raw=texts['source/'+name].encode()
                blob=subprocess.check_output(['git','show',package['source_commit']+':'+entry['path']],cwd=ROOT)
                if hashlib.sha256(raw).hexdigest()!=entry['sha256'] or raw!=blob:raise ValueError('Bundled source differs: '+name)
        if p['cache_sha256']!=FILES or p['metadata_sha256']!=META_SHA or p['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Input identities differ')
        if prep['status']!='PREPARATION_PASS' or prep['payload_sha256']!=FILES or prep['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Preparation receipt differs')
        if pre['status']!='MODEL_PREFLIGHT_PASS' or pre['initial_model_digest']!=r['initial_model_digest']:raise ValueError('Preflight differs')
        for k in ('unknown_predicate_gradients_zero','synchronized_step_checked','isolated_positive_gradient_to_frame','isolated_truth_gradient_to_frame','constant_feature_pooling_invariant','state_restored'):
            if pre[k] is not True:raise ValueError('Failed preflight '+k)
        if pre['trainable_parameters']!=4720672 or pre['frozen_parameters']!=5120000 or pre['auxiliary_output_head_parameters']!=2570000:raise ValueError('Parameter count differs')
        if len(r['rows'])!=64 or len({v['pair_id'] for v in r['rows']})!=64 or {v['pair_id'] for v in r['rows']}!=set(frozen):raise ValueError('Heldout IDs differ')
        for v in r['rows']:
            f=frozen[v['pair_id']];gap=v['likelihood_gap']
            if v['relation']!=RELATIONS.index(f['relation']) or v['stratum']!=('contact_proximity' if f['relation'] in CONTACT else 'projective_containment'):raise ValueError('Heldout relation differs')
            if not np.isfinite(gap) or v['correct']!=float(gap>0) or v['tie']!=(gap==0):raise ValueError('Invalid ranking row')
        if r['correct_image_accuracy']!=np.mean([v['correct'] for v in r['rows']]):raise ValueError('Accuracy receipt differs')
        logs=[json.loads(line) for line in texts['training.jsonl'].splitlines() if line.strip()]
        if [v['step'] for v in logs]!=[1]+list(range(10,201,10)):raise ValueError('Endpoint trace differs')
        for v in logs:
            expected=v['pos'] if arm=='RelPos' else v['pos']+.5*v['truth']+.5*v['pair']
            if v['arm']!=arm or not np.isfinite([v[k] for k in ('loss','pos','truth','pair','grad_norm','seconds')]).all() or abs(v['loss']-expected)>1e-5:raise ValueError('Training objective trace differs')
        reports[arm]=dict(zip_sha256=loaded['zip_sha256'],checkpoint_sha256_reported=r['checkpoint_sha256'],
            initial_model_digest=r['initial_model_digest'],frozen_embedding_digest=pre['frozen_embedding_digest'],
            trainable_parameters=pre['trainable_parameters'],frozen_parameters=pre['frozen_parameters'],
            gpu=pre['gpu'],torch=pre['torch'],model_preflight='PASS',logged_updates=[logs[0]['step'],logs[-1]['step']],
            logged_first=logs[0],logged_last=logs[-1],device_gpu_hours_reported=r['device_gpu_hours'],
            near_zero_gaps_below_1e_5=sum(abs(v['likelihood_gap'])<1e-5 for v in r['rows']))
        results[arm]=r
    if reports['RelPos']['frozen_embedding_digest']!=reports['RelContrast']['frozen_embedding_digest']:raise ValueError('Frozen embeddings differ')
    decision=compare(results['RelPos'],results['RelContrast'])
    x={v['pair_id']:v['correct'] for v in results['RelPos']['rows']};y={v['pair_id']:v['correct'] for v in results['RelContrast']['rows']}
    decision.update(date='2026-10-28',audit='SOURCE_PROTOCOL_PREFLIGHT_AND_FIXED_PAIR_METADATA_PASS',
        source_commit=package['source_commit'],runs=reports,
        counts=dict(RelPos=sum(x.values()),RelContrast=sum(y.values()),n=64),
        discordance=dict(contrast_only=sum(y[k]>x[k] for k in x),pos_only=sum(x[k]>y[k] for k in x),
            both_correct=sum(x[k]==y[k]==1 for k in x),both_wrong=sum(x[k]==y[k]==0 for k in x)),
        actual_checkpoint_or_hdf5_rehashed_locally=False,local_neural_execution=False,
        interpretation='Observed-predicate training loss falls, but primary common GRU likelihood gate fails. Training loss is not heldout truth accuracy. Opposite subgroup changes are descriptive, not a mechanism diagnosis.')
    save(HERE/'RESULT.json',decision)
    return decision


if __name__=='__main__':
    r=audit();print(json.dumps({k:r[k] for k in ('audit','status','counts','difference','paired_difference_ci95','strata','discordance')},indent=2))
