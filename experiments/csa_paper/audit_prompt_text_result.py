"""Stored-text/cache-receipt audit only; never import or execute neural code."""
import hashlib
from pathlib import Path
import subprocess
from real_bank_data import read,save,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    p=HERE/'prompt_text_review_text';m=read(p/'cache_manifest.json');r=read(p/'cache_receipt.json');s=read(p/'run_status.json')
    source=read(p/'source_receipt.json');archive=read(p/'saved_text_receipt.json');checks=0
    def check(value):
        nonlocal checks
        if not value:raise AssertionError('Stored cache check failed:'+str(checks+1))
        checks+=1
    check(m['status']==r['status']=='PROMPT_TEXT_CACHE_COMPLETE' and s['complete'] and s['neural_fixture_passed'])
    check(not any(s[k] for k in ('training','test_used','actual_checkpoint161_smoke')))
    check(source['commit']=='418a717969eda67eb7f362efc1e22e14521b2337' and len(source['sha256'])==22)
    for name,h in source['sha256'].items():
        check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)).hexdigest()==h)
    check(m['source_version']=='ducanh2403/2-vi-cai-kia-qua-nhieu/167')
    old=read(HERE/'prompt_mapping_review_text/receipt.json');mapping=read(HERE/'PROMPT_MAPPING_RESULT.json')
    check(m['mapping_receipt_sha256']==sha(HERE/'prompt_mapping_review_text/receipt.json'))
    check(m['source_payload_sha256']==old['payload_sha256'])
    check(m['validation']['images']==15000 and m['validation']['unique_texts']==318 and not m['validation']['test_used'])
    for group in ('train','val'):
        check(all(m['validation']['groups'][group][k]==mapping['counts'][group][k] for k in ('images','kept_objects','mapped_relations')))
    spec=read(HERE/'RERANK_TEST_INPUTS.json')['clip']
    check(m['model']==spec['id'] and m['revision']==spec['revision'])
    check(m['files_sha256']==spec['files_sha256'] and len(m['files_sha256'])==8)
    check(all(not m['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')))
    check(m['texts']==318 and m['shape']==[318,512] and m['dtype']=='float32' and m['device']=='cpu' and m['frozen'])
    check(not m['training'] and not m['test_used'] and not m['truncation'] and m['max_tokens']==13)
    check(set(r['payload_sha256'])=={'prompt_embeddings.npy','prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json'})
    check(sha(p/'cache_manifest.json')==r['payload_sha256']['cache_manifest.json'])
    for name in ('prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json'):
        check(r['payload_sha256'][name]==m['source_payload_sha256'][name])
    for name,v in archive['files'].items():check(sha(p/name)==v['sha256'])
    log=(p/'notebook.log').read_text(encoding='utf-8')
    for marker in ('Ran 7 tests','Ran 5 tests','Ran 3 tests','Isolated HF dependency versions PASS',
                   'test_closed_and_empty_identity_and_no_nan_rows','test_opened_text_path_changes_memory_and_has_finite_gradients',
                   'test_invalid_pair_or_padding_is_rejected','CLIP text CPU 318 / 318','PROMPT TEXT CACHE COMPLETE'):
        check(marker in log)
    check('Traceback' not in log and 'FAILED' not in log)
    result=dict(status='PROMPT_TEXT_CACHE_STORED_REVIEW_PASS',checks=checks,zip_sha256=archive['zip_sha256'],
        user_reported_output_source='ducanh2403/2-vi-cai-kia-qua-nhieu/169',source=source['commit'],
        mapping_source=m['source_version'],texts=318,embedding_shape=[318,512],dtype='float32',device='cpu',frozen=True,
        cohort=m['validation'],model=m['model'],revision=m['revision'],clip_assets_sha256=m['files_sha256'],
        max_tokens=13,truncation=False,phase_seconds=m['seconds'],payload_sha256=r['payload_sha256'],
        user_neural_fixture_tests=3,actual_checkpoint161_smoke=False,caption_training=False,test_used=False,
        local_author_models_or_scorers_executed=False,
        user_gpu_budget='15h reported; per-account versus combined not specified, do not assume30h',
        next_step='Preserve complete version169 cache; prepare paired500-update CONTROL/PROMPT from exact161, real-loaded checkpoint/CUDA/DDP guards before training.',
        limits=['Embedding NPY/links/texts and originalCLIP weights absent review; their hashes/norm checks are user-run evidence.',
                'Three CPU fixtures use a small randomly initialized CSA, not the loaded161 captioner or CUDA/DDP.',
                'Opened adapter text sensitivity/finite gradients in fixtures are not learned caption or relation-grounding performance.',
                '19.46s phase excludes setup/API/dependency installs/fixture tests and is not training ETA or GPU billing.',
                'Torch version is not recorded in the cache manifest; do not infer it from older runs.',
                'Pip dependency conflict warnings did not stop this run; COMPLETE and logs verified, no author model rerun.'])
    save(HERE/'PROMPT_TEXT_CACHE_RESULT.json',result)
    print('PASS',checks,'metadata/text/source/cache-receipt checks; source169; user3CPUfixtures PASS; no caption metric.')


if __name__=='__main__':run()
