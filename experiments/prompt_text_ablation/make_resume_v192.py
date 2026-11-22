"""Wrap the unchanged run1 notebook with exact output API retrieval for resume."""
import ast
import hashlib
import json
from pathlib import Path
from make_notebook import once

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='FULL_ZERO_PROMPT_TEXT_RESUME_V192_API_2GPU.ipynb'


def literals(source):
    values={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            try:values[node.targets[0].id]=ast.literal_eval(node.value)
            except (TypeError,ValueError):pass
    return values


def build():
    nb=json.loads((HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
    cached=HERE/'review_cache'
    receipt=json.loads((cached/'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))
    checkpoint=json.loads((cached/'checkpoint_receipt.json').read_text(encoding='utf-8'))
    best=json.loads((cached/'best_receipt.json').read_text(encoding='utf-8'))
    selected={name:sha for name,sha in receipt['files'].items() if name in (
        'checkpoint_receipt.json','best_receipt.json','identity.json','source_receipt.json','progress.json','training.jsonl') or name.startswith('evaluations/')}
    locked=dict(source_version='ducanh3/notebook7bc83c4c76/192',checkpoint_sha256=checkpoint['sha256'],
        best_sha256=best['sha256'],metadata_sha256=selected,
        identity=json.loads((cached/'identity.json').read_text(encoding='utf-8')))
    helper=(HERE/'resume_v192_data.py').read_text(encoding='utf-8')
    setup=''.join(nb['cells'][1]['source'])
    setup=once(setup,"RUN_TAG='run1'","RUN_TAG='resume192'")
    setup=once(setup,'\nWALL_HOURS=11.5 #','\nWALL_HOURS=6 #')
    setup=once(setup,"shutil.disk_usage('/tmp').free<4*1024**3","shutil.disk_usage('/tmp').free<6*1024**3")
    setup=setup.replace('Need 4 GiB temporary scratch','Need 6 GiB temporary scratch')
    setup+='\nSOURCE_VERSION='+repr(locked['source_version'])+'\nSECRET_NAMES=(\'codex\',\'KAGGLE_API_TOKEN\')\nRESUME_LOCK='+repr(locked)+'\nRESUME_HELPER='+repr(helper)+'\nRESUME_HELPER_SHA='+repr(hashlib.sha256(helper.encode()).hexdigest())+'\nTOKENS=[]\n'
    setup+='''
def redact(text):
    for secret in TOKENS:text=text.replace(secret,'[REDACTED]')
    return text
'''
    # Reuse the established redacting subprocess wrapper; no training source changes.
    old=json.loads((ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_H5_SEED43_RESUME_V178_API_2GPU.ipynb').read_text(encoding='utf-8'))
    wrapper=''.join(old['cells'][2]['source'])
    api_test=json.loads((ROOT/'experiments/matched_visual_baseline/BASELINE_MATCHED_PARALLEL3_V188_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))
    routing=literals(''.join(api_test['cells'][3]['source']))['ROUTING_PREFLIGHT']
    execution=''.join(nb['cells'][3]['source'])
    pattern=r'(^|/)full_zero_prompt_text_seed42_run1/(checkpoint[.]bin|best[.]bin|checkpoint_receipt[.]json|best_receipt[.]json|identity[.]json|source_receipt[.]json|progress[.]json|training[.]jsonl|evaluations/.*[.]json)$'
    bootstrap='''
    if hashlib.sha256(RESUME_HELPER.encode()).hexdigest()!=RESUME_HELPER_SHA:raise ValueError('Changed resume verifier')
    (OUTPUT/'resume_helper.py').write_text(RESUME_HELPER,encoding='utf-8')
    namespace={};exec(compile(RESUME_HELPER,'resume_helper','exec'),namespace)
    if RESUME_ROOT:
        resume=namespace['verify_resume'](RESUME_ROOT,RESUME_LOCK)
        origin='mounted exact v192 output'
    else:
        api_deps=SOURCE/'kaggle-deps'
        run([sys.executable,'-m','pip','install','-q','--target',api_deps,
             'https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],600)
        from kaggle_secrets import UserSecretsClient
        token=None
        for secret_name in SECRET_NAMES:
            try:value=UserSecretsClient().get_secret(secret_name).strip()
            except Exception:continue
            if value:token=value;break
        if not token:raise RuntimeError('Enable Kaggle Secret codex or KAGGLE_API_TOKEN with access to source v192.')
        if token.startswith('sk-'):raise RuntimeError('Use a Kaggle API token, not a language-model API key.')
        TOKENS.append(token)
        api_env=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):api_env.pop(key,None)
        api_env.update(PYTHONPATH=str(api_deps),KAGGLE_API_TOKEN=token,PYTHONUNBUFFERED='1')
        run([sys.executable,'-u','-c','SOURCE_VERSION='+repr(SOURCE_VERSION)+'\\n'+ROUTING_PREFLIGHT],120,api_env)
        target=SOURCE/'resume192';target.mkdir()
        run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',SOURCE_VERSION,
             '-p',str(target),'--file-pattern',RESUME_FILE_PATTERN],1200,api_env)
        resume=namespace['verify_resume'](target,RESUME_LOCK)
        origin='exact version192 API'
    cfg['resume_root']=str(resume);write_config()
    (OUTPUT/'resume_source.json').write_text(json.dumps(dict(version=SOURCE_VERSION,origin=origin,
        checkpoint_sha256=RESUME_LOCK['checkpoint_sha256'],best_sha256=RESUME_LOCK['best_sha256'],
        resume_updates=141433,remaining_updates=35677,metadata_files_verified=len(RESUME_LOCK['metadata_sha256']),
        wrapper_verifier_sha256=RESUME_HELPER_SHA,training_sources_unchanged=True),indent=2))
    print('Exact v192 weights/history verified; continue at update141433, remaining35677. No fresh restart.',flush=True)
'''
    execution='ROUTING_PREFLIGHT='+repr(routing)+'\nRESUME_FILE_PATTERN='+repr(pattern)+'\n'+execution
    marker="    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)"
    execution=once(execution,marker,marker+'\n'+bootstrap)
    execution=once(execution,"(OUTPUT/'failure.txt').write_text(traceback.format_exc())","(OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()))")
    intro='''# Tiếp tục zero-text từ version192 — không train lại

**T4 ×2, Internet ON, Save & Run All.** Giữ cùng bốn dataset: visual-cache, vuthetam/mscoco-2014, csa-semantic-banks-v130, csa-prompt-full-karpathy-v175. Không cần tạo Dataset checkpoint mới.

Bật Secret **codex** hoặc **KAGGLE_API_TOKEN** có quyền đọc `ducanh3/notebook7bc83c4c76/192`. Để RESUME_ROOT trống thì notebook tải latest checkpoint.bin, best.bin, receipts và lịch sử VALID từ exactv192; kiểm SHA mới nạp. Nếu đã gắn toàn bộ outputv192, có thể điền RESUME_ROOT và bỏ qua API. Không dùng ZIPreview hoặc chỉbest.bin làm resume.

Lượt cũ dừng do safetydeadline: epoch8/cursor17456/update141433. Còn **35.677update** để đủ10epochs, gồm255cuối epoch8 và hai epoch9–10. Khôi phục optimizer/RNG/cursor/history bằng runner nguyên bản; không reset LR hoặc khởi tạo từbest7. Best tạm7/CIDEr119.42 chưa là kết quả ablation cuối.

Toàn bộ41training sources và source_receipt giữ đúng run1 để strictidentity hoạt động; helperAPI nằm riêng ngoài training payload. Samefull53.18Mparams/M1/M2/M3/ROI/geometry, object_text+relation_text vẫn0 trong TRAIN/VALID. CPU/GPU/data/software kiểm lại; không tự cài lại Torch hoặc fallback sang checkpoint khác.

Cap6h/phiên tiếp tục là giới hạn, không ETA/billing/quota hiện tại. CheckCUDA trước downloadweights; stageH5/semantics như cũ, khôngcopy fullpatch. Preflight bỏ64steps trên freshcopy; mainworker phải nạp latestv192. Hếtcap thì giữ output để tiếp tục, không autorestart. Kết thúc10epochs mới chấm best/end VALID; chưa TEST/SCST/rerank/M2-only.

Gửi `full_zero_prompt_text_seed42_resume192_review.zip`, giữ toàn bộoutput chứa checkpoint.bin/best.bin/receipts/evaluations/source. Local chỉ kiểm static/mock/metadata; API thật, optimizer/RNG binaryrestore và CUDA chờ lượt chạy của bạn.
'''
    for i,source in ((0,intro),(1,setup),(2,wrapper),(3,execution)):nb['cells'][i]['source']=source.splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell{i}','exec');cell['execution_count']=None;cell['outputs']=[]
    target=HERE/NAME;target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (HERE/'RESUME_V192_LOCK.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(NAME,'unchanged41 training files;',len(selected),'sealed metadata files; no API/model executed')
    return nb


if __name__=='__main__':build()
