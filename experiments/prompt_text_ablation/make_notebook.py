"""Prepare the zero-text arm with the existing full per-caption XE10 recipe."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_H5_SEED43_XE10_2GPU.ipynb'
METHOD = 'full_zero_prompt_text_xe10_v1'
STATUS = 'FULL_ZERO_PROMPT_TEXT_XE10_COMPLETE'
NAME = 'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Changed source anchor: '+old[:100])
    return source.replace(old, new, 1)


def source_assignments(source):
    wanted = {'REVISION','MAPPING','HASHES','SEED_SOURCES'}
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in wanted}


def build():
    nb = json.loads(PARENT.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    original = source_assignments(setup)
    payloads, origins = {}, {}
    for path, name in original['MAPPING'].items():
        raw = subprocess.check_output(['git','show',original['REVISION']+':'+path],cwd=ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != original['HASHES'][path]:
            raise ValueError('Changed pinned source: '+path)
        payloads[name] = raw.decode('utf-8')
        origins[name] = dict(commit=original['REVISION'],path=path,sha256=digest)
    for name, code in original['SEED_SOURCES'].items():
        payloads[name] = code
        origins[name]['seed_scaffold_override'] = True
    runner = payloads['run_csa_full.py']
    runner = runner.replace('csa_structure_patch_prompt_grouped_per_caption_accelerate_v1',METHOD)
    runner = once(runner,"schema='clip_patch_full_xe10_v1'","schema='full_zero_prompt_text_xe10_v1'")
    runner = once(runner,"method='"+METHOD+"',software=", "method='"+METHOD+"',prompt_text_mode='zero',software=")
    payloads['run_csa_full.py'] = runner
    runtime = payloads['csa_full_runtime.py']
    runtime = once(runtime,'    from csa_patch_model import CSAPatchCaptioner','    from zero_prompt_model import ZeroTextCaptioner')
    runtime = once(runtime,"    if not cfg.get('csa_per_caption_full') or cfg.get('semantic_strength_gate'):",
                   "    if (not cfg.get('csa_per_caption_full') or cfg.get('semantic_strength_gate') or not cfg.get('semantic')\n        or cfg.get('prompt_text_mode')!='zero' or cfg.get('method')!='full_zero_prompt_text_xe10_v1' or cfg.get('seed')!=42):")
    runtime = once(runtime,'    return CSAPatchCaptioner(vocab,*banks)','    return ZeroTextCaptioner(vocab,*banks)')
    marker = "    context=SemanticContext(cfg['stage']);order=order_for_epoch(data,1,cfg.get('seed',42));model.eval();initial=tensor_digest(model)"
    runtime = once(runtime,marker,"    from zero_prompt_model import contract_guard\n    contract_guard(model,len(words),cfg,rank,device,out)\n"+marker)
    payloads['csa_full_runtime.py'] = runtime
    scoring = payloads['score_csa_full.py'].replace('csa_structure_patch_prompt_grouped_per_caption_accelerate_v1',METHOD).replace('CSA_PER_CAPTION_FULL_XE10_COMPLETE',STATUS)
    a, b = scoring.index("    r['limits']+="), scoring.index('    save(path,r)')
    scoring = scoring[:a]+'''    r['prompt_text_mode']='zero'
    r['zero_text_contract']=read(Path(cfg['output'])/'zero_text_contract_rank0.json')
    r['limits']=['Full model with object_text and relation_text zeroed in both training and evaluation.',
        'M1/M2 banks, all adapters, ROI links, geometry and declared parameter count retained.',
        'One seed/reused VALID; inactive text-input columns are not an equal-effective-capacity claim.',
        'Compared with historical fullseed42 recipe; direct-H5 versus historical local patch staging.']
''' + scoring[b:]
    payloads['score_csa_full.py'] = scoring
    payloads['zero_prompt_model.py'] = (HERE/'zero_prompt_model.py').read_text(encoding='utf-8')
    payloads['ZERO_PROMPT_TEXT_PROTOCOL.md'] = (HERE/'PROTOCOL.md').read_text(encoding='utf-8')
    for name in ('run_csa_full.py','csa_full_runtime.py','score_csa_full.py','zero_prompt_model.py'):
        origins[name] = dict(origins.get(name,{}),zero_text_override=True)
    hashes = {name:hashlib.sha256(code.encode('utf-8')).hexdigest() for name,code in payloads.items()}
    for name, code in payloads.items():
        if name.endswith('.py'):ast.parse(code,filename=name)
    lines = setup.splitlines(keepends=True)
    for node in reversed(ast.parse(setup).body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('MAPPING','HASHES','SEED_SOURCES'):
            del lines[node.lineno-1:node.end_lineno]
    setup = ''.join(lines)
    setup = once(setup,'SEED=43','SEED=42')
    setup = once(setup,"OUTPUT=Path(f'/kaggle/working/csa_per_caption_h5_seed{SEED}_{RUN_TAG}')",
                 "OUTPUT=Path(f'/kaggle/working/full_zero_prompt_text_seed{SEED}_{RUN_TAG}')")
    setup = once(setup,'cfg=dict(seed=SEED,csa_per_caption_full=True,',
                 "cfg=dict(method='"+METHOD+"',prompt_text_mode='zero',seed=SEED,csa_per_caption_full=True,")
    setup = once(setup,"    if json.loads((root/'identity.json').read_text())['seed']!=SEED:\n        raise ValueError('Resume seed differs')",
                 "    prior=json.loads((root/'identity.json').read_text())\n    if prior['seed']!=SEED or prior.get('method')!=cfg['method'] or prior.get('prompt_text_mode')!='zero':\n        raise ValueError('Resume only this zero-text arm; full-prompt/baseline checkpoints are incompatible')")
    setup += '\nSOURCE_PAYLOADS = '+repr(payloads)+'\nSOURCE_SHA256 = '+repr(hashes)+'\nSOURCE_ORIGINS = '+repr(origins)+'\n'
    run = ''.join(nb['cells'][3]['source'])
    a = run.index('    upstream=')
    b = run.index("    run([sys.executable,'-u',SOURCE/'test_full_data.py']",a)
    run = run[:a]+'''    preserved=OUTPUT/'source';preserved.mkdir()
    for name,code in SOURCE_PAYLOADS.items():
        if Path(name).name!=name:raise ValueError('Invalid source name')
        raw=code.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed embedded source: '+name)
        (SOURCE/name).write_bytes(raw);(preserved/name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=SOURCE_SHA256,origins=SOURCE_ORIGINS),indent=2))
''' + run[b:]
    run = once(run,'from csa_patch_model import CSAPatchCaptioner;from csa_full_runtime import make_model;',
               'from zero_prompt_model import ZeroTextCaptioner;from csa_full_runtime import make_model;')
    run = run.replace('CSA_PER_CAPTION_FULL_XE10_COMPLETE',STATUS).replace('CSA PER-CAPTION FULL10 COMPLETE;','ZERO-TEXT FULL10 COMPLETE;')
    run = once(run,'dict(complete=complete,seed=SEED,epochs_requested=10,test_used=False,SCST=False,reranking=False)',
               "dict(complete=complete,method=cfg['method'],prompt_text_mode='zero',seed=SEED,epochs_requested=10,test_used=False,SCST=False,reranking=False)")
    run = once(run,"('.json','.jsonl','.log','.txt')","('.json','.jsonl','.log','.txt','.py','.md')")
    intro = '''# Ablation: full model, zero prompt text — seed42 / CE10

**GPU T4 ×2, Internet ON, Save & Run All.** Người dùng báo25hGPU còn lại; cap11.5h/phiên là giới hạn, không ETA hoặc giờ tính quota.

Gắn bốn Dataset đã có:
1. `ducanh2403/visual-cache`
2. `vuthetam/mscoco-2014`
3. `ducanh2403/csa-semantic-banks-v130`
4. `ducanh2403/csa-prompt-full-karpathy-v175`

**Giữ nguyên full model53.183.736tham số, bank/M2, hai adapter M3, ROI, validmasks và hình học. Chỉ object_text và relation_text của M3 thành0 trong mọi forward/encode, cả train và VALID.** Không zero noun/verb bank; không tắt M3. Constructor và state keys như fullmodel, freshseed42, không dùng checkpoint đã học.

Same10epochCE/113287TRAIN/5000VALID/global32/177110updates, grouped42+epoch, FP32/Accelerate1.15.0/DDP, percaption/no sharing/AMP/compile. Best=maxVALIDCIDEr mỗi epoch, beam5/max20; noSCST/rerank/TEST. DirectH5, không chép fullpatch, không chạyCLIP/UPT/YOLO lại. Nguồn nhúng/hash sẵn, không cần Secret.

Preflight Kaggle kiểm cùng tham số/khởi tạo với constructorfull; ROI/geometry/masks còn nguyên; text0 không đổi khi thay chữ. Guard bật outputprojection trên **copy** để không bị adapteroutputkhởi tạo0 che lỗi. Giữ gradientDDPfirst32/tail27,causal/beam và64discarded timingsteps; mainworker khởi tạo lại. Local chỉ AST/mock, chưa chạy mô hình.

Nếu hếtcap, giữ toàn bộ OUTPUT (checkpoint.bin,best.bin,identity,receipts,evaluations,source). Gắn output cũ, điền RESUME_ROOT và đổi RUN_TAG='run2' để tiếp tục **chính zero-textarm này**; không nạp checkpoint fullprompt/baseline. Không tự restart hoặc đổi lịch học. Mộtseed chưa đủ đánh giá độổnđịnh; historicalfullseed42 cókhácI/Ostaging, không cam kết bitwise replay.

Sau10epochs: VALID CIDEr/BLEU1/4/ROUGE-L/CHAIRs/i/recall trênbest/end đã lưu. METEOR/SPICE/TEST để lượt sau review. Gửi `full_zero_prompt_text_seed42_run1_review.zip`; giữ đầy đủ trọng số. Cùng tham số khai báo nhưng các cột textnhận0 không có tín hiệu tương ứng; không gọi là equal-effectivecapacity. Chưa mở M2-only/nhiềuseed từ notebook này.
'''
    for index, source in ((0,intro),(1,setup),(3,run)):nb['cells'][index]['source']=source.splitlines(keepends=True)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'cell{i}','exec');cell['execution_count']=None;cell['outputs']=[]
    target=HERE/NAME
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(parent_notebook=str(PARENT.relative_to(ROOT)),
        parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),revision=original['REVISION'],method=METHOD,
        sha256=hashes,origins=origins,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
    print(NAME,len(payloads),'embedded sources; no models executed')
    return nb,payloads


if __name__=='__main__':
    build()
