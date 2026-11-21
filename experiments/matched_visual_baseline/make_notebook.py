"""Build the source-contained, user-run matched baseline from the proven XE runner."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/clip_patch_baseline'
METHOD = 'matched_visual_parallel3_xe10_v1'
STATUS = 'MATCHED_VISUAL_PARALLEL3_XE10_COMPLETE'
NAME = 'BASELINE_MATCHED_PARALLEL3_XE10_2GPU.ipynb'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Changed source anchor: ' + old[:120])
    return source.replace(old, new, 1)


def assignments(source):
    names = {'REVISION', 'MAPPING', 'HASHES', 'SEED_SOURCES'}
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


def remove_assignments(source, names):
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            del lines[node.lineno-1:node.end_lineno]
    return ''.join(lines)


def build():
    original = OLD / 'CSA_PER_CAPTION_H5_SEED43_XE10_2GPU.ipynb'
    nb = json.loads(original.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    pins = assignments(setup)
    payloads, origins = {}, {}
    for path, destination in pins['MAPPING'].items():
        raw = subprocess.check_output(['git', 'show', pins['REVISION'] + ':' + path], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != pins['HASHES'][path]:
            raise ValueError('Pinned source hash differs: ' + path)
        payloads[destination] = raw.decode('utf-8')
        origins[destination] = dict(commit=pins['REVISION'], path=path, sha256=hashlib.sha256(raw).hexdigest())
    for name, code in pins['SEED_SOURCES'].items():
        payloads[name] = code
        origins[name]['seed_scaffold_override'] = True

    runner = payloads['run_csa_full.py']
    runner = once(runner, '    from semantic_data import SemanticContext',
                  '    from visual_only_model import VisualOnlyContext as SemanticContext')
    runner = once(runner,
                  "        if stage.get('semantic',{}).get('method')!='bank_prompt_patch_v1':raise ValueError('Need complete full semantic stage')",
                  "        if cfg.get('semantic') or stage.get('semantic'):raise ValueError('Visual-only baseline cannot use semantic staging')")
    runner = runner.replace('csa_structure_patch_prompt_grouped_per_caption_accelerate_v1', METHOD)
    runner = runner.replace("schema='clip_patch_full_xe10_v1'", "schema='matched_visual_parallel3_xe10_v1'")
    runner = once(runner, "method='" + METHOD + "',software=", "method='" + METHOD + "',decoder_layers=3,semantic_conditioning=False,software=")
    payloads['run_csa_full.py'] = runner

    runtime = payloads['csa_full_runtime.py']
    start = runtime.index('def make_model(')
    end = runtime.index('\n\ndef ', start + 1)
    runtime = runtime[:start] + '''def make_model(vocab,cfg):
    from visual_only_model import VisualOnlyCaptioner
    if (cfg.get('method')!='matched_visual_parallel3_xe10_v1' or cfg.get('semantic')
        or cfg.get('hybrid_sharing') or cfg.get('semantic_strength_gate') or cfg.get('seed')!=42):
        raise ValueError('Expected fresh seed42 visual-only matched baseline recipe')
    return VisualOnlyCaptioner(vocab)
''' + runtime[end:]
    runtime = once(runtime, '    from semantic_data import SemanticContext',
                   '    from visual_only_model import VisualOnlyContext as SemanticContext, contract_guard')
    runtime = once(runtime, "    context=SemanticContext(cfg['stage']);order=order_for_epoch(data,1,cfg.get('seed',42));model.eval();initial=tensor_digest(model)",
                   "    contract_guard(model,len(words),cfg,rank,device,out)\n    context=SemanticContext(cfg['stage']);order=order_for_epoch(data,1,cfg.get('seed',42));model.eval();initial=tensor_digest(model)")
    payloads['csa_full_runtime.py'] = runtime
    scoring = payloads['score_csa_full.py'].replace('csa_structure_patch_prompt_grouped_per_caption_accelerate_v1', METHOD)
    scoring = scoring.replace('CSA_PER_CAPTION_FULL_XE10_COMPLETE', STATUS)
    a = scoring.index("    r['limits']+=")
    b = scoring.index('    save(path,r)', a)
    scoring = scoring[:a] + '''    r['limits'] = ['One seed and reused VALID; TEST not evaluated.',
        'Same three-layer parallel decoder and training recipe as the final model; M2/M3 removed.',
        'Total parameter count differs; this tests the entire semantic addition, not prompt text alone.',
        'Both decoder attention branches read the same projected visual tokens.']
    r['baseline_contract'] = read(Path(cfg['output'])/'baseline_contract_rank0.json')
''' + scoring[b:]
    payloads['score_csa_full.py'] = scoring
    payloads['visual_only_model.py'] = (HERE/'visual_only_model.py').read_text(encoding='utf-8')
    payloads['MATCHED_BASELINE_PROTOCOL.md'] = (HERE/'PROTOCOL.md').read_text(encoding='utf-8')
    for name in ('run_csa_full.py', 'csa_full_runtime.py', 'score_csa_full.py', 'visual_only_model.py'):
        origins[name] = dict(origins.get(name, {}), matched_baseline_override=True)
    hashes = {name: hashlib.sha256(code.encode('utf-8')).hexdigest() for name, code in payloads.items()}
    for name, code in payloads.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)

    setup = remove_assignments(setup, {'MAPPING', 'HASHES', 'SEED_SOURCES'})
    setup = once(setup, "BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')\n", '')
    setup = once(setup, "PROMPT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')\n", '')
    setup = once(setup, 'SEED=43', 'SEED=42')
    setup = once(setup, "OUTPUT=Path(f'/kaggle/working/csa_per_caption_h5_seed{SEED}_{RUN_TAG}')",
                 "OUTPUT=Path(f'/kaggle/working/baseline_matched_parallel3_seed{SEED}_{RUN_TAG}')")
    setup = once(setup, 'for p in (VISUAL_CACHE,COCO_JSON,BANK_ROOT,PROMPT_ROOT):', 'for p in (VISUAL_CACHE,COCO_JSON):')
    setup = once(setup, "cfg=dict(seed=SEED,csa_per_caption_full=True,semantic=True,bank_root=str(BANK_ROOT),prompt_root=str(PROMPT_ROOT),",
                 "cfg=dict(method='" + METHOD + "',seed=SEED,csa_per_caption_full=True,semantic=False,")
    setup = once(setup, "    if json.loads((root/'identity.json').read_text())['seed']!=SEED:\n        raise ValueError('Resume seed differs')",
                 "    identity=json.loads((root/'identity.json').read_text())\n    if identity['seed']!=SEED or identity.get('method')!=cfg['method']:\n        raise ValueError('Resume must be from this matched baseline and seed; full-model/old-baseline checkpoints are incompatible')")
    setup += '\nSOURCE_PAYLOADS = ' + repr(payloads) + '\nSOURCE_SHA256 = ' + repr(hashes) + '\nSOURCE_ORIGINS = ' + repr(origins) + '\n'

    run = ''.join(nb['cells'][3]['source'])
    start = run.index('    upstream=')
    end = run.index("    run([sys.executable,'-u',SOURCE/'test_full_data.py']", start)
    run = run[:start] + '''    preserved=OUTPUT/'source';preserved.mkdir()
    for name,code in SOURCE_PAYLOADS.items():
        if Path(name).name!=name:raise ValueError('Invalid source filename')
        raw=code.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed embedded source: '+name)
        (SOURCE/name).write_bytes(raw);(preserved/name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=SOURCE_SHA256,origins=SOURCE_ORIGINS),indent=2))
''' + run[end:]
    run = once(run, "from csa_patch_model import CSAPatchCaptioner;from csa_full_runtime import make_model;print('Neural module imports PASS; no model instantiated',flush=True)",
               "from visual_only_model import VisualOnlyCaptioner;from csa_full_runtime import make_model;print('Matched baseline imports PASS; no model instantiated',flush=True)")
    run = run.replace('CSA_PER_CAPTION_FULL_XE10_COMPLETE', STATUS)
    run = run.replace('CSA PER-CAPTION FULL10 COMPLETE; VALID only, no TEST/SCST/rerank.',
                      'MATCHED BASELINE FULL10 COMPLETE; VALID only, no TEST/SCST/rerank.')
    run = once(run, 'dict(complete=complete,seed=SEED,epochs_requested=10,test_used=False,SCST=False,reranking=False)',
               "dict(complete=complete,method=cfg['method'],seed=SEED,epochs_requested=10,test_used=False,SCST=False,reranking=False)")
    run = once(run, "('.json','.jsonl','.log','.txt')", "('.json','.jsonl','.log','.txt','.py','.md')")

    markdown = '''# Baseline thị giác — cùng decoder 3 lớp với mô hình cuối

**Người dùng tự chạy Kaggle: GPU T4 ×2, Internet ON, Save & Run All.**

Chỉ gắn hai Dataset:
1. `/kaggle/input/datasets/ducanh2403/visual-cache`
2. `/kaggle/input/datasets/vuthetam/mscoco-2014` (có `dataset_coco.json`).

Patch CLIP ViT-B/16 đã cache → linear768→512 → cùng V vào hai nhánh của decoder parallel3lớp/dropout0.2 → vocabulary head. **Không M1/M2/M3, không prompt/ROI/detector, không khởi tạo từ checkpoint đã học.** Constructor tạm tạo rồi bỏ module ngữ nghĩa để giữ chính xác thứ tự RNG và trọng số khởi đầu của các phần chung; constructor không đọc bank/prompt Dataset.

Fresh seed42,10epoch CE,113287TRAIN/5000VALID,566747caption events/epoch,global32,177110updates. Cùng sampler, FP32, optimizer/lịch học, beam5/max20 với full model; best=maxVALID CIDEr, tie giữ epoch sớm. Accelerate1.15.0/DDP haiGPU; không feature sharing/AMP/compile. Đọc H5 trực tiếp, không chép full cache. Nguồn đã nhúng và kiểm SHA; chỉ dependencies/evaluator tải qua Internet. Không cần Kaggle Secret.

**Trước khi chạy kiểm tra quota còn lại.** Cap11.5h là giới hạn, không phải ETA/billing. Preflight discarded64updates kiểm tra CUDA/DDP, khởi tạo tensor chung, hai memory bằng V, prompt không ảnh hưởng, causal/beam và gradient first32/tail27; main worker khởi tạo lại. Chưa chạy neural/local; GPU PASS phải có từ lượt của bạn.

Nếu chạm cap, giữ **toàn bộ OUTPUT** gồm checkpoint.bin/best.bin/receipts/evaluations. Để tiếp tục chính lượt baseline này, gắn output cũ, điền RESUME_ROOT ở cell cấu hình và đổi RUN_TAG='run2'. Không đưa checkpoint full model hoặc baseline199 vào. Không tự restart/tăng cap.

Sau10epoch chấm các caption best/end trên VALID: BLEU1/4,ROUGE-L,CIDEr,CHAIRs/i và recall. METEOR/SPICE và TEST chưa thực hiện ở notebook này. Gửi `baseline_matched_parallel3_seed42_run1_review.zip`; giữ trọng số đầy đủ. Một seed và reusedVALID; tổng tham số khác full model, nên đây không phải đối chứng riêng embedding chữ của M3.
'''
    for index, source in ((0, markdown), (1, setup), (3, run)):
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell{i}', 'exec')
            cell['execution_count'] = None
            cell['outputs'] = []
    target = HERE / NAME
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(
        method=METHOD, parent_notebook=str(original.relative_to(ROOT)), parent_notebook_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        revision=pins['REVISION'], sha256=hashes, origins=origins,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(NAME, len(payloads), 'embedded sources; no model executed')
    return nb, payloads


if __name__ == '__main__':
    build()
