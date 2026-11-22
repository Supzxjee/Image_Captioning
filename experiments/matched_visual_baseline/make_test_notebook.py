"""Source-contained TEST notebook using the successful final-model TEST scaffold."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_notebook import once, remove_assignments

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'BASELINE_MATCHED_PARALLEL3_V188_TEST5000_API_2GPU.ipynb'
STATUS = 'BASELINE_MATCHED_PARALLEL3_TEST5000_COMPLETE'


def literal_assignments(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def build():
    parent = ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_SEED43_TEST5000_API_2GPU.ipynb'
    nb = json.loads(parent.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    run = ''.join(nb['cells'][3]['source'])
    values = literal_assignments(setup)
    payloads, origins = {}, {}
    for path, name in values['MAPPING'].items():
        raw = subprocess.check_output(['git', 'show', values['REVISION']+':'+path], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != values['HASHES'][path]:
            raise ValueError('Changed TEST scaffold source: '+path)
        payloads[name] = raw.decode('utf-8')
        origins[name] = dict(commit=values['REVISION'], path=path, sha256=values['HASHES'][path])
    # Extract literal file writes without executing any source or notebook code.
    for node in ast.walk(ast.parse(run)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr=='write_text' and node.args):
            continue
        receiver = node.func.value
        if not (isinstance(receiver, ast.BinOp) and isinstance(receiver.left, ast.Name)
                and receiver.left.id=='SOURCE' and isinstance(receiver.right, ast.Constant)):
            continue
        code = ast.literal_eval(node.args[0])
        name = receiver.right.value
        payloads[name] = code
        origins[name] = dict(parent_notebook=str(parent.relative_to(ROOT)), embedded_literal=True)
    training_nb = json.loads((HERE/'BASELINE_MATCHED_PARALLEL3_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
    training = literal_assignments(''.join(training_nb['cells'][1]['source']))['SOURCE_PAYLOADS']
    training_manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('csa_patch_model.py', 'semantic_patch_model.py', 'memory_fusion.py', 'bank_layers.py', 'visual_only_model.py'):
        payloads[name] = training[name]
        origins[name] = dict(training_notebook=True, training_source_sha256=training_manifest['sha256'][name])
    payloads['baseline_test_data.py'] = (HERE/'baseline_test_data.py').read_text(encoding='utf-8')
    result = json.loads((HERE/'RESULT.json').read_text(encoding='utf-8'))
    payloads['BASELINE_TRAINING_LOCK.json'] = json.dumps(dict(source_version=result['source'],
        best_checkpoint_sha256=result['receipts']['best_receipt.json']['sha256'],
        best_VALID_CIDEr=result['metrics_by_epoch']['9']['CIDEr'],source_sha256=training_manifest['sha256']), indent=2)+'\n'
    payloads['MATCHED_BASELINE_TEST_PROTOCOL.md'] = (HERE/'TEST_PROTOCOL.md').read_text(encoding='utf-8')

    worker = payloads['run_csa_per_caption_test.py']
    start = worker.index('def contracts(')
    end = worker.index('\n\ndef checkpoint_preflight', start)
    worker = worker[:start] + '''def contracts(cfg):
    if not cfg.get('matched_baseline') or cfg.get('semantic'):
        raise ValueError('Expected matched visual-only baseline TEST')
    import baseline_test_data as module
    return module


def checked_weights(cfg):
    import torch
    d=contracts(cfg)
    ck=torch.load(d.checkpoint(cfg['checkpoint_root']),map_location='cpu',weights_only=False)
    d.validate_checkpoint_state(ck)
    return ck,d.runtime_policy(ck['identity']['software']['torch'],str(torch.__version__))
''' + worker[end:]
    a = worker.index('        context=None\n')
    b = worker.index('        model=model.to(device);', a)
    worker = worker[:a] + '''        from visual_only_model import VisualOnlyCaptioner,VisualOnlyContext
        model=VisualOnlyCaptioner(len(words));context=VisualOnlyContext(cfg['stage'])
''' + worker[b:]
    worker = once(worker, "model.load_state_dict(ck['model'],strict=True);del ck;model.eval();rows=[];images=",
                  "model.load_state_dict(ck['model'],strict=True);del ck;model.requires_grad_(False);model.eval();rows=[];beams=[];images=")
    marker = "                for i,s,v in zip(index.tolist(),seq[:,0].cpu().tolist(),scores[:,0].cpu().tolist()):"
    worker = once(worker, marker, '''                for i,seqs,values in zip(index.tolist(),seq.cpu().tolist(),scores.cpu().tolist()):
                    beams.append(dict(coco_id=int(data['ids'][i]),candidates=[dict(ids=ids,caption=text_from_ids(ids,words),logprob=float(value)) for ids,value in zip(seqs,values)]))
''' + marker)
    worker = once(worker, "            save(out/f'captions_rank{rank}.json',rows)",
                  "            save(out/f'captions_rank{rank}.json',rows)\n            save(out/f'beams_rank{rank}.json',beams)")
    worker = once(worker, "caption_sha256=sha(out/f'captions_rank{rank}.json'),runtime_policy=policy",
                  "caption_sha256=sha(out/f'captions_rank{rank}.json'),beams_sha256=sha(out/f'beams_rank{rank}.json'),runtime_policy=policy")
    worker = once(worker, "status='CSA_PER_CAPTION_SEED_TEST5000_COMPLETE' if cfg.get('semantic') else 'CLIP_PATCH_199_TEST5000_COMPLETE'",
                  "status='"+STATUS+"'")
    payloads['run_csa_per_caption_test.py'] = worker
    origins['run_csa_per_caption_test.py']['matched_baseline_override'] = True
    hashes = {name: hashlib.sha256(code.encode('utf-8')).hexdigest() for name, code in payloads.items()}
    for name, code in payloads.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)

    setup = remove_assignments(setup, {'MAPPING', 'HASHES', 'TRAIN_REVISION', 'TRAIN_PINS'})
    setup = once(setup, "PROMPT_TEST_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-test-cache-v181')\n", '')
    setup = once(setup, 'for p in (VISUAL_CACHE,COCO_JSON,PROMPT_TEST_ROOT):', 'for p in (VISUAL_CACHE,COCO_JSON):')
    setup = once(setup, "cfg=dict(semantic=True,prompt_test_root=str(PROMPT_TEST_ROOT),", "cfg=dict(matched_baseline=True,semantic=False,")
    setup = once(setup, "OUTPUT=Path('/kaggle/working/csa_per_caption_seed43_test5000')", "OUTPUT=Path('/kaggle/working/baseline_matched_parallel3_v188_test5000')")
    setup = once(setup, 'SEED=43', 'SEED=42')
    setup = once(setup, "SOURCE_VERSION='ducanh3/notebook7bc83c4c76/179'", "SOURCE_VERSION='ducanh3/notebook7bc83c4c76/188'")
    setup += '\nSOURCE_PAYLOADS = '+repr(payloads)+'\nSOURCE_SHA256 = '+repr(hashes)+'\n'
    start = run.index('    for path,name in MAPPING.items():')
    end = run.index("    run([sys.executable,'-u',SOURCE/'test_test199.py']", start)
    run = run[:start] + '''    preserved=OUTPUT/'source';preserved.mkdir()
    for name,code in SOURCE_PAYLOADS.items():
        if Path(name).name!=name:raise ValueError('Invalid source file')
        raw=code.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed embedded source: '+name)
        (SOURCE/name).write_bytes(raw);(preserved/name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(parent_test_commit=REVISION,sha256=SOURCE_SHA256),indent=2))
''' + run[end:]
    run = once(run, "    run([sys.executable,'-u',SOURCE/'test_semantic_test.py'],120)\n", '')
    run = once(run, 'from csa_per_caption_test_data import checkpoint,VERSION,CHECKPOINT_SHA,prompt_inputs',
               'from baseline_test_data import checkpoint,VERSION,CHECKPOINT_SHA')
    run = once(run, '    prompt_inputs(cfg) # exact TEST181 seal/scope check before expensive setup/stage\n', '')
    run = once(run, '        TOKENS.append(token)', "        if token.startswith('sk-'):raise RuntimeError('Use a Kaggle API token, not a language-model API key.')\n        TOKENS.append(token)")
    run = run.replace('CSA_PER_CAPTION_SEED_TEST5000_COMPLETE', STATUS)
    run = once(run, "('.json','.jsonl','.log','.txt')", "('.json','.jsonl','.log','.txt','.py','.md')")
    intro = '''# Baseline cùng decoder — TEST5000, exact v188 best9

**GPU T4 ×2, Internet ON, Save & Run All.** Chỉ gắn `ducanh2403/visual-cache` và `vuthetam/mscoco-2014`. Không cần cache bank/prompt/detector hoặc Dataset checkpoint mới.

Bật Secret **codex** hoặc **KAGGLE_API_TOKEN** có quyền đọc output `ducanh3/notebook7bc83c4c76/188`. Để CHECKPOINT_ROOT trống: API chỉ tải best.bin version188, kiểm SHA1eade1fc… trước nạp. Nếu đã gắn đúng file, có thể điền CHECKPOINT_ROOT; vẫn kiểm SHA. Không gửi token vào chat hoặc sửa thành tên biến chưa định nghĩa.

Khóa best epoch9/seed42/cursor159399update, cùng decoder parallel3lớp/dropout0.2 đã train. Chỉ inference FP32 haiGPU, beam5/max20/rawlogprob/no length normalization, không training/SCST/rerank/tuning. Cohort5000TEST và scorer giống full model đã báo cáo. TEST references dùng chấm điểm và NLL sau sinh câu, không chọn caption. Chưa có kết quả của lượt này.

Nguồn đã nhúng sẵn và kiểm SHA; stage chỉ5000TESTpatch≈1.4GiB từ H5, không chép fullTRAINcache/không chạy CLIP lại. Lưu top1 và cả5beam/ảnh trước CPUmetrics. Chấm đủ BLEU1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i (+Recall,NLL); từngphase lưu riêng để phục hồi CPU nếu lỗi, không tự sinh lại câu.

Cap2h gồm setup/scorers, không phải ETA hoặc giờ quota. Thiếu quyềnAPI/CUDA/metric thì giữ log và output, không tự đổi checkpoint/cài lại Torch. Nếu scorer lỗi, giữ caption để chấm lại bằng CPU sau. Gửi `baseline_matched_parallel3_v188_test5000_review.zip`; giữ toàn bộ output. Local chỉ static/mock; API thực, strict binary load, CUDA và Java/scorers chờ lượt của bạn. Một seed; dự án đã từng dùng TEST trong phát triển, không coi đây là TEST chưa từng thấy của toàn bộ dự án.
'''
    for index, source in ((0, intro), (1, setup), (3, run)):
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']), f'cell{index}', 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    path = HERE/NAME
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    (HERE/'TEST_SOURCE_MANIFEST.json').write_text(json.dumps(dict(
        parent_notebook=str(parent.relative_to(ROOT)), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source_version=result['source'], checkpoint_sha256=result['receipts']['best_receipt.json']['sha256'],
        sha256=hashes, origins=origins, notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2)+'\n', encoding='utf-8')
    print(NAME, len(payloads), 'embedded files; no model or API executed')
    return nb, payloads


if __name__=='__main__':
    build()
