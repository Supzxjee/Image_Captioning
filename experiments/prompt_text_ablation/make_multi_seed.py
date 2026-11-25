"""Clone the completed seed42 zero-text recipe; no model or Kaggle execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb'


def literals(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected source anchor: '+old[:100])
    return source.replace(old, new, 1)


def replace_assignments(source, changes):
    lines = source.splitlines(keepends=True)
    found = set()
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in changes:
                if name in found:
                    raise ValueError('Duplicate assignment: '+name)
                found.add(name)
                lines[node.lineno-1:node.end_lineno] = [name+' = '+repr(changes[name])+'\n']
    if found != set(changes):
        raise ValueError('Missing assignments')
    return ''.join(lines)


def build(seed):
    if seed not in (43, 44):
        raise ValueError('Only the two preselected additional seeds')
    nb = json.loads(PARENT.read_text(encoding='utf-8'))
    setup = ''.join(nb['cells'][1]['source'])
    original = literals(setup)
    for name, code in original['SOURCE_PAYLOADS'].items():
        assert hashlib.sha256(code.encode()).hexdigest() == original['SOURCE_SHA256'][name]
    payloads = dict(original['SOURCE_PAYLOADS'])
    payloads['csa_full_runtime.py'] = once(payloads['csa_full_runtime.py'],
        "cfg.get('seed')!=42", f"cfg.get('seed')!={seed}")
    payloads['score_csa_full.py'] = once(payloads['score_csa_full.py'],
        'Compared with historical fullseed42 recipe; direct-H5 versus historical local patch staging.',
        f'Compare with fixed full-prompt seed{seed}; both use the direct-H5 seed scaffold. Training nondeterminism remains.')
    payloads['ZERO_PROMPT_TEXT_MULTI_SEED_PROTOCOL.md'] = (HERE/'MULTI_SEED_PROTOCOL.md').read_text(encoding='utf-8')
    origins = dict(original['SOURCE_ORIGINS'])
    for name in ('csa_full_runtime.py', 'score_csa_full.py', 'ZERO_PROMPT_TEXT_MULTI_SEED_PROTOCOL.md'):
        origins[name] = dict(origins.get(name, {}), multi_seed_override=True,
                             parent_notebook=PARENT.name, seed=seed)
    hashes = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in payloads.items()}
    setup = replace_assignments(setup, dict(SEED=seed, SOURCE_PAYLOADS=payloads,
                                SOURCE_SHA256=hashes, SOURCE_ORIGINS=origins))
    intro = f'''# Zero-text M3 — seed {seed} / 10 epoch CE

**Đã chuẩn bị, chưa chạy. Người dùng báo không đủ quota GPU ngày25/11. Chỉ chạy khi có quota; không đổi lịch học để ép vừa quota.**

Khi chạy: GPU T4 ×2, Internet ON, Save & Run All. Đây là train từ đầu, không dùng checkpoint đã học. Full prompt seed43/44 đã có, không chạy lại Full.

Gắn bốn Dataset cũ:
1. `ducanh2403/visual-cache`
2. `vuthetam/mscoco-2014`
3. `ducanh2403/csa-semantic-banks-v130`
4. `ducanh2403/csa-prompt-full-karpathy-v175`

Giữ full model53.183.736tham số, bank/M2, adapterM3, ROI/mask/hình học. Chỉ đặt object_text và relation_text thành0 trong TRAIN/VALID/inference. Noun/verb bank không đặt0.

Seed{seed}, đủ10epochCE/177110updates, global32, FP32/Accelerate1.15.0/DDP, per-caption/directH5. Best=maxVALIDCIDEr, beam5/max20. Không SCST/rerank/TEST; không chạy lại CLIP/UPT/YOLO hoặc chép full patch cache. Guard khởi tạo/ROI/zero-text/DDP/causal/beam chạy trên Kaggle trước train; local chỉ kiểm tra static/mock.

**WALL_HOURS=11.5 là cap của một phiên kế thừa scaffold, không phải thời gian đủ10epoch hoặc số giờ quota.** Seed42 từng phải resume. Nếu hết cap, giữ toàn bộ output gồm checkpoint.bin,best.bin,identity,receipts,evaluations,source; không chỉ giữ review.zip. Không chạy lại từ đầu. Resume chỉ đúng seed{seed}/zero-text, đủ optimizer/RNG/cursor; cần nguồn Save Version thực tế để khóa API, hoặc mount nguyên output rồi điền RESUME_ROOT và đổi RUN_TAG='run2'. Không dùng checkpoint Full, seed42 hoặc seed khác.

Gửi `full_zero_prompt_text_seed{seed}_run1_review.zip` sau lượt chạy. Chỉ chấm TEST sau khi audit đủ10epoch. Chạy cả hai seed43/44 theo kế hoạch dù kết quả thuận hoặc nghịch; không kỳ vọng chắc chắn sẽ cải thiện.
'''
    nb['cells'][0]['source'] = intro.splitlines(keepends=True)
    nb['cells'][1]['source'] = setup.splitlines(keepends=True)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell{i}', 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    for name, code in payloads.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)
    path = HERE/f'FULL_ZERO_PROMPT_TEXT_SEED{seed}_XE10_2GPU.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    manifest = dict(seed=seed, method='full_zero_prompt_text_xe10_v1', status='READY_UNRUN_QUOTA_INSUFFICIENT',
        parent_notebook=PARENT.name, parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sha256=hashes,
        changed_existing_payloads=[name for name in original['SOURCE_PAYLOADS'] if payloads[name] != original['SOURCE_PAYLOADS'][name]],
        added_payloads=sorted(set(payloads)-set(original['SOURCE_PAYLOADS'])),
        model_inference=False, Kaggle_executed=False, quota_hours=None)
    (HERE/f'SEED{seed}_SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(path.name, len(payloads), 'embedded sources; no model executed')


if __name__ == '__main__':
    for seed in (43, 44):
        build(seed)
