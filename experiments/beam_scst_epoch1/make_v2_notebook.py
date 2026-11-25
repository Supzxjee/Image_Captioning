"""Proposed repair: exact API179 CE seed43, verified before expensive setup/cache."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from make_notebook import HERE, ROOT, literals, once

SEED = 43
VERSION = 'ducanh3/notebook7bc83c4c76/179'
XE_SHA = 'a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'


def embedded(source, filename):
    return next(ast.literal_eval(n.args[0]) for n in ast.walk(ast.parse(source))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text'
        and isinstance(n.func.value,ast.BinOp) and isinstance(n.func.value.right,ast.Constant)
        and n.func.value.right.value==filename)


def build():
    old = HERE/'BEAM_SCST_FULL_EPOCH1_VALID5000_2GPU.ipynb'
    nb = json.loads(old.read_text(encoding='utf-8'))
    values = literals(''.join(nb['cells'][1]['source']))
    payloads = dict(values['SOURCE_PAYLOADS'])
    reference_nb = ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_SEED43_SCST1500_TEST_RERANK_2GPU.ipynb'
    seed_nb = json.loads(reference_nb.read_text(encoding='utf-8'))
    seed_source = ''.join(seed_nb['cells'][3]['source'])
    payloads['csa_per_caption_test_data.py'] = embedded(seed_source,'csa_per_caption_test_data.py')
    payloads['checkpoint_input.py'] = (HERE/'checkpoint_input.py').read_text(encoding='utf-8')
    payloads['PROTOCOL.md'] = (HERE/'PROTOCOL.md').read_text(encoding='utf-8')
    trainer = payloads['run_beam_epoch.py']
    trainer = once(trainer, "SOURCE_SHA = 'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'", f'SOURCE_SHA = {XE_SHA!r}')
    if trainer.count('4242+rank') != 3:
        raise ValueError('Unexpected policy RNG anchors')
    trainer = trainer.replace('4242+rank','4342+rank')
    trainer = once(trainer,'seed=4242, order=', 'seed=4342, XE_seed=43, order=')
    trainer = once(trainer,'Need original CE best9, not an SCST checkpoint','Need seed43/v179 CE best8, not an SCST checkpoint')
    payloads['run_beam_epoch.py'] = trainer
    routing = literals(seed_source)['ROUTING_PREFLIGHT']
    rows = json.loads((ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_XE10_THREE_SEEDS_VALID_RESULT.json').read_text(encoding='utf-8'))['rows']
    row = next(r for r in rows if r['seed']==SEED)
    assert row['best_epoch']==8 and row['checkpoint_sha256']==XE_SHA
    reference = {k:row[k]/100 for k in ('CIDEr','BLEU_1','BLEU_4','ROUGE_L','CHAIRs','CHAIRi','Recall')}
    reference['NLL'] = row['NLL']
    hashes = {name:hashlib.sha256(code.encode()).hexdigest() for name,code in payloads.items()}
    header = ''.join(nb['cells'][1]['source']).split('\nSOURCE_PAYLOADS=',1)[0]
    header = once(header,"CHECKPOINT_ROOT='/kaggle/input/datasets/ducanh3/upt-coco-full-cache'", "CHECKPOINT_ROOT='' # Blank: exact API179. Optional mounted file/root must match source SHA.")
    header = once(header,"OUTPUT=Path(f'/kaggle/working/beam_scst_epoch1_seed42_{RUN_TAG}')", "OUTPUT=Path(f'/kaggle/working/beam_scst_epoch1_seed43_v179_v2_{RUN_TAG}')")
    header = once(header,'for path in (Path(CHECKPOINT_ROOT),Path(CACHE_ROOT),COCO_JSON):', 'for path in (Path(CACHE_ROOT),COCO_JSON):')
    header = once(header,'DEPS=SOURCE/\'deps\';CHAIR=', "CLI=SOURCE/'cli';TOKENS=[]\nDEPS=SOURCE/'deps';CHAIR=")
    header = once(header,'def redact(line):return line # No secrets or authentication in this notebook.',
        "def redact(line):\n    for token in TOKENS:line=line.replace(token,'[REDACTED]')\n    return line")
    header += f'\nSEED={SEED}\nSOURCE_VERSION={VERSION!r}\nCHECKPOINT_SHA={XE_SHA!r}\nSECRET_NAMES=(\'codex\',\'KAGGLE_API_TOKEN\')\n'
    header += '\nSOURCE_PAYLOADS='+repr(payloads)+'\nSOURCE_SHA256='+repr(hashes)+'\nREFERENCE_VALID='+repr(reference)+'\nROUTING_PREFLIGHT='+repr(routing)+'\n'
    acquisition = '''    sys.path.insert(0,str(SOURCE))
    from checkpoint_input import resolve_mounted
    def checkpoint(root):
        return resolve_mounted(root,CHECKPOINT_SHA,OUTPUT/'checkpoint_input_report.json')
    if CHECKPOINT_ROOT:
        weights=checkpoint(CHECKPOINT_ROOT);origin='mounted exact source bytes'
    else:
        from kaggle_secrets import UserSecretsClient
        token=None
        for secret_name in SECRET_NAMES:
            try:value=UserSecretsClient().get_secret(secret_name).strip()
            except Exception:continue
            if value:token=value;break
        if not token:raise RuntimeError('Enable Secret codex or KAGGLE_API_TOKEN with access to ducanh3/notebook7bc83c4c76/version179. No cache/model run yet.')
        TOKENS.append(token)
        run([sys.executable,'-m','pip','install','-q','--target',CLI,'https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],600)
        api_env=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):api_env.pop(key,None)
        api_env.update(PYTHONPATH=str(CLI),KAGGLE_API_TOKEN=token,PYTHONUNBUFFERED='1')
        run([sys.executable,'-c','SOURCE_VERSION='+repr(SOURCE_VERSION)+'\\n'+ROUTING_PREFLIGHT],120,api_env)
        target=OUTPUT/'input_source';target.mkdir()
        run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',SOURCE_VERSION,
             '-p',str(target),'--file-pattern','(^|/)best[.]bin$'],1200,api_env)
        weights=checkpoint(target);origin='exact Save Version API download'
    cfg['checkpoint_root']=str(weights);write_config()
    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(seed=SEED,version=SOURCE_VERSION,
        epoch=8,sha256=CHECKPOINT_SHA,path=str(weights),origin=origin),indent=2))
'''
    runner = ''.join(nb['cells'][3]['source'])
    runner = once(runner,"    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)",
        acquisition+"    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)")
    runner = once(runner,'    prepare(cfg);write_config()',
        "    prepare(cfg)\n    from csa_per_caption_test_data import bind_verified_cache\n    bind_verified_cache(cfg,'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef')\n    write_config()")
    runner = once(runner,"(OUTPUT/'failure.txt').write_text(traceback.format_exc())", "(OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()))")
    runner = once(runner,"    print('LATEST:',OUTPUT/'checkpoint.bin','ENDPOINT:',OUTPUT/'pilot_checkpoint.bin',flush=True)",
        "    for name in ('checkpoint.bin','pilot_checkpoint.bin'):\n        if (OUTPUT/name).is_file():print('SAVED WEIGHTS:',OUTPUT/name,flush=True)\n    if not (OUTPUT/'checkpoint.bin').is_file():print('No beam-SCST training checkpoint saved yet.',flush=True)")
    intro = '''# Beam-mean SCST V2 — seed43 / CE source v179 / một epoch TRAIN

Thay nguồn seed42 chưa tìm được bằng **Full seed43, best CE epoch8**, đúng notebook `ducanh3/notebook7bc83c4c76/version179`, SHA a51e7ec2…; mốc VALID của seed43 được thay đồng bộ. Đây là lựa chọn nguồn có provenance rõ sau lỗi đầu vào, không chọn theo TEST. Giữ một epoch113287ảnh/3541updates/global32,beam5/trainmode/meanbaseline/pureSCST/FP32/DDP.

**Không cần Dataset upt-coco-full-cache.** Chỉ gắn `ducanh3/csa-scst-full-karpathy-cache-v1` và `vuthetam/mscoco-2014`. Bật Internet/GPU T4×2 và Kaggle Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền đọc nguồn ducanh3/version179. Không dán token vào cell/chat. Blank CHECKPOINT_ROOT tải đúng best.bin bằng API; nếu đã có file, điền path file/root ở cell cấu hình, tên file có thể đổi nhưng SHA phải đúng.

Kiểm tra nguồn diễn ra **trước pip scorer, cache staging, CUDA/model**. Không tìm được/không có quyền API sẽ dừng ngay; không dùng latest hoặc checkpoint khác. File nguồn tải về được giữ ở OUTPUT/input_source/best.bin; reviewZIP không chứa file này.

Giữ nguyên loss/beam/model/optimizer/global32/local4×accum4 và cache82/83/84; cache frozen từ seed42 được xác minh nguyên vẹn rồi chỉ rebind metadata VALID sang checkpoint43 theo helper đã chạy trước đây. Không tái sinh hay đổi feature arrays. Endpoint VALID5000, không TEST/rerank/epoch2.

Cap11.5h/phiên không phải ETA/quota. Preflight6updates rồi bỏ đo cả3shard/tail7; train khởi tạo mới từCE43. Lỗi trước đây có0updates, không cần resume nó. Nếu phiên mới bị ngắt sau train, giữ toàn bộOUTPUT để resume đúng recipe; reviewZIP không chứa trọng số. Actual API authorization/CUDA/throughput/quality của bản mới chưa được agent chạy.
'''
    for i,source in ((0,intro),(1,header),(3,runner)):
        nb['cells'][i]['source'] = source.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            compile(''.join(c['source']),f'cell{i}','exec');c['execution_count']=None;c['outputs']=[]
    for name,code in payloads.items():
        if name.endswith('.py'):ast.parse(code,filename=name)
    path = HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest=dict(parent_notebook=old.name,parent_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        reference_notebook=str(reference_nb.relative_to(ROOT)),reference_sha256=hashlib.sha256(reference_nb.read_bytes()).hexdigest(),
        seed=SEED,checkpoint_version=VERSION,checkpoint_sha256=XE_SHA,best_epoch=8,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sha256=hashes,
        reference_VALID=reference,actual_API_or_model_executed=False)
    (HERE/'V2_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(path.name,len(payloads),'embedded files; no API/model execution')


if __name__=='__main__':
    build()
