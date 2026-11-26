"""Bundle a bounded same-session throughput comparison; no model execution."""
import hashlib
import json
from pathlib import Path
from make_notebook import literals,once

HERE=Path(__file__).resolve().parent
NAME='BEAM_SCST_V199_SPEED_COMPARE_2GPU.ipynb'
SHA='5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c'


def build():
    nb=json.loads((HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_text(encoding='utf-8'))
    values=literals(''.join(nb['cells'][1]['source']));payloads=values['SOURCE_PAYLOADS'].copy()
    receipt=json.loads((HERE/'PARENT_CHECKPOINT_RECEIPT.json').read_text())
    assert receipt['sha256']==SHA and receipt['updates']==3541 and receipt['cursor_images']==113287
    assert receipt['identity']['source_hashes']==values['SOURCE_SHA256']
    for name in ('speed_candidate.py','run_speed_probe.py','PARENT_CHECKPOINT_RECEIPT.json','THROUGHPUT_PROTOCOL.md'):
        payloads[name]=(HERE/name).read_text(encoding='utf-8')
    header=''.join(nb['cells'][1]['source']).split('\nSOURCE_PAYLOADS=',1)[0]
    header=once(header,'WALL_HOURS=11.5','WALL_HOURS=1.0')
    header=once(header,'beam_scst_epoch1_seed43_v179_v2_{RUN_TAG}','beam_scst_v199_speed_compare_{RUN_TAG}')
    header=once(header,"SOURCE_VERSION='ducanh3/notebook7bc83c4c76/179'","SOURCE_VERSION='ducanh3/notebook7bc83c4c76/199'")
    header=once(header,"CHECKPOINT_SHA='a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'",f'CHECKPOINT_SHA={SHA!r}')
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in payloads.items()}
    header+='\nSOURCE_PAYLOADS='+repr(payloads)+'\nSOURCE_SHA256='+repr(hashes)+'\nROUTING_PREFLIGHT='+repr(values['ROUTING_PREFLIGHT'])+'\n'
    original=''.join(nb['cells'][3]['source'])
    prefix=original[:original.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'")]
    prefix=once(prefix,"'(^|/)best[.]bin$'","'(^|/)beam_scst_epoch1_seed43_v179_v2_run1/checkpoint[.]bin$'")
    prefix=prefix.replace('version179','version199').replace('epoch=8','epoch=1')
    runner=prefix+'''    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'nltk==3.9.2'],300)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    from full_epoch_data import prepare
    prepare(cfg)
    from csa_per_caption_test_data import bind_verified_cache
    bind_verified_cache(cfg,'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef')
    cfg['probe_seconds']=min(1200,PARENT_DEADLINE-time.monotonic()-120)
    if cfg['probe_seconds']<300:raise TimeoutError('Insufficient time for bounded speed comparison')
    write_config()
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_speed_probe.py',CONFIG],cfg['probe_seconds']+30)
    result=json.loads((OUTPUT/'result.json').read_text())
    if result['status']!='SCST_SPEED_PILOT_COMPLETE':raise ValueError('Incomplete comparison')
    print(json.dumps(result,indent=2),flush=True);complete=True
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,profiling_only=True,
        production_training=False,all_optimizer_updates_discarded=True,test_used=False,valid_evaluated=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log','.py','.md'):
                archive.write(path,str(path.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('PROBE COMPLETE; no production checkpoint' if complete else 'INCOMPLETE; preserve log; no automatic fallback.',flush=True)
'''
    intro='''# Beam SCST v199 — so tốc độ bản cũ và bản tối ưu, T4×2

Lượt ngắn, **không train epoch2**. Tải đúng full `checkpoint.bin` của `ducanh3/notebook7bc83c4c76/version199` (SHA5b8d33fc…), khôi phục model/Adam/RNG giống nhau cho hai nhánh. Giữ beam5/FP32/global32/loss/LR và cache hiện có. Thử gộp chuyển prompt theo batch và giảm đồng bộ kiểm tra finite-logit; không đổi batch/FP16/feature sharing.

Kiểm tra dữ liệu/beam/logprob/loss/gradient/trọng số/RNG trước; nếu không khớp thì dừng. So ABBA trên3shard và tail7; **mọi update đều bỏ**, không dùng kết quả pilot làm checkpoint học tiếp. Có số đo từng phần và thời gian tổng; chưa biết mức tăng tốc trước khi chạy. Không suy ra điểm sẽ tăng đều qua30epoch từ một epoch đã chạy.

Gắn2Dataset `ducanh3/csa-scst-full-karpathy-cache-v1`, `vuthetam/mscoco-2014`. Bật Internet, **GPU T4×2**, Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền notebook của Ngân. Không cần Dataset checkpoint/visual-cache/TEST181. Để CHECKPOINT_ROOT trống để tải bằngAPI. Cap60phút là giới hạn dừng gồmsetup, không phải ETA/quota. Không chép lại fullpatch cache.

Save&RunAll, gửi `beam_scst_v199_speed_compare_run1_review.zip`. Chưa áp dụng tối ưu cho train dài; agent chỉ kiểm tra static/NumPy/mock, chưa chạy GPU/API/mô hình/scorer.
'''
    for i,text in ((0,intro),(1,header),(3,runner)):nb['cells'][i]['source']=text.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec');c['execution_count']=None;c['outputs']=[]
    for name,text in payloads.items():
        if name.endswith('.py'):compile(text,name,'exec')
    path=HERE/NAME;path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    m=dict(notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),full_checkpoint_sha256=SHA,
        checkpoint_version='ducanh3/notebook7bc83c4c76/199',sha256=hashes,actual_models_executed=False)
    (HERE/'SPEED_SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    print(NAME,len(payloads),'embedded files; unrun')


if __name__=='__main__':build()
