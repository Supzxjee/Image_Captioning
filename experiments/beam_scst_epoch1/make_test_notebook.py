"""Freeze the proven inference/rerank scaffold on beam SCST endpoint199."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT/'experiments/clip_patch_baseline'))
from make_seed_rerank_notebooks import code, embedded
from make_seed_scst1500_notebooks import replace_functions
from make_notebook import literals, once

SHA='3b03acc29f025b07dc358846b30618ae9c83f8634bc12bf7905e9d908b971f24'
XE_SHA='a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'
VERSION='ducanh3/notebook7bc83c4c76/199'
NAME='BEAM_SCST_V199_TEST5000_RERANK_API_2GPU.ipynb'


def build():
    base=ROOT/'experiments/clip_patch_baseline'
    prior=json.loads((base/'CLIP_PATCH_CSA_SCST_ONLY1500_RERANK_TEST5000_2GPU.ipynb').read_text(encoding='utf-8'))
    seed=json.loads((base/'CSA_PER_CAPTION_SEED43_RERANK_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))
    v2=json.loads((HERE/'BEAM_SCST_FULL_EPOCH1_SEED43_V179_V2_2GPU.ipynb').read_text(encoding='utf-8'))
    payloads=dict(literals(code(v2,1))['SOURCE_PAYLOADS'])
    for n in ('beam_policy.py','full_epoch_data.py','run_beam_epoch.py','scst_cache.py','scst_dataset_mount.py','captioning/scst.py','CACHE_PARTS.json','PROTOCOL.md'):
        del payloads[n]
    data=embedded(code(prior,3),'csa_per_caption_test_data.py')
    data=data.replace('Fixed best9 TEST prompt/patch contracts','Fixed beam SCST v199 TEST prompt/patch contracts')
    data=data.replace("CHECKPOINT_SHA='80e1280b9eef216c5db421a738c7a5e0550bb34b1a62a2a13b2c8e76d40b1a24'",f'CHECKPOINT_SHA={SHA!r}')
    data=data.replace("XE_SOURCE_SHA='b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'",f'XE_SOURCE_SHA={XE_SHA!r}')
    data=data.replace("VERSION='csa_scst_only1500_rerank_valid5000'",f'VERSION={VERSION!r}')
    data=data.replace("PARENT_SHA='56a90b2223b3bb64e433a342d048389dc3e80693fa574c2c1543e31e39a4377b'\n",'')
    data=data.replace("value['epoch']!=9", "value['epoch']!=1").replace("value['updates']!=1500", "value['updates']!=3541")
    data=data.replace('epoch=9,updates=1500','epoch=1,updates=3541')
    data=data.replace('Wrong frozen SCST1500 TEST lock','Wrong frozen beam SCST199 TEST lock')
    data=replace_functions(data,{
        'checkpoint':'''def checkpoint(root):
    from checkpoint_input import resolve_mounted
    return resolve_mounted(root,CHECKPOINT_SHA)
''',
        'validate_checkpoint_state':'''def validate_checkpoint_state(ck):
    if ck['updates']!=3541 or ck['source_sha256']!=XE_SOURCE_SHA:
        raise ValueError('Wrong beam SCST199 endpoint lineage')
    if not {'core.nouns','core.verbs'}<=set(ck['model']):
        raise ValueError('Missing frozen TRAIN banks')
    return lock()
'''})
    payloads['csa_per_caption_test_data.py']=data
    payloads['run_csa_per_caption_test.py']=embedded(code(prior,3),'run_csa_per_caption_test.py')
    payloads['csa_per_caption_rerank_data.py']=embedded(code(prior,3),'csa_per_caption_rerank_data.py')
    rerank=embedded(code(prior,3),'run_csa_per_caption_rerank.py')
    # Preserve basic scores/captions even if a later Java scorer fails.
    marker="        meteor = Meteor()"
    rerank=once(rerank,marker,"""        save(out / (policy + '_captions.json'), selected)
        save(out / (policy + '_cider_per_image.json'), [dict(coco_id=int(i), CIDEr=float(v)) for i,v in zip(ref,per)])
        partial=dict(CIDEr=float(c),BLEU_1=float(b[0]),BLEU_4=float(b[3]),ROUGE_L=float(r),**m,counts=counts)
        save(out / (policy + '_basic_chair.json'),partial)
"""+marker)
    rerank=once(rerank,'        spice_score, spice_per = Spice().compute_score(ref, hyp)',"""        save(out / (policy + '_meteor.json'),dict(METEOR=float(meteor_score),per=[float(v) for v in meteor_per]))
        spice_score, spice_per = Spice().compute_score(ref, hyp)""")
    payloads['run_csa_per_caption_rerank.py']=rerank
    revision='277851d0fbc4643ae7001bc7702dd0ce5c91d0f6'
    for origin in ('captioning/object_consistency.py','captioning/multiscore_reranking.py','experiments/csa_paper/RERANK_TEST_INPUTS.json'):
        raw=subprocess.check_output(['git','show',revision+':'+origin],cwd=ROOT)
        name=origin if origin.startswith('captioning/') else Path(origin).name
        payloads[name]=raw.decode('utf-8')
    payloads['REPLAY_RECEIPTS.json']=json.dumps(dict(checkpoint_sha256=SHA))+'\n'
    payloads['TEST_PROTOCOL.md']=(HERE/'TEST_PROTOCOL.md').read_text(encoding='utf-8')
    reference=json.loads((base/'CSA_PER_CAPTION_SEED43_RERANK_TEST5000_RESULT.json').read_text(encoding='utf-8'))
    keys=('BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall','NLL')
    baseline={p:{k:m[k] for k in keys} for p,m in reference['result']['metrics'].items()}
    selector=reference['result']['selector']
    ref_sha=reference['rerank_lock']['references_sha256']
    helper=payloads['csa_per_caption_rerank_data.py']
    anchor="    source = read(test/'test_lock.json')"
    helper=once(helper,anchor,anchor+f"\n    if sha(test/'test_references.json')!={ref_sha!r}:raise ValueError('TEST references differ from same-seed CE')")
    payloads['csa_per_caption_rerank_data.py']=helper
    header=code(seed,1)
    tree=ast.parse(header)
    remove={'REVISION','MAPPING','HASHES','TRAIN_REVISION','TRAIN_PINS','TEST_OUTPUT_VERSION'}
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None) in remove)]
    header=ast.unparse(tree)+'\n'
    header=once(header,"SOURCE_VERSION = 'ducanh3/notebook7bc83c4c76/179'",f'SOURCE_VERSION = {VERSION!r}')
    header=once(header,"OUTPUT = Path('/kaggle/working/csa_per_caption_seed43_rerank_test5000')","OUTPUT = Path('/kaggle/working/beam_scst_v199_test5000_rerank_v1')")
    header+=f'\nSOURCE_PAYLOADS={payloads!r}\nSOURCE_SHA256={ {n:hashlib.sha256(s.encode()).hexdigest() for n,s in payloads.items()}!r}\nCE_REFERENCE={baseline!r}\nFROZEN_SELECTOR={selector!r}\n'
    original=code(seed,3);route=literals(original)['ROUTING_PREFLIGHT']
    # Source acquisition already worked for179; only endpoint pattern/version changes.
    acquisition=code(v2,3)
    acquisition=acquisition[acquisition.index('    sys.path.insert(0,str(SOURCE))'):acquisition.index("    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py'")]
    acquisition=acquisition.replace('CHECKPOINT_SHA', 'ENDPOINT_SHA')
    acquisition=acquisition.replace("'(^|/)best[.]bin$'","'(^|/)beam_scst_epoch1_seed43_v179_v2_run1/pilot_checkpoint[.]bin$'")
    acquisition=acquisition.replace('version179','version199').replace('epoch=8','epoch=1')
    header+=f'ENDPOINT_SHA={SHA!r}\n'
    original_tail=code(prior,3)
    middle=original_tail[original_tail.index("    prompt_inputs(cfg)"):original_tail.index("    for path,digest in {'captioning/object_consistency.py'")]
    middle=middle.replace("    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(version=VERSION,kaggle_source=SOURCE_VERSION,sha256=CHECKPOINT_SHA,path=str(weights),origin=origin),indent=2))\n",'')
    rtail=original_tail[original_tail.index("    rerank_out=OUTPUT/'rerank'"):original_tail.index("    xe9=")]
    rtail=rtail.replace("    run([sys.executable,'-u',script,'score',rerank_config],3600)","    run([sys.executable,'-u',script,'score',rerank_config],5400)")
    suffix='''    workers=[json.loads((OUTPUT/f'worker{rank}.json').read_text()) for rank in (0,1)]
    nll=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    if result['selector']!=FROZEN_SELECTOR:raise ValueError('Changed fixed selector')
    for metrics in result['metrics'].values():metrics['NLL']=nll
    comparison=dict(seed=43,split='test',images=5000,checkpoint_source=SOURCE_VERSION,
        checkpoint_sha256=ENDPOINT_SHA,CE_source='ducanh3/notebook7bc83c4c76/179',
        CE_metrics=CE_REFERENCE,SCST_metrics=result['metrics'],
        delta_vs_same_seed_CE={p:{k:result['metrics'][p][k]-v for k,v in m.items()} for p,m in CE_REFERENCE.items()},
        test_tuning=False,prior_project_test_exposure=True)
    result.update(status='BEAM_SCST_V199_TEST5000_RERANK_COMPLETE',seed=43,
        checkpoint_source=SOURCE_VERSION,updates=3541,epochs_SCST=1,CE_source='ducanh3/notebook7bc83c4c76/179',
        NLL_note='Same frozen captioner NLL for both policies; no target-conditioned selection.')
    (OUTPUT/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False))
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (rerank_out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(comparison,indent=2),flush=True);complete=True
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_tuning=False,
        source_version=SOURCE_VERSION,seed=43,test_used=True,reranking=True)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt','.py','.md'):
                archive.write(path,str(path.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('TEST + FIXED RERANK COMPLETE' if complete else 'INCOMPLETE; preserve candidates/scores; no re-decode/retraining.',flush=True)
'''
    runner='ROUTING_PREFLIGHT='+repr(route)+'''\ncomplete=False
try:
    preserved=OUTPUT/'source';preserved.mkdir()
    for name,source in SOURCE_PAYLOADS.items():
        raw=source.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed source '+name)
        for root in (SOURCE,preserved):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(sha256=SOURCE_SHA256),indent=2))
'''+acquisition+'''    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    from csa_per_caption_test_data import checkpoint,VERSION,CHECKPOINT_SHA,prompt_inputs
'''+middle+rtail+suffix
    nb=copy.deepcopy(prior)
    nb['cells'][0]['source']=['''# Beam SCST v199 · TEST5000 + fixed rerank

Tải đúng `ducanh3/notebook7bc83c4c76/version199`, endpoint `pilot_checkpoint.bin` SHA3b03acc2…, seed43,3541updates. Không huấn luyện. Sinh beam5 một lần và chấm **top1 + rerank cố định0.3decoder/0.4CLIP/0.3object**, đủ BLEU1/BLEU4/METEOR/ROUGE-L/CIDEr/SPICE/CHAIRs/CHAIRi + Recall/NLL. So với CE179 cùng seed43; không chọn trọng số theo TEST.

GPU **T4×2**, Internet, Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền nguồn ducanh3. Gắn4Dataset: `ducanh2403/visual-cache`, `vuthetam/mscoco-2014`, `ducanh2403/csa-prompt-test-cache-v181`, `ducanh2403/objectdetectionecache`. Blank CHECKPOINT_ROOT tải bằng API; không cần Dataset checkpoint/fullSCSTcache. Các path nằm ở cell cấu hình.

Cap3h là giới hạn dừng, không phải ETA hay quota. Giữ toàn bộ Output; gửi reviewZIP. Caption/beam/CLIPscores/selectedranks và kết quả cơ bản được lưu trước Java metrics để có thể chấm CPU lại nếu lỗi. Hai chính sách đều báo cáo dù rerank bất lợi; không có tự động học lại hay đổi checkpoint. Đây là TEST đã được dự án sử dụng trước đây. Agent chỉ kiểm tra static/mock, chưa chạy API/CUDA/scorer.
''']
    nb['cells'][1]['source']=header.splitlines(keepends=True)
    nb['cells'][3]['source']=runner.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec');c['execution_count']=None;c['outputs']=[]
    for n,s in payloads.items():
        if n.endswith('.py'):compile(s,n,'exec')
    target=HERE/NAME;target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest=dict(source_version=VERSION,endpoint_sha256=SHA,XE_source_sha256=XE_SHA,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in payloads.items()},
        CE_reference=baseline,selector=selector,API_or_model_executed=False)
    (HERE/'TEST_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(NAME,len(payloads),'embedded files; unrun')


if __name__=='__main__':build()
