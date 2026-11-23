"""Frozen zero-text TEST from the successful full-model TEST scaffold."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_notebook import once

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='FULL_ZERO_PROMPT_TEXT_V193_TEST5000_API_2GPU.ipynb'
STATUS='FULL_ZERO_PROMPT_TEXT_TEST5000_COMPLETE'
METHOD='full_zero_prompt_text_xe10_v1'


def literals(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            try:result[node.targets[0].id]=ast.literal_eval(node.value)
            except (TypeError,ValueError):pass
    return result


def build():
    parent=ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_SEED43_TEST5000_API_2GPU.ipynb'
    nb=json.loads(parent.read_text(encoding='utf-8'));setup=''.join(nb['cells'][1]['source']);run=''.join(nb['cells'][3]['source'])
    values=literals(setup);payloads={};origins={}
    for path,name in values['MAPPING'].items():
        raw=subprocess.check_output(['git','show',values['REVISION']+':'+path],cwd=ROOT)
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=values['HASHES'][path]:raise ValueError('Changed pinned TEST source: '+path)
        payloads[name]=raw.decode('utf-8');origins[name]=dict(commit=values['REVISION'],path=path,sha256=digest)
    for node in ast.walk(ast.parse(run)):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args):continue
        receiver=node.func.value
        if not (isinstance(receiver,ast.BinOp) and isinstance(receiver.left,ast.Name) and receiver.left.id=='SOURCE' and isinstance(receiver.right,ast.Constant)):continue
        name=receiver.right.value;payloads[name]=ast.literal_eval(node.args[0]);origins[name]=dict(parent_embedded=True)
    training_nb=json.loads((HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
    training=literals(''.join(training_nb['cells'][1]['source']))['SOURCE_PAYLOADS']
    train_manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('csa_patch_model.py','semantic_patch_model.py','memory_fusion.py','bank_layers.py','zero_prompt_model.py'):
        payloads[name]=training[name];origins[name]=dict(training_source_sha256=train_manifest['sha256'][name])
    result=json.loads((HERE/'FULL_XE10_RESULT.json').read_text(encoding='utf-8'))
    weight_sha=result['receipts']['best_receipt.json']['sha256'];version=result['source_version']
    payloads['ZERO_TEXT_TRAINING_LOCK.json']=json.dumps(dict(source_version=version,checkpoint_sha256=weight_sha,
        best_VALID_CIDEr=result['metrics_by_epoch']['8']['CIDEr'],source_sha256=train_manifest['sha256']),indent=2)+'\n'
    payloads['ZERO_TEXT_TEST_PROTOCOL.md']=(HERE/'TEST_PROTOCOL.md').read_text(encoding='utf-8')
    data=payloads['csa_per_caption_test_data.py']
    data=once(data,"CHECKPOINT_SHA='a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'","CHECKPOINT_SHA="+repr(weight_sha))
    data=once(data,"VERSION='ducanh3/notebook7bc83c4c76/179'","VERSION="+repr(version))
    data=data.replace('csa_structure_patch_prompt_grouped_per_caption_accelerate_v1',METHOD)
    data=data.replace('seed43','zero-text seed42').replace('SHAa51e7ec2','SHA0aa81a0c')
    data=once(data,"schema='csa_per_caption_seed_test_v1'","schema='zero_prompt_text_test5000_v1'")
    data=once(data,"epoch=8,updates=141688,method='"+METHOD+"')", "epoch=8,updates=141688,method='"+METHOD+"',prompt_text_mode='zero',seed=42)")
    marker='    return value\n'
    data=once(data,marker,"    if value.get('prompt_text_mode')!='zero' or value.get('seed')!=42:raise ValueError('Wrong zero-text TEST mode or seed')\n"+marker)
    data=once(data,"    if i.get('seed')!=43:raise ValueError('Wrong training seed')",
        "    if i.get('seed')!=42 or i.get('prompt_text_mode')!='zero':raise ValueError('Wrong zero-text training seed or mode')\n    sealed=read(HERE/'ZERO_TEXT_TRAINING_LOCK.json')\n    if i['source']['sha256']!=sealed['source_sha256'] or abs(s['best_CIDEr']-sealed['best_VALID_CIDEr'])>1e-12:raise ValueError('Wrong training sources or selected checkpoint')")
    payloads['csa_per_caption_test_data.py']=data
    worker=payloads['run_csa_per_caption_test.py'];a=worker.index('def contracts(');b=worker.index('\n\ndef checkpoint_preflight',a)
    worker=worker[:a]+'''def contracts(cfg):
    if not cfg.get('semantic') or cfg.get('prompt_text_mode')!='zero':
        raise ValueError('Expected frozen zero-text TEST with unchanged ROI/geometry')
    import csa_per_caption_test_data as module
    return module


def checked_weights(cfg):
    import torch
    d=contracts(cfg);ck=torch.load(d.checkpoint(cfg['checkpoint_root']),map_location='cpu',weights_only=False)
    d.validate_checkpoint_state(ck)
    return ck,d.runtime_policy(ck['identity']['software']['torch'],str(torch.__version__))
''' + worker[b:]
    a=worker.index('        context=None\n');b=worker.index('        model=model.to(device);',a)
    worker=worker[:a]+'''        from zero_prompt_model import ZeroTextCaptioner
        from semantic_data import SemanticContext
        model=ZeroTextCaptioner(len(words),ck['model']['core.nouns'],ck['model']['core.verbs'])
        context=SemanticContext(cfg['stage'])
''' + worker[b:]
    worker=once(worker,"model.load_state_dict(ck['model'],strict=True);del ck;model.eval();rows=[];images=",
                "model.load_state_dict(ck['model'],strict=True);del ck;model.requires_grad_(False);model.eval();rows=[];beams=[];images=")
    worker=once(worker,'                if offset==0:check_paths(model,memory,seq,scores)', '''                if offset==0:
                    from zero_prompt_model import zero_prompt_text
                    prompts=context.images(index,device);zeroed=zero_prompt_text(prompts)
                    if any(torch.count_nonzero(zeroed[k]).item() for k in ('object_text','relation_text')):raise ValueError('Nonzero text in zero-text TEST')
                    if any(zeroed[k] is not v for k,v in prompts.items() if k not in ('object_text','relation_text')):raise ValueError('Zeroing changed ROI/geometry/masks')
                    changed=dict(prompts)
                    for key in ('object_text','relation_text'):changed[key]=torch.full_like(prompts[key],float('nan'))
                    perturbed=model.encode(x,changed)
                    torch.testing.assert_close(memory,perturbed,rtol=1e-4,atol=5e-5)
                    save(out/f'zero_text_inference_rank{rank}.json',dict(passed=True,images=len(index),prompt_text_mode='zero',
                        changed_text_invariant=True,ROI_geometry_masks_unchanged=True,
                        max_memory_difference=float((memory-perturbed).abs().max()),
                        captioner_parameters=sum(p.numel() for p in model.parameters()),training=False))
                    check_paths(model,memory,seq,scores)''')
    marker="                for i,s,v in zip(index.tolist(),seq[:,0].cpu().tolist(),scores[:,0].cpu().tolist()):"
    worker=once(worker,marker,'''                for i,seqs,values in zip(index.tolist(),seq.cpu().tolist(),scores.cpu().tolist()):
                    beams.append(dict(coco_id=int(data['ids'][i]),candidates=[dict(ids=ids,caption=text_from_ids(ids,words),logprob=float(value)) for ids,value in zip(seqs,values)]))
'''+marker)
    worker=once(worker,"            save(out/f'captions_rank{rank}.json',rows)","            save(out/f'captions_rank{rank}.json',rows)\n            save(out/f'beams_rank{rank}.json',beams)")
    worker=once(worker,"caption_sha256=sha(out/f'captions_rank{rank}.json'),runtime_policy=policy", "caption_sha256=sha(out/f'captions_rank{rank}.json'),beams_sha256=sha(out/f'beams_rank{rank}.json'),prompt_text_mode='zero',zero_guard_sha256=sha(out/f'zero_text_inference_rank{rank}.json'),runtime_policy=policy")
    worker=once(worker,"status='CSA_PER_CAPTION_SEED_TEST5000_COMPLETE' if cfg.get('semantic') else 'CLIP_PATCH_199_TEST5000_COMPLETE'", "status='"+STATUS+"',prompt_text_mode='zero',seed=42")
    payloads['run_csa_per_caption_test.py']=worker
    for name in ('csa_per_caption_test_data.py','run_csa_per_caption_test.py'):origins[name]['zero_text_override']=True
    hashes={n:hashlib.sha256(code.encode('utf-8')).hexdigest() for n,code in payloads.items()}
    for name,code in payloads.items():
        if name.endswith('.py'):ast.parse(code,filename=name)
    lines=setup.splitlines(keepends=True)
    for node in reversed(ast.parse(setup).body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('MAPPING','HASHES','TRAIN_REVISION','TRAIN_PINS'):
            del lines[node.lineno-1:node.end_lineno]
    setup=''.join(lines)
    setup=once(setup,'cfg=dict(semantic=True,',"cfg=dict(prompt_text_mode='zero',semantic=True,")
    setup=once(setup,"OUTPUT=Path('/kaggle/working/csa_per_caption_seed43_test5000')","OUTPUT=Path('/kaggle/working/full_zero_prompt_text_v193_test5000')")
    setup=once(setup,'\nSEED=43\n','\nSEED=42\n')
    setup=once(setup,"SOURCE_VERSION='ducanh3/notebook7bc83c4c76/179'","SOURCE_VERSION="+repr(version))
    setup+='\nSOURCE_PAYLOADS='+repr(payloads)+'\nSOURCE_SHA256='+repr(hashes)+'\n'
    a=run.index('    for path,name in MAPPING.items():');b=run.index("    run([sys.executable,'-u',SOURCE/'test_test199.py']",a)
    run=run[:a]+'''    preserved=OUTPUT/'source';preserved.mkdir()
    for name,code in SOURCE_PAYLOADS.items():
        if Path(name).name!=name:raise ValueError('Invalid source filename')
        raw=code.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed embedded source: '+name)
        (SOURCE/name).write_bytes(raw);(preserved/name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(parent_test_commit=REVISION,sha256=SOURCE_SHA256),indent=2))
''' + run[b:]
    run=once(run,'        TOKENS.append(token)',"        if token.startswith('sk-'):raise RuntimeError('Use a Kaggle API token, not a language-model API key.')\n        TOKENS.append(token)")
    run=run.replace('CSA_PER_CAPTION_SEED_TEST5000_COMPLETE',STATUS)
    run=once(run,'dict(complete=complete,training=False,test_tuning=False,seed=SEED,source_version=SOURCE_VERSION)',"dict(complete=complete,training=False,test_tuning=False,seed=SEED,source_version=SOURCE_VERSION,prompt_text_mode='zero')")
    run=once(run,"('.json','.jsonl','.log','.txt')","('.json','.jsonl','.log','.txt','.py','.md')")
    intro='''# Zero-text M3 — TEST5000, exactv193 best8

**GPU T4 ×2, Internet ON, Save & Run All.** Gắn ba Dataset:
1. `ducanh2403/visual-cache`
2. `vuthetam/mscoco-2014`
3. `ducanh2403/csa-prompt-test-cache-v181`

Bật Secret **codex** hoặc **KAGGLE_API_TOKEN** có quyền đọc `ducanh3/notebook7bc83c4c76/193`. Giữ CHECKPOINT_ROOT='' để API chỉ tải exactbest.bin epoch8/SHA0aa81a0c… . Nếu đã gắn đúng file/root, có thể điền đường dẫn; vẫn kiểm SHA. Không dùngfinal10 hoặc checkpoint đúngchữ.

Giữ ZeroTextCaptioner như khi train: **chữ object_text/relation_text của M3 vẫn0**, còn M1/M2/bankbuffers/ROI/geometry/masks giữ nguyên. TEST181 cần cho ROI và hình học, không phải để bật lại chữ. Guard đầu mỗiGPU thay chữ trên dictcopy không đổi memory rồi kiểm beamrescore. Không train/SCST/rerank/tuning. Bank nằm trong checkpoint, không cần bank130 hoặc prompt175TRAIN Dataset.

Cùng TEST5000/FP32/beam5/max20/rawlogprob/no length normalization và scorers với fullpromptseed42. Lưu top1 và cả5beam/ảnh trướcCPUmetrics. Chấm đủ BLEU1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i (+Recall,NLL). ReferenceTEST chỉ chấm điểm/NLL sau decode, không chọn câu/checkpoint.

Nguồn nhúng và kiểm SHA; stage chỉTEST≈1.4GiB, không chạyCLIP/UPT/YOLO hoặc copyfullTRAINcache. Cap2h gồmsetup/CPUmetrics, không ETA/billing/quota. Nếu scorer lỗi, giữcaption+perphaseoutput để phục hồiCPU, không tựdecode lại. API/CUDA/strictstate/scorer runtime chờ user-run; local chỉ static/mock.

Gửi `full_zero_prompt_text_v193_test5000_review.zip`, giữ entireoutput. Mộtseed và dự án đã dùngTEST trước đây; không gọi là untouchedholdout hoặc formalnoninferiority. Lượt này không mở M2-only hoặc nhiều seed.
'''
    for i,code in ((0,intro),(1,setup),(3,run)):nb['cells'][i]['source']=code.splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell{i}','exec');cell['execution_count']=None;cell['outputs']=[]
    target=HERE/NAME;target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (HERE/'TEST_SOURCE_MANIFEST.json').write_text(json.dumps(dict(source_version=version,checkpoint_sha256=weight_sha,
        parent_notebook=str(parent.relative_to(ROOT)),parent_notebook_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        sha256=hashes,origins=origins,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
    print(NAME,len(payloads),'embedded files; no model or API executed')
    return nb,payloads


if __name__=='__main__':build()
