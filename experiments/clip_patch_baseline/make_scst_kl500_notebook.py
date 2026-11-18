"""Build one fixed seed43 KL pilot using the successful SCST/VALID scaffolding."""
import ast
import copy
import json

from make_seed_rerank_notebooks import HERE, code, embedded, embedded_node
from make_seed_scst1500_notebooks import load, pilot_source


def kl_trainer(source):
    source = source.replace('UPDATES = 1000', 'UPDATES = 500')
    source = source.replace('import argparse\n', 'import argparse\nimport copy\n', 1)
    source = source.replace('    from scst_pilot_model import action_logits, policy_loss, captions, selected_logps',
        '    from scst_pilot_model import action_logits, policy_loss, captions, selected_logps\n    from scst_kl import reference_kl')
    source = source.replace('        model.to(device).eval()', '''        model.to(device).eval()
        before_reference = torch.cuda.get_rng_state(device)
        reference = copy.deepcopy(model).requires_grad_(False).eval()
        if not torch.equal(before_reference, torch.cuda.get_rng_state(device)):
            raise ValueError('Teacher construction changed sampling RNG')
        reference_versions = tuple(p._version for p in reference.parameters())''', 1)
    source = source.replace("        from scst_cache import load_reward", '''        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if any(p.requires_grad or id(p) in optimizer_ids for p in reference.parameters()):
            raise ValueError('Teacher must be frozen and outside optimizer')
        torch.cuda.reset_peak_memory_stats(device)
        from scst_cache import load_reward''', 1)
    source = source.replace('            loss = rl', '''            if step == 0:
                before_teacher = torch.cuda.get_rng_state(device)
            with torch.no_grad():
                reference_memory = reference.encode(x, p)
                reference_logits = reference.decode(reference_memory, sample[:, :-1])
            if step == 0:
                if not torch.equal(before_teacher, torch.cuda.get_rng_state(device)):
                    raise ValueError('Teacher forward changed sampling RNG')
                torch.testing.assert_close(logits.detach(), reference_logits, rtol=1e-4, atol=5e-5)
            kl = reference_kl(logits, reference_logits, mask)
            if step == 0 and abs(float(kl.detach())) > 1e-5:
                raise ValueError('Initial teacher/current KL differs')
            loss = rl + cfg['kl_beta']*kl''', 1)
    source = source.replace('            loss.backward()', '''            loss.backward()
            if step == 0:
                if any(p.grad is not None for p in reference.parameters()):
                    raise ValueError('Teacher received gradients')
                save(out/f'kl_start_guard_rank{rank}.json',dict(initial_KL=float(kl.detach()),
                    teacher_frozen=True, teacher_outside_optimizer=True, teacher_rng_unchanged=True,
                    policy_replay_passed=True, beta=cfg['kl_beta'], torch_version=str(torch.__version__)))''', 1)
    source = source.replace('                dist.all_reduce(stats)', '''                dist.all_reduce(stats)
                kl_stats = torch.tensor([float(kl.detach()),float(loss.detach())],dtype=torch.float64,device=device)
                dist.all_reduce(kl_stats)''', 1)
    source = source.replace("                    print('SCST pilot', row, flush=True)", '''                    row.update(KL=float(kl_stats[0])/2, loss=float(kl_stats[1])/2, beta=cfg['kl_beta'],
                               rank0_peak_memory_bytes=torch.cuda.max_memory_allocated(device))
                    print('SCST+KL pilot', row, flush=True)''', 1)
    source = source.replace('updates=step, source_sha256=sha(src)',
        "updates=step, source_sha256=sha(src), kl_beta=cfg['kl_beta'], reference_sha256=sha(src)")
    source = source.replace('source_sha256=sha(src)))',
        "source_sha256=sha(src), kl_beta=cfg['kl_beta'], reference_sha256=sha(src)))")
    source = source.replace('    finally:\n        dist.destroy_process_group()', '''        if reference_versions != tuple(p._version for p in reference.parameters()):
            raise ValueError('Teacher parameters mutated')
        if any(p.grad is not None for p in reference.parameters()):
            raise ValueError('Teacher has final gradients')
        save(out/f'kl_end_guard_rank{rank}.json',dict(updates=UPDATES,teacher_versions_unchanged=True,
            teacher_gradients_absent=True,peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
    finally:
        dist.destroy_process_group()''', 1)
    return source


def build():
    original = load('CSA_PER_CAPTION_SEED43_SCST1500_TEST_RERANK_2GPU.ipynb')
    valid_template = load('CLIP_PATCH_CSA_SCST_RESUME1500_RERANK_VALID5000_2GPU.ipynb')
    control = load('CSA_PER_CAPTION_SEED43_SCST1500_TEST_RESULT.json')['checkpoints']['500']
    audit = load('CSA_SEED43_SCST_VALID_REGRESSIONS_RESULT.json')
    nb = copy.deepcopy(original)
    nb['cells'][0]['source'] = ['''# Seed43: SCST + KL500, matched VALID5000 only

Fresh from own best XE179, frozen own-XE teacher, beta1.0,500updates/global8/RNG4342, no CE/object reward.
Compares XE, saved pure-SCST500 from185 and new KL500 with the same cached VALID, beam5 and frozen0.3/0.4/0.3 rerank.
Reports all8metrics and NLL; no TEST, coefficient sweep or automatic extension.

Run on the ducanh3 account with T4 x2, Internet and Kaggle Secret codex or KAGGLE_API_TOKEN.
Attach only csa-scst-full-karpathy-cache-v1, mscoco-2014 and objectdetectionecache.
Exact XE/control weights download by API. Cached patch arrays are linked readonly.
The tensor/CUDA preflight and first-step teacher/on-policy guards must pass before accepting the run.
Keep the entire output. Send csa_seed43_scst_kl500_valid5000_review.zip. Three-hour STOP cap, not an ETA.
''']
    setup = code(original, 1).replace('WALL_HOURS=4', 'WALL_HOURS=3')
    setup = setup.replace('csa_per_caption_seed43_scst1500_test_rerank', 'csa_seed43_scst_kl500_valid5000')
    setup = setup.replace('for p in (VISUAL_CACHE,COCO_JSON,PROMPT_TEST_ROOT):', 'for p in (COCO_JSON,):')
    setup = '\n'.join(line for line in setup.split('\n') if not line.startswith('TEST_OUTPUT_VERSION='))
    setup = setup.replace("CHECKPOINT_ROOT='' # Optional exact best.bin. Blank downloads it through API.",
        "CHECKPOINT_ROOT='' # API-only exact best.bin; no checkpoint mount needed.")
    setup += "\nCONTROL_VERSION='ducanh3/notebook7bc83c4c76/185'\nKL_BETA=1.0\n"
    nb['cells'][1]['source'] = [setup]
    main = code(original, 3)
    prefix = main[:main.index("    (SOURCE/'run_scst_resume1500.py').write_text")]
    tree = ast.parse(prefix + 'except BaseException:\n    raise\n')
    top = next(n for n in tree.body if isinstance(n,ast.Try))
    # Both source versions require the API; remove the inherited mount-only branch.
    top.body = [child for n in top.body for child in (
        n.orelse if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='CHECKPOINT_ROOT' else [n])]
    call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=='write_text' and isinstance(n.func.value,ast.Name) and n.func.value.id=='pilot')
    call.args[0] = ast.Constant(kl_trainer(ast.literal_eval(call.args[0])))
    prefix = ast.unparse(tree).split('\nexcept BaseException:')[0]+'\n'
    helper = embedded(code(valid_template, 3), 'csa_per_caption_rerank_data.py')
    helper = helper.replace("receipt['updates'] != 1500", "receipt['updates'] != replay['updates']")
    rerank_runner = embedded(code(valid_template, 3), 'run_csa_per_caption_rerank.py')
    # The saved framework is unchanged; this pilot supplies only a new regularizer.
    body = f'''    (SOURCE/'scst_kl.py').write_text({(HERE/'scst_kl.py').read_text(encoding='utf-8')!r},encoding='utf-8')
    (SOURCE/'scst_kl_valid_compare.py').write_text({(HERE/'scst_kl_valid_compare.py').read_text(encoding='utf-8')!r},encoding='utf-8')
    run([sys.executable,'-c','SOURCE_VERSION='+repr(CONTROL_VERSION)+'\\n'+ROUTING_PREFLIGHT],120,api_env)
    control_download=SOURCE/'control500';control_download.mkdir()
    run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',CONTROL_VERSION,
        '-p',control_download,'--file-pattern',r'(^|/)update0500/(pilot_checkpoint[.]bin|pilot_checkpoint_receipt[.]json)$'],1200,api_env)
    control_files=list(control_download.rglob('pilot_checkpoint.bin'))
    if len(control_files)!=1:raise ValueError('Need exact185 update0500 checkpoint')
    control_weights=control_files[0]
    control_receipt=json.loads(control_weights.with_name('pilot_checkpoint_receipt.json').read_text())
    if control_receipt!={control!r} or hashlib.file_digest(control_weights.open('rb'),'sha256').hexdigest()!=control_receipt['sha256']:
        raise ValueError('Changed saved SCST500 control')
    (ROOT_OUTPUT/'control_source.json').write_text(json.dumps(dict(source=CONTROL_VERSION,checkpoint=control_receipt)))
    original_predictions=SOURCE/'original_XE';original_predictions.mkdir()
    run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',SOURCE_VERSION,
        '-p',original_predictions,'--file-pattern',r'(^|/)evaluations/epoch_08/val_rank[01][.]json$'],900,api_env)
    original_rows=[]
    for name,digest in {audit['XE_prediction_payload_sha256']!r}.items():
        matches=list(original_predictions.rglob(name))
        if len(matches)!=1 or hashlib.file_digest(matches[0].open('rb'),'sha256').hexdigest()!=digest:
            raise ValueError('Changed original XE VALID caption payload')
        original_rows.extend(json.loads(matches[0].read_text()))
    original_captions={{r['coco_id']:r['caption'] for r in original_rows}}
    (ROOT_OUTPUT/'original_XE_VALID.json').write_text(json.dumps(original_rows))
    for path,digest in {{'captioning/object_consistency.py':'5fbd75b0ff4a35f0475d460a294894009d6007f468e248a0e4b528aaa8fabf73',
        'captioning/multiscore_reranking.py':'d1baa60e244c86b32e721486d7923ead1391c3a2fb744651c257995f2d0bb443',
        'experiments/csa_paper/RERANK_TEST_INPUTS.json':'33395ec35a0803ebf711737f0bdba3b6be8c16244f4c292057658e0fa3feaf46'}}.items():
        target=SOURCE/path if path.startswith('captioning/') else SOURCE/Path(path).name
        download('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/277851d0fbc4643ae7001bc7702dd0ce5c91d0f6/'+path,digest,target)
    (SOURCE/'csa_per_caption_rerank_data.py').write_text({helper!r},encoding='utf-8')
    (SOURCE/'run_csa_per_caption_rerank.py').write_text({rerank_runner!r},encoding='utf-8')
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
        'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2'],300)
    evaluator=SOURCE/'run_csa_per_caption_test.py'
    for mode,cap in (('environment',1200),('smoke',600)):
        run([sys.executable,'-u',evaluator,mode,'--config',CONFIG],cap)
    run([sys.executable,'-u',SOURCE/'scst_kl.py','--output',ROOT_OUTPUT/'kl_tensor_preflight.json'],180)
    bootstrap_smoke="from scst_kl_valid_compare import bootstrap; r=bootstrap({{0:(.2,1,1,2),1:(.2,1,1,2)}},{{0:(.1,0,0,2),1:(.1,0,0,2)}},20); assert all(abs(x-v)<1e-8 for k,v in [('CIDEr_x100',10),('CHAIRs_percentage_points',100),('CHAIRi_percentage_points',50)] for x in r[k]); print('Paired bootstrap arithmetic PASS')"
    run([sys.executable,'-c',bootstrap_smoke],120)
    base_cfg=dict(cfg)
    xe_weights=weights
    xe_receipt=dict(sha256=XE_CHECKPOINT_SHA,source_sha256=XE_CHECKPOINT_SHA,updates=0)

    def evaluate_arm(name, path, receipt, training_torch):
        folder=ROOT_OUTPUT/name;folder.mkdir()
        for file in ('vocabulary.json','stage_receipt.json','test_lock.json','test_references.json','environment.json'):
            shutil.copyfile(ROOT_OUTPUT/file,folder/file)
        (folder/'training_runtime.json').write_text(json.dumps(dict(torch_version=training_torch)))
        (folder/'pilot_checkpoint_receipt.json').write_text(json.dumps(receipt))
        evaluation=dict(base_cfg,output=str(folder),checkpoint_root=str(xe_weights),
            evaluate_pilot=receipt['updates']!=0,eval_update=receipt['updates'],eval_checkpoint=str(path))
        evaluation['worker_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-900)
        if evaluation['worker_seconds']<300:raise TimeoutError('Insufficient VALID inference budget')
        config=SOURCE/(name+'_config.json');config.write_text(json.dumps(evaluation))
        (folder/'protocol.json').write_text(json.dumps(evaluation))
        run([sys.executable,'-u',evaluator,'checkpoint_preflight','--config',config],120)
        run([*launch,evaluator,'worker','--config',config],evaluation['worker_seconds']+30)
        (SOURCE/'REPLAY_RECEIPTS.json').write_text(json.dumps(dict(checkpoint_sha256=receipt['sha256'],
            xe_source_sha256=XE_CHECKPOINT_SHA,updates=receipt['updates'])))
        rerank=folder/'rerank';rerank.mkdir()
        rcfg=dict(output=str(rerank),test_output=str(folder),dataset_json=str(COCO_JSON),
            detections=str(DETECTIONS),chair=str(CHAIR),model_cache=str(SOURCE/'hf_cache'),phase_seconds=2100)
        rconfig=SOURCE/(name+'_rerank_config.json');rconfig.write_text(json.dumps(rcfg))
        run([sys.executable,'-u',SOURCE/'csa_per_caption_rerank_data.py',rconfig],900)
        script=SOURCE/'run_csa_per_caption_rerank.py'
        for mode,cap in (('prefetch',1200),('text_preflight',180)):
            run([sys.executable,'-u',script,mode,rconfig],cap)
        run([*launch,script,'worker',rconfig],2130)
        run([sys.executable,'-u',script,'score',rconfig],3600)
        result=json.loads((rerank/'result.json').read_text())
        if result['status']!='CSA_PER_CAPTION_FIXED_RERANK_VALID5000_COMPLETE' or result['checkpoint_sha256']!=receipt['sha256']:
            raise ValueError('Wrong arm VALID checkpoint/result')
        workers=[json.loads((folder/f'worker{{rank}}.json').read_text()) for rank in (0,1)]
        nll=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
        for metric in result['metrics'].values():metric['NLL']=nll
        result.update(arm=name,checkpoint=receipt,NLL_note='Same frozen captioner teacher-forced NLL for both selections.')
        (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        return result

    evaluate_arm('xe',xe_weights,xe_receipt,'2.11.0+cu128')
    current_rows=json.loads((ROOT_OUTPUT/'xe/rerank/decoder_rank0_captions.json').read_text())
    if len(original_captions)!=5000 or set(original_captions)!={{r['coco_id'] for r in current_rows}}:
        raise ValueError('Original/current XE VALID cohorts differ')
    changed=sum(r['caption']!=original_captions[r['coco_id']] for r in current_rows)
    (ROOT_OUTPUT/'xe_replay.json').write_text(json.dumps(dict(images=5000,changed_captions=changed,
        byte_identical_decode_claimed=False,reference='same cached evaluator across all pilot arms')))
    evaluate_arm('scst500',control_weights,control_receipt,'2.11.0+cu128')
    training=ROOT_OUTPUT/'train';training.mkdir()
    shutil.copyfile(ROOT_OUTPUT/'vocabulary.json',training/'vocabulary.json')
    cfg=dict(base_cfg,output=str(training),checkpoint_root=str(xe_weights),kl_beta=KL_BETA)
    cfg['train_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-1800)
    if cfg['train_seconds']<300:raise TimeoutError('Insufficient KL500 budget; baseline outputs preserved')
    write_config()
    run([*launch,pilot,'train',CONFIG],cfg['train_seconds']+30)
    final=training/'update0500'
    receipt=json.loads((final/'pilot_checkpoint_receipt.json').read_text())
    if receipt['updates']!=500 or receipt['source_sha256']!=XE_CHECKPOINT_SHA or receipt['kl_beta']!=KL_BETA or receipt['reference_sha256']!=XE_CHECKPOINT_SHA:
        raise ValueError('Wrong KL endpoint/reference')
    for rank in (0,1):
        guard=json.loads((training/f'kl_end_guard_rank{{rank}}.json').read_text())
        if not guard['teacher_versions_unchanged'] or not guard['teacher_gradients_absent']:raise ValueError('Teacher mutation')
    evaluate_arm('kl500',final/'pilot_checkpoint.bin',receipt,SCST_TRAIN_TORCH)
    run([sys.executable,'-u',SOURCE/'scst_kl_valid_compare.py',ROOT_OUTPUT],600)
    complete=json.loads((ROOT_OUTPUT/'result.json').read_text())['status']=='SCST_KL500_VALID_COMPLETE'
except BaseException:
    (ROOT_OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (ROOT_OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=True,test_used=False,
        seed=43,updates=500,beta=KL_BETA,xe_source=SOURCE_VERSION,control_source=CONTROL_VERSION)))
    review=ROOT_OUTPUT.with_name(ROOT_OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for path in ROOT_OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt'):z.write(path,str(path.relative_to(ROOT_OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',ROOT_OUTPUT,flush=True)
    print('VALID pilot complete; no TEST or automatic extension.' if complete else 'INCOMPLETE; preserve artifacts. No automatic restart.',flush=True)
'''
    nb['cells'][3]['source'] = [prefix+body]
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'notebook cell','exec')
            cell['execution_count']=None;cell['outputs']=[]
    for node in ast.walk(ast.parse(prefix+body)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
            if isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str) and '\n' in node.args[0].value:
                compile(node.args[0].value,'embedded source','exec')
    return nb


if __name__=='__main__':
    path=HERE/'CSA_SEED43_SCST_KL500_VALID5000_2GPU.ipynb'
    path.write_text(json.dumps(build(),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name)
