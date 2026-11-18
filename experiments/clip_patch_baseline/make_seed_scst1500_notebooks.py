"""Replicate the historical 1000+500 SCST schedule for two XE seeds."""
import ast
import copy
import json

from make_seed_rerank_notebooks import HERE, code, embedded, embedded_node


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def pilot_source(source):
    return next(ast.literal_eval(n.args[0]) for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'write_text' and isinstance(n.func.value, ast.Name)
                and n.func.value.id == 'pilot')


def replace_functions(source, replacements):
    tree = ast.parse(source)
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            name = node.name if isinstance(node, ast.FunctionDef) else getattr(node.targets[0], 'id', None)
            if name in replacements:
                tree.body[i] = ast.parse(replacements[name]).body[0]
    return ast.unparse(tree) + '\n'


RUNTIME_POLICY = '''def runtime_policy(training, current):
    allowed = ('2.11.0+cu128', '2.11.0+cu130')
    if training not in allowed or current not in allowed:
        raise RuntimeError('Unreviewed Torch build; keep log, do not blindly reinstall.')
    return dict(training_torch=training, inference_torch=current,
        cuda_wheel_changed=training!=current,
        policy='Recorded training and inference Torch builds; FP32 and strict checkpoint checks.',
        bitwise_cross_build_equivalence_established=False)
'''


def seed_valid_contract(seed, fresh, original):
    source = embedded(code(fresh, 3), 'csa_per_caption_test_data.py')
    nodes = ast.parse(embedded(code(original, 3), 'csa_per_caption_test_data.py')).body
    replacements = {}
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and n.name in ('lock', 'validate_lock', 'checkpoint', 'validate_checkpoint_state'):
            replacements[n.name] = ast.unparse(n).replace('shared.lock()', 'valid_cohort()')
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) in ('CHECKPOINT_SHA', 'VERSION'):
            replacements[n.targets[0].id] = ast.unparse(n)
    replacements['runtime_policy'] = RUNTIME_POLICY
    source = replace_functions(source, replacements)
    source += '''
def bind_verified_cache(cfg, cache_source_sha):
    configure_valid(cfg)
    out = Path(cfg['output'])
    cached = read(out/'test_lock.json')
    expected = lock()
    same = ('expected_test_ids', 'test_id_sha256', 'beam_width', 'max_new_tokens',
            'length_normalization', 'reranking', 'training_commit', 'method')
    if any(cached[k] != expected[k] for k in same):
        raise ValueError('Cache cohort, decoder or architecture differs')
    if cached['checkpoint_sha256'] != cache_source_sha:
        raise ValueError('Wrong original cache source')
    rebound = dict(cached, **expected)
    rebound['checkpoint'] = cfg['checkpoint_root']
    validate_lock(rebound)
    save(out/'cache_checkpoint_binding.json', dict(
        original_cache_source_sha256=cache_source_sha, xe_checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_source=VERSION, original_lock=cached, bound_lock=rebound,
        cache_payloads_changed=False, training_references_scope='TRAIN/restval only'))
    save(out/'test_lock.json', rebound)
'''
    return source


def test_contract(seed, template, best_epoch, xe_sha):
    source = embedded(code(template, 3), 'csa_per_caption_test_data.py')
    replacements = {
        'CHECKPOINT_SHA': "CHECKPOINT_SHA=LINEAGE['sha256']",
        'XE_SOURCE_SHA': f'XE_SOURCE_SHA={xe_sha!r}',
        'PARENT_SHA': "PARENT_SHA=LINEAGE['parent_sha256']",
        'VERSION': f"VERSION='csa_per_caption_seed{seed}_scst1500'",
        'runtime_policy': RUNTIME_POLICY,
    }
    tree = ast.parse(source)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ('lock', 'validate_lock'):
            class Epoch(ast.NodeTransformer):
                def visit_Constant(self, node):
                    return ast.copy_location(ast.Constant(best_epoch), node) if node.value == 9 else node
            replacements[n.name] = ast.unparse(Epoch().visit(n))
    source = replace_functions(source, replacements)
    source = source.replace('CHECKPOINT_SHA = LINEAGE', "LINEAGE = read(HERE/'SCST_TEST_LINEAGE.json')\nCHECKPOINT_SHA = LINEAGE", 1)
    return source.replace('Missing exact v94 update1500 checkpoint SHA80e1280b', 'Missing this seed exact update1500 checkpoint')


def build(seed):
    original = load(f'CSA_PER_CAPTION_SEED{seed}_TEST5000_API_2GPU.ipynb')
    fresh = load('CLIP_PATCH_CSA_SCST_ONLY1000_MERGED_DATASET_2GPU.ipynb')
    resume = load('CLIP_PATCH_CSA_SCST_RESUME1500_RERANK_VALID5000_2GPU.ipynb')
    test = load('CLIP_PATCH_CSA_SCST_ONLY1500_RERANK_TEST5000_2GPU.ipynb')
    previous = load(f'CSA_PER_CAPTION_SEED{seed}_RERANK_TEST5000_RESULT.json')
    best_epoch = previous['test_lock']['epoch']
    xe_sha = previous['result']['checkpoint_sha256']
    nb = copy.deepcopy(original)
    output_name = f'csa_per_caption_seed{seed}_scst1500_test_rerank'
    nb['cells'][0]['source'] = [
        f'# Seed {seed}: SCST1500, then TEST5000 + fixed rerank\n\n'
        f'Own best XE source {previous["result"]["checkpoint_source"]}. '
        'Fresh1000 + optimizer-resume500, CIDEr-D SCST only, no CE/object reward, batch8, FP32 DDP. '
        'Uses the same sealed TRAIN order as seed42. Fixed endpoint1500; no TEST checkpoint selection. '
        'Runs VALID basic/CHAIR/NLL diagnostic, then TEST top1 and fixed0.3/0.4/0.3 rerank with eight metrics.\n\n'
        'T4 x2 + Internet + Kaggle Secret codex or KAGGLE_API_TOKEN with access to the own XE source. '
        'Attach csa-scst-full-karpathy-cache-v1, visual-cache, mscoco-2014, '
        'csa-prompt-test-cache-v181 and objectdetectionecache. Exact best.bin downloads by API. '
        'No checkpoint mount required. Cached patch arrays are linked readonly. '
        'Keep the entire output with update500/1000/1500 weights, beams and CLIP scores. '
        f'Send {output_name}_review.zip. Four-hour STOP cap, not an ETA; no automatic restart.\n']
    setup = code(load(f'CSA_PER_CAPTION_SEED{seed}_RERANK_TEST5000_API_2GPU.ipynb'), 1)
    setup = setup.replace('WALL_HOURS=3', 'WALL_HOURS=4')
    setup = setup.replace(f'csa_per_caption_seed{seed}_rerank_test5000', output_name)
    setup = setup.replace('6*1024**3', '8*1024**3').replace('Need6GiB', 'Need8GiB')
    setup += "\nROOT_OUTPUT=OUTPUT\nCACHE_ROOT='/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1'\n"
    setup += "if not Path(CACHE_ROOT).is_dir():raise FileNotFoundError('Attach full SCST cache or correct CACHE_ROOT')\n"
    setup += "cfg.update(train_stage=str(SOURCE/'pilot_train'),train_seconds=3600)\nwrite_config()\n"
    nb['cells'][1]['source'] = [setup]

    seed_main = code(original, 3)
    prefix = seed_main[:seed_main.index("    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS")]
    prefix = prefix.replace("    prompt_inputs(cfg) # exact TEST181 seal/scope check before expensive setup/stage\n", '')
    tree = ast.parse(prefix + 'except BaseException:\n    raise\n')
    embedded_node(tree, 'csa_per_caption_test_data.py').args[0] = ast.Constant(seed_valid_contract(seed, fresh, original))
    valid_runner = embedded(code(resume, 3), 'run_csa_per_caption_test.py').replace(
        "d.runtime_policy('2.11.0+cu128',str(torch.__version__))",
        "d.runtime_policy(read(Path(cfg['output'])/'training_runtime.json')['torch_version'],str(torch.__version__))")
    embedded_node(tree, 'run_csa_per_caption_test.py').args[0] = ast.Constant(valid_runner)
    prefix = ast.unparse(tree).split('\nexcept BaseException:')[0] + '\n'
    prefix += "    XE_CHECKPOINT_SHA=CHECKPOINT_SHA\n"
    prefix += "    training_runtime=json.loads((ROOT_OUTPUT/'cuda_runtime.json').read_text())\n"
    prefix += "    SCST_TRAIN_TORCH=training_runtime['torch_version']\n"
    prefix += "    (ROOT_OUTPUT/'training_runtime.json').write_text(json.dumps(training_runtime))\n"

    old = code(fresh, 3)
    cache = old[old.index("    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS"):old.index("    cfg['train_seconds']=min(3600")]
    cache = cache.replace('    mount(cfg,local,OUTPUT,linked_patches=links)',
        "    mount(cfg,local,OUTPUT,linked_patches=links)\n"
        "    from csa_per_caption_test_data import bind_verified_cache\n"
        "    bind_verified_cache(cfg,'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef')")
    first = pilot_source(old).replace('torch.manual_seed(4242+rank)', f'torch.manual_seed({seed*100+42}+rank)')
    cache_tree = ast.parse('if True:\n' + cache)
    call = next(n for n in ast.walk(cache_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'write_text' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'pilot')
    call.args[0] = ast.Constant(first)
    cache = '\n'.join(ast.unparse(cache_tree).splitlines()[1:]) + '\n'
    second = pilot_source(code(resume, 3)).replace('torch.manual_seed(4242+rank)', f'torch.manual_seed({seed*100+42}+rank)')
    second = second.replace('torch.manual_seed(5242+rank)', f'torch.manual_seed({seed*100+1042}+rank)')
    test_runner = embedded(code(test,3),'run_csa_per_caption_test.py').replace(
        "d.runtime_policy('2.11.0+cu128',str(torch.__version__))",
        "d.runtime_policy(d.LINEAGE['training_torch'],str(torch.__version__))")
    train = f'''    (SOURCE/'run_scst_resume1500.py').write_text({second!r}, encoding='utf-8')
    cfg['train_seconds']=min(3600,PARENT_DEADLINE-time.monotonic()-2100)
    if cfg['train_seconds']<300:raise TimeoutError('Insufficient time before first1000')
    write_config()
    run([*launch,pilot,'train',CONFIG],cfg['train_seconds']+30)
    first_folder=ROOT_OUTPUT/'update1000'
    first_receipt=json.loads((first_folder/'pilot_checkpoint_receipt.json').read_text())
    if first_receipt['updates']!=1000 or first_receipt['source_sha256']!=XE_CHECKPOINT_SHA:
        raise ValueError('Wrong first1000 lineage')
    first_weights=first_folder/'pilot_checkpoint.bin'
    if hashlib.file_digest(first_weights.open('rb'),'sha256').hexdigest()!=first_receipt['sha256']:
        raise ValueError('Changed first1000 checkpoint')
    continuation=ROOT_OUTPUT/'continuation';continuation.mkdir()
    shutil.copyfile(ROOT_OUTPUT/'vocabulary.json',continuation/'vocabulary.json')
    cfg.update(output=str(continuation),checkpoint_root=str(first_weights),resume_sha256=first_receipt['sha256'])
    cfg['train_seconds']=min(3600,PARENT_DEADLINE-time.monotonic()-2100)
    if cfg['train_seconds']<300:raise TimeoutError('Insufficient time before continuation;1000 preserved')
    write_config()
    run([*launch,SOURCE/'run_scst_resume1500.py','train',CONFIG],cfg['train_seconds']+30)
    final_folder=continuation/'update1500'
    receipt=json.loads((final_folder/'pilot_checkpoint_receipt.json').read_text())
    if receipt['updates']!=1500 or receipt['source_sha256']!=XE_CHECKPOINT_SHA or receipt['parent_sha256']!=first_receipt['sha256']:
        raise ValueError('Wrong final1500 lineage')
    final_weights=final_folder/'pilot_checkpoint.bin'
    if hashlib.file_digest(final_weights.open('rb'),'sha256').hexdigest()!=receipt['sha256']:
        raise ValueError('Changed final1500 checkpoint')
    shutil.copyfile(final_folder/'pilot_checkpoint_receipt.json',ROOT_OUTPUT/'pilot_checkpoint_receipt.json')
    cfg.update(output=str(ROOT_OUTPUT),checkpoint_root=str(weights),evaluate_pilot=True,eval_update=1500,eval_checkpoint=str(final_weights))
    cfg['worker_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-1800)
    if cfg['worker_seconds']<300:raise TimeoutError('Insufficient VALID budget;1500 preserved')
    write_config()
    evaluator=SOURCE/'run_csa_per_caption_test.py'
    run([*launch,evaluator,'worker','--config',CONFIG],cfg['worker_seconds']+30)
    for mode,cap in (('tokenize',180),('basic',240),('chair',600)):
        run([sys.executable,'-u',evaluator,mode,'--config',CONFIG],cap)
    valid_metrics={{}}
    for mode in ('basic','chair'):valid_metrics.update(json.loads((ROOT_OUTPUT/(mode+'.json')).read_text())['metrics'])
    valid_workers=[json.loads((ROOT_OUTPUT/f'worker{{rank}}.json').read_text()) for rank in (0,1)]
    valid_metrics['NLL']=sum(w['NLL_sum'] for w in valid_workers)/sum(w['NLL_tokens'] for w in valid_workers)
    (ROOT_OUTPUT/'valid1500.json').write_text(json.dumps(dict(split='val',images=5000,metrics=valid_metrics,checkpoint=receipt,checkpoint_selection='fixed1500, not selected using these metrics')))
    TRAIN_SOURCE=SOURCE
    SOURCE=TRAIN_SOURCE/'test_code';SOURCE.mkdir()
    for pattern in ('*.py','*.json'):
        for path in TRAIN_SOURCE.glob(pattern):shutil.copyfile(path,SOURCE/path.name)
    shutil.copytree(TRAIN_SOURCE/'captioning',SOURCE/'captioning')
    OUTPUT=ROOT_OUTPUT/'test';OUTPUT.mkdir()
    CONFIG=SOURCE/'test_config.json'
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    (SOURCE/'SCST_TEST_LINEAGE.json').write_text(json.dumps(dict(receipt,training_torch=SCST_TRAIN_TORCH)))
    (SOURCE/'csa_per_caption_test_data.py').write_text({test_contract(seed,test,best_epoch,xe_sha)!r},encoding='utf-8')
    (SOURCE/'run_csa_per_caption_test.py').write_text({test_runner!r},encoding='utf-8')
    cfg=dict(semantic=True,prompt_test_root=str(PROMPT_TEST_ROOT),output=str(OUTPUT),stage=str(SOURCE/'test_patches'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),checkpoint_root=str(final_weights),chair=str(CHAIR),worker_seconds=1800)
    write_config()
    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(xe_source=SOURCE_VERSION,xe_sha256=XE_CHECKPOINT_SHA,scst=receipt,seed=SEED,fixed_updates=1500)))
    for mode,cap in (('checkpoint_preflight',120),('environment',1200),('smoke',600),('prepare',1800)):
        run([sys.executable,'-u',SOURCE/'run_csa_per_caption_test.py',mode,'--config',CONFIG],cap)
    cfg['worker_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-1800)
    if cfg['worker_seconds']<300:raise TimeoutError('Insufficient TEST budget;1500 and VALID preserved')
    write_config()
    run([*launch,SOURCE/'run_csa_per_caption_test.py','worker','--config',CONFIG],cfg['worker_seconds']+30)
    CHECKPOINT_SHA=receipt['sha256']
'''
    old_test = code(test, 3)
    rerank = old_test[old_test.index("    for path,digest in {'captioning/object_consistency.py'"):old_test.index("    result=json.loads((rerank_out/'result.json').read_text())")]
    baseline = previous['result']['metrics']
    keys = ['BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall']
    xe_metrics = {policy: {key: baseline[policy][key] for key in keys} for policy in ('decoder_rank0','clip_object_fixed')}
    end = f'''    result=json.loads((rerank_out/'result.json').read_text())
    if result['status']!='CSA_PER_CAPTION_FIXED_RERANK_TEST5000_COMPLETE' or result['images']!=5000 or result['checkpoint_sha256']!=receipt['sha256']:
        raise ValueError('Incomplete final TEST/rerank')
    if result['selector']!={previous['result']['selector']!r}:raise ValueError('Changed fixed selector')
    test_workers=[json.loads((OUTPUT/f'worker{{rank}}.json').read_text()) for rank in (0,1)]
    nll=sum(w['NLL_sum'] for w in test_workers)/sum(w['NLL_tokens'] for w in test_workers)
    for metrics in result['metrics'].values():metrics['NLL']=nll
    xe_metrics={xe_metrics!r}
    comparison=dict(seed=SEED,split='test',images=5000,xe_checkpoint_source=SOURCE_VERSION,xe_sha256=XE_CHECKPOINT_SHA,scst_checkpoint=receipt,
        xe_rerank_output_source={previous['rerank_output_source']!r},
        xe_metrics=xe_metrics,scst_metrics=result['metrics'],delta_vs_same_seed_XE={{p:{{k:result['metrics'][p][k]-v for k,v in m.items()}} for p,m in xe_metrics.items()}},
        fixed_endpoint=1500,selected_on_test=False,prior_project_test_exposure=True)
    (ROOT_OUTPUT/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False))
    result.update(seed=SEED,xe_checkpoint_source=SOURCE_VERSION,xe_checkpoint_sha256=XE_CHECKPOINT_SHA,scst_checkpoint=receipt,
        fixed_updates=1500,training_torch=SCST_TRAIN_TORCH,training_rule='SCST only, fresh1000 then optimizer-resume500',
        policy_rng_seeds=[{seed*100+42},{seed*100+1042}],prior_XE_test_source=TEST_OUTPUT_VERSION,
        prior_XE_rerank_source={previous['rerank_output_source']!r},
        NLL_note='Measured frozen-captioner teacher-forced NLL, identical for both selection policies.')
    (rerank_out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    run_result=dict(result,status='CSA_PER_CAPTION_SEED_SCST1500_TEST_RERANK_COMPLETE',training=True)
    (ROOT_OUTPUT/'result.json').write_text(json.dumps(run_result,indent=2,allow_nan=False))
    complete=True
except BaseException:
    (ROOT_OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (ROOT_OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=True,test_used=bool((ROOT_OUTPUT/'test').exists()),test_tuning=False,seed=SEED,xe_source=SOURCE_VERSION,target_updates=1500)))
    review=ROOT_OUTPUT.with_name(ROOT_OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for path in ROOT_OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt'):z.write(path,str(path.relative_to(ROOT_OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',ROOT_OUTPUT,flush=True)
    print('SCST1500 + TEST/rerank complete.' if complete else 'INCOMPLETE; preserve checkpoints/logs. No automatic restart.',flush=True)
'''
    main = prefix + cache + train + rerank + end
    ast.parse(main)
    nb['cells'][3]['source'] = [main]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'seed{seed}-cell', 'exec')
            cell['execution_count'] = None
            cell['outputs'] = []
    for node in ast.walk(ast.parse(main)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text' and node.args:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) and '\n' in node.args[0].value:
                compile(node.args[0].value, 'embedded source', 'exec')
    return nb


if __name__ == '__main__':
    for seed in (43,44):
        path = HERE/f'CSA_PER_CAPTION_SEED{seed}_SCST1500_TEST_RERANK_2GPU.ipynb'
        path.write_text(json.dumps(build(seed),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(path.name)
