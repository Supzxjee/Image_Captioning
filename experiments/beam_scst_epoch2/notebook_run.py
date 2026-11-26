complete=False;training_complete=False;test_started=False;phase='source'
try:
    for label,root,payloads,hashes in (('train',SOURCE,TRAIN_PAYLOADS,TRAIN_SHA256),
                                      ('test',TEST_SOURCE,TEST_PAYLOADS,TEST_SHA256)):
        preserved=OUTPUT/'source'/label;preserved.mkdir(parents=True)
        for name,source in payloads.items():
            raw=source.encode('utf-8')
            if hashlib.sha256(raw).hexdigest()!=hashes[name]:raise ValueError('Changed embedded '+label+'/'+name)
            for dest in (root,preserved):
                path=dest/name
                if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe source path')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(train=TRAIN_SHA256,test=TEST_SHA256),indent=2))
    cfg['source_hashes']=TRAIN_SHA256;write_config()
    sys.path.insert(0,str(SOURCE))
    from checkpoint_input import resolve_mounted
    def checkpoint(root):return resolve_mounted(root,CHECKPOINT_SHA,OUTPUT/'checkpoint_input_report.json')
    if CHECKPOINT_ROOT:
        weights=checkpoint(CHECKPOINT_ROOT);origin='mounted exact source bytes'
    else:
        from kaggle_secrets import UserSecretsClient
        token=None
        for secret_name in SECRET_NAMES:
            try:value=UserSecretsClient().get_secret(secret_name).strip()
            except Exception:continue
            if value:token=value;break
        if not token:raise RuntimeError('Enable Secret codex or KAGGLE_API_TOKEN with access to exact ducanh3/version199')
        TOKENS.append(token)
        run([sys.executable,'-m','pip','install','-q','--target',CLI,
             'https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],600)
        api_env=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):api_env.pop(key,None)
        api_env.update(PYTHONPATH=str(CLI),KAGGLE_API_TOKEN=token,PYTHONUNBUFFERED='1')
        run([sys.executable,'-c','SOURCE_VERSION='+repr(SOURCE_VERSION)+'\n'+ROUTING_PREFLIGHT],120,api_env)
        target=OUTPUT/'input_source';target.mkdir()
        run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',SOURCE_VERSION,
             '-p',str(target),'--file-pattern','(^|/)beam_scst_epoch1_seed43_v179_v2_run1/checkpoint[.]bin$'],1200,api_env)
        weights=checkpoint(target);origin='exact Save Version API download'
    cfg['checkpoint_root']=str(weights);write_config()
    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(seed=43,version=SOURCE_VERSION,epoch=1,
        sha256=CHECKPOINT_SHA,path=str(weights),origin=origin),indent=2))
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    shutil.copyfile(OUTPUT/'cuda_runtime.json',OUTPUT/'training_runtime.json')
    from importlib.metadata import version
    expected_software=json.loads(TRAIN_PAYLOADS['PARENT_CHECKPOINT_RECEIPT.json'])['identity']
    if (json.loads((OUTPUT/'training_runtime.json').read_text())['torch_version']!=expected_software['torch']
            or version('numpy')!=expected_software['numpy']):
        raise RuntimeError('Training software differs from v199. Preserve log; no automatic reinstall or fallback.')
    phase='setup'
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
         'pycocoevalcap==1.2','nltk==3.9.2','future'],600)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
         'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],600)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],300)
    from full_epoch_data import prepare
    prepare(cfg)
    from csa_per_caption_test_data import bind_verified_cache
    bind_verified_cache(cfg,'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef')
    write_config()
    if hashlib.sha256(DETECTIONS.read_bytes()).hexdigest()!='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c':
        raise ValueError('Changed frozen rerank detector cache')
    test_out=OUTPUT/'test';test_out.mkdir()
    test_cfg=dict(semantic=True,prompt_test_root=str(PROMPT_TEST_ROOT),output=str(test_out),
        stage=str(WORK/'test_patches'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),
        checkpoint_root=str(OUTPUT/'pilot_checkpoint.bin'),chair=str(CHAIR),worker_seconds=900)
    test_config=WORK/'test_config.json';test_config.write_text(json.dumps(test_cfg,indent=2))
    def test_env():
        env=os.environ.copy();env['PYTHONPATH']=str(DEPS)+os.pathsep+str(TEST_SOURCE);return env
    evaluator=TEST_SOURCE/'run_csa_per_caption_test.py'
    # Check Java/metrics before spending the production epoch; no checkpoint/model needed.
    run([sys.executable,'-u',evaluator,'environment','--config',test_config],1200,test_env())
    run([sys.executable,'-u',evaluator,'smoke','--config',test_config],600,test_env())
    environment=json.loads((test_out/'environment.json').read_text())
    os.environ['PATH']=str(Path(environment['java']).parent)+os.pathsep+os.environ['PATH']
    os.environ['JAVA_HOME']=str(Path(environment['java']).parent.parent)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    phase='preflight'
    cfg['train_seconds']=min(1200,PARENT_DEADLINE-time.monotonic()-EVALUATION_RESERVE_SECONDS);write_config()
    if cfg['train_seconds']<300:raise TimeoutError('Insufficient budget before training preflight')
    run([*launch,SOURCE/'run_epoch2.py','preflight',CONFIG],cfg['train_seconds']+30)
    guards=[json.loads((OUTPUT/f'beam_preflight_rank{rank}.json').read_text()) for rank in (0,1)]
    if not all(g['passed'] for g in guards):raise ValueError('Incomplete epoch2 preflight')
    cfg['train_seconds']=PARENT_DEADLINE-time.monotonic()-EVALUATION_RESERVE_SECONDS;write_config()
    if cfg['train_seconds']<600:raise TimeoutError('Insufficient budget to start epoch2')
    print('Epoch2 guard PASS; full-epoch train projection hours:',round(max(g['projected_training_seconds'] for g in guards)/3600,2),
          '; available train hours:',round(cfg['train_seconds']/3600,2),'; not ETA/quota guarantee',flush=True)
    phase='train'
    run([*launch,SOURCE/'run_epoch2.py','train',CONFIG],cfg['train_seconds']+60)
    trained=json.loads((OUTPUT/'training_complete.json').read_text())
    if (not trained['complete'] or trained['epochs']!=2 or trained['train_images']!=113287
            or trained['updates']!=7082 or trained['cumulative_image_events']!=226574):
        raise ValueError('Refuse incomplete epoch2 endpoint')
    training_complete=True
    from epoch2_contract import validate_resume,validate_eval_binding
    full=json.loads((OUTPUT/'checkpoint_receipt.json').read_text())
    endpoint=json.loads((OUTPUT/'pilot_checkpoint_receipt.json').read_text())
    validate_resume(full,full['identity']);validate_resume(endpoint,full['identity'])
    if any(endpoint[k]!=full[k] for k in ('identity','epoch','updates','cursor_images','parent_sha256','source_sha256','complete_epoch')):
        raise ValueError('Epoch2 full/export receipt mismatch')
    if hashlib.sha256((OUTPUT/'pilot_checkpoint.bin').read_bytes()).hexdigest()!=endpoint['sha256']:
        raise ValueError('Changed endpoint export')
    binding=validate_eval_binding(dict(seed=43,version=OUTPUT.name+'/epoch2_endpoint',
        **{k:endpoint[k] for k in ('sha256','epoch','updates','cursor_images','parent_sha256','source_sha256','complete_epoch')}))
    (OUTPUT/'evaluation_binding.json').write_text(json.dumps(binding,indent=2))
    for name,value in (('EPOCH2_EVAL_LOCK.json',binding),('REPLAY_RECEIPTS.json',dict(checkpoint_sha256=binding['sha256']))):
        raw=(json.dumps(value,indent=2)+'\n').encode()
        for root in (TEST_SOURCE,OUTPUT/'source'/'test'):(root/name).write_bytes(raw)
    (OUTPUT/'runtime_input_receipt.json').write_text(json.dumps({n:hashlib.sha256((TEST_SOURCE/n).read_bytes()).hexdigest()
        for n in ('EPOCH2_EVAL_LOCK.json','REPLAY_RECEIPTS.json')},indent=2))
    phase='VALID'
    cfg.update(evaluate_pilot=True,eval_update=7082,worker_seconds=min(900,PARENT_DEADLINE-time.monotonic()-3600));write_config()
    if cfg['worker_seconds']<300:raise TimeoutError('Epoch2 saved; VALID needs recovery')
    valid_evaluator=SOURCE/'run_csa_per_caption_test.py'
    run([*launch,valid_evaluator,'worker','--config',CONFIG],cfg['worker_seconds']+30)
    for mode,cap in (('tokenize',180),('basic',240),('chair',600)):
        run([sys.executable,'-u',valid_evaluator,mode,'--config',CONFIG],cap)
    metrics={}
    for mode in ('basic','chair'):metrics.update(json.loads((OUTPUT/(mode+'.json')).read_text())['metrics'])
    workers=[json.loads((OUTPUT/f'worker{r}.json').read_text()) for r in (0,1)]
    metrics['NLL']=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    valid_result=dict(split='val',images=5000,epoch=2,updates=7082,checkpoint_sha256=binding['sha256'],
        metrics=metrics,epoch1_reference=REFERENCES['epoch1_VALID'],
        delta_vs_epoch1={k:v-REFERENCES['epoch1_VALID'][k] for k,v in metrics.items()},
        endpoint_fixed_before_TEST=True)
    (OUTPUT/'valid_result.json').write_text(json.dumps(valid_result,indent=2,allow_nan=False))
    print('VALID:',json.dumps(valid_result,indent=2),flush=True)
    phase='TEST';test_started=True
    run([sys.executable,'-u',evaluator,'checkpoint_preflight','--config',test_config],120,test_env())
    run([sys.executable,'-u',evaluator,'prepare','--config',test_config],1800,test_env())
    test_cfg['worker_seconds']=min(900,PARENT_DEADLINE-time.monotonic()-2400)
    if test_cfg['worker_seconds']<300:raise TimeoutError('Endpoint saved; TEST inference needs recovery')
    test_config.write_text(json.dumps(test_cfg,indent=2));(test_out/'protocol.json').write_text(test_config.read_text())
    run([*launch,evaluator,'worker','--config',test_config],test_cfg['worker_seconds']+30,test_env())
    rerank_out=test_out/'rerank';rerank_out.mkdir()
    rerank_cfg=dict(output=str(rerank_out),test_output=str(test_out),dataset_json=str(COCO_JSON),
        detections=str(DETECTIONS),chair=str(CHAIR),model_cache=str(WORK/'hf_cache'),phase_seconds=1200)
    rerank_config=WORK/'rerank_config.json';rerank_config.write_text(json.dumps(rerank_cfg,indent=2))
    (rerank_out/'protocol.json').write_text(rerank_config.read_text())
    run([sys.executable,'-u',TEST_SOURCE/'csa_per_caption_rerank_data.py',rerank_config],600,test_env())
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
         'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2'],300)
    reranker=TEST_SOURCE/'run_csa_per_caption_rerank.py'
    run([sys.executable,'-u',reranker,'prefetch',rerank_config],900,test_env())
    run([sys.executable,'-u',reranker,'text_preflight',rerank_config],180,test_env())
    run([*launch,reranker,'worker',rerank_config],1230,test_env())
    phase='CPU_metrics'
    run([sys.executable,'-u',reranker,'score',rerank_config],5400,test_env())
    result=json.loads((rerank_out/'result.json').read_text())
    if (result['status']!='CSA_PER_CAPTION_FIXED_RERANK_TEST5000_COMPLETE' or result['images']!=5000
            or result['checkpoint_sha256']!=binding['sha256'] or result['selector']!=REFERENCES['selector']):
        raise ValueError('Incomplete or changed TEST comparison')
    workers=[json.loads((test_out/f'worker{r}.json').read_text()) for r in (0,1)]
    nll=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    for m in result['metrics'].values():m['NLL']=nll
    keys=('BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall','NLL')
    comparison=dict(seed=43,split='test',images=5000,epoch=2,updates=7082,checkpoint=binding,
        CE_metrics=REFERENCES['CE'],epoch1_metrics=REFERENCES['epoch1_TEST'],epoch2_metrics=result['metrics'],
        delta_vs_epoch1={p:{k:m[k]-REFERENCES['epoch1_TEST'][p][k] for k in keys} for p,m in result['metrics'].items()},
        delta_vs_CE={p:{k:m[k]-REFERENCES['CE'][p][k] for k in keys} for p,m in result['metrics'].items()},
        checkpoint_selection='fixed epoch2 endpoint, no TEST selection',test_tuning=False,prior_project_test_exposure=True)
    result.update(status='BEAM_SCST_EPOCH2_TEST5000_RERANK_COMPLETE',epochs_SCST=2,updates=7082,seed=43)
    (rerank_out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (test_out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (OUTPUT/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False))
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='BEAM_SCST_EPOCH2_VALID_TEST_COMPLETE',
        training=trained,VALID=valid_result,TEST=result),indent=2,allow_nan=False))
    print('TEST comparison:',json.dumps(comparison,indent=2),flush=True);complete=True;phase='complete'
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,phase=phase,training_complete=training_complete,
        requested_epoch=2,requested_epoch2_images=113287,requested_cumulative_updates=7082,
        parent_source=SOURCE_VERSION,test_started=test_started,test_tuning=False,reranking=True),indent=2))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log','.py','.md'):
                archive.write(path,str(path.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    for name in ('checkpoint.bin','pilot_checkpoint.bin'):
        if (OUTPUT/name).is_file():print('SAVED WEIGHTS:',OUTPUT/name,flush=True)
    print('EPOCH2 + VALID + TEST COMPLETE' if complete else 'INCOMPLETE; preserve weights/captions/scores. No automatic restart, partial ranking or recipe fallback.',flush=True)
