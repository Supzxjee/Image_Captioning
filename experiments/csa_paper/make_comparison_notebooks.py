"""Author two independent source-pinned user-run notebooks; never runs models."""
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER
from make_full_xe_notebook import FILES as BASE

FILES=list(dict.fromkeys(BASE+['plain_model.py','run_plain_smoke.py','prepare_test_regions.py','run_csa10_test.py',
    'test_comparison_setup.py','make_comparison_notebooks.py','PLAIN_VS_CSA10_PROTOCOL.md','prepare_real_assets.py','audit_assets.py','data.py',
    'run_best9_decode.py','saved_quality_data.py','best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json','SAVE_XE139_PACKAGE.json',
    'full_roi_review_text/input_receipt.json']))


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    bootstrap=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session/output path.')
if not all(p.is_dir() for p in INPUT_ROOTS):raise FileNotFoundError('Add the required Datasets and check paths before running.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa10-comparison-'));LOG=OUTPUT/'notebook.log';STAGE=SOURCE/'stage'
os.environ['OMP_NUM_THREADS']='1'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    archive='''
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,test_used=TEST_USED)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE: preserve logs/progress, do not restart automatically.',flush=True)
'''
    plain_setup='''
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
RESUME_ROOT='' # Fresh baseline. Only set to a COMPLETE saved baseline output folder after interruption.
OUTPUT=Path('/kaggle/working/plain_full_xe_10')
INPUT_ROOTS=[FULL_ROOT,BANK_ROOT];ARM='plain';TEST_USED=False
'''
    # Import Path before user settings; identical source bootstrap in both notebooks.
    plain_setup='from pathlib import Path\n'+plain_setup+bootstrap
    plain='''
complete=False;deadline=time.monotonic()+10.5*3600
common=['--bank-root',BANK_ROOT,'--stage',STAGE,'--output',OUTPUT,'--arm','plain','--target-epochs','10']
try:
    if shutil.disk_usage('/tmp').free<6*1024**3:raise RuntimeError('Need6GiB free temporary disk')
    run([sys.executable,'-u',SOURCE/'test_full_xe.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_comparison_setup.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2'],timeout=300)
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','prepare','--full-root',FULL_ROOT,*common],timeout=900)
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','metric_preflight',*common],timeout=300)
    run([sys.executable,'-u',SOURCE/'run_plain_smoke.py','--stage',STAGE,'--output',OUTPUT],timeout=180)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    preflight=OUTPUT/'preflight';preflight.mkdir()
    run([*launch,SOURCE/'run_sharing_benchmark.py','worker','--bank-root',BANK_ROOT,'--stage',STAGE,'--output',preflight],timeout=600)
    if json.loads((preflight/'result.json').read_text(encoding='utf-8'))['preflight_cases']!=6:raise ValueError('Missing DDP guards')
    remaining=deadline-time.monotonic()
    if remaining<600:raise TimeoutError('No training margin after setup')
    run([*launch,SOURCE/'run_full_xe.py','worker',*common,'--resume-root',RESUME_ROOT,'--wall-seconds',remaining-60],timeout=remaining)
    result=OUTPUT/'result.json'
    complete=result.is_file() and json.loads(result.read_text(encoding='utf-8'))['status']=='PLAIN_XE_10_COMPLETE'
    if not complete:raise RuntimeError('Training stopped early. Preserve full output to resume.')
    print('PRIMARY comparison uses checkpoint.bin EXACT epoch10, not best.bin.',flush=True)
'''+archive
    test_setup='''
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
CHECKPOINT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-xe10-v139')
IMAGE_ROOT=COCO_JSON.parent/'images'
OUTPUT=Path('/kaggle/working/csa_epoch10_karpathy_test')
INPUT_ROOTS=[COCO_JSON.parent,VISUAL_CACHE,DETECTIONS.parent,CHECKPOINT_ROOT];ARM='csa10-test';TEST_USED=True
'''
    test_setup='from pathlib import Path\n'+test_setup+bootstrap
    old=json.loads((HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb').read_text(encoding='utf-8'))
    oldcode=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'complete=False' in c['source'])
    dependencies=oldcode[oldcode.index("    run([sys.executable,'-m','pip'"):oldcode.index("    run([sys.executable,'-u','-m','torch.distributed.run'")]
    test='''
complete=False;deadline=time.monotonic()+7200
try:
    sys.path.insert(0,str(SOURCE))
    from run_best9_decode import unique_file
    from run_csa10_test import CHECKPOINT_SHA
    locked=unique_file(CHECKPOINT_ROOT,'checkpoint.bin',CHECKPOINT_SHA)
    (OUTPUT/'test_lock.json').write_text(json.dumps(dict(epoch=10,updates=177110,checkpoint_sha256=CHECKPOINT_SHA,checkpoint=str(locked),beam=5,max_new_tokens=20,selection='fixed beforeTEST')),encoding='utf-8')
    for p in (COCO_JSON,DETECTIONS):
        if not p.is_file():raise FileNotFoundError(p)
    run([sys.executable,'-u',SOURCE/'test_comparison_setup.py'],timeout=120)
'''+dependencies+'''
    CONFIG=SOURCE/'test_config.json'
    CONFIG.write_text(json.dumps(dict(dataset_json=str(COCO_JSON),detections=str(DETECTIONS),visual_cache=str(VISUAL_CACHE),image_root=str(IMAGE_ROOT),output=str(OUTPUT))),encoding='utf-8')
    run([sys.executable,'-u',SOURCE/'prepare_test_regions.py','--config',CONFIG],timeout=min(3720,deadline-time.monotonic()))
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    remaining=deadline-time.monotonic()
    if remaining<300:raise TimeoutError('No TEST inference margin')
    run([*launch,SOURCE/'run_csa10_test.py','worker','--output',OUTPUT,'--checkpoint-root',CHECKPOINT_ROOT],timeout=min(2460,remaining))
    run([sys.executable,'-u',SOURCE/'run_csa10_test.py','score','--output',OUTPUT,'--chair',CHAIR],timeout=min(1200,deadline-time.monotonic()))
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='CSA10_KARPATHY_TEST_COMPLETE'
    if not complete:raise RuntimeError('TEST evaluation incomplete')
'''+archive
    notes=[('PLAIN_XE10_2GPU.ipynb',plain_setup,plain,'''# NoCSA baseline — exact10XE, Karpathy TRAIN/VALID
Use account with30h available. GPU T4x2, InternetON, SaveVersion/RunAll. Add fullROI134 andsemanticbanks130 Datasets; paths prefilled.
Fresh baseline: same projection/decoder/head,3ordinary visualTransformer layers, SAME outputmemory to both decoderports. No banksemantics inforward. Banks input only validates originalassets and disposableDDPpreflight. No pretrainedCSAweights.
Same10epochs/global32/FP32/Adam/LR/warmup/order/beam5/max20 as historicalCSA10. Common initial layers match seed42 exactly; parametercount differs. Actual-input disposableplain smoke+6DDPguards beforetraining. Full5000VALID each epoch; TESTnever used here.
PRIMARY endpoint comparison uses EXACT epoch10 checkpoint.bin, not validationbest. Keep bothcheckpoint.bin andbest.bin withreceipts/fulloutput. Later evaluate baseline10 on the reusableTESTcache produced by notebook2; noTESTcache is required to starttraining.
10.5h elapsed stopcap, notETA/billing. No SCST ornovelalignment. Send plain_full_xe_10_review.zip. Do not deleteoutputweights.
'''),('CSA_EPOCH10_KARPATHY_TEST_2GPU.ipynb',test_setup,test,'''# FixedCSA epoch10 — Karpathy TEST5000
GPU T4x2/InternetON, account with20h available; can run independently alongsidebaseline. Add4Datasets: mscoco-2014 (images+dataset_coco.json), visual-cache (originalrawCLIPH5), objectdetectionecache, csa-full-xe10-v139. Paths prefilled; editonlyactualmountpaths.
Exactlatest epoch10 checkpoint272f843a/177110updates; neverbest9 orepoch15. Fixedbeam5/max20/rawscore, noTESTcheckpointselection ortraining. TwoindependentGPUinferenceworkers, noDDPgradientcollectives. SeparateteacherforcedTESTNLL; references neverenterbeamgeneration.
TRAIN/VALIDROI134 hasnoTEST. FirstCPU-pool originalcachedTESTrawpatches withSAME detector/ROI rules/JPEGdimensions/vocabulary/profile, theninfer5000testcaptions. No freshCLIP/detector/weightsdownload; missingoriginalcache stops ratherthansubstitutinginputs. Save test_regions.npz/test_metadata.json/test_receipt.json/vocabulary.json asreusablecache forbaselineTEST later.
ReportCIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/counts/meanlength/NLL. CHAIRobjectmetric only.2hparent stopcap, notETA. Send csa_epoch10_karpathy_test_review.zip and preserveENTIREoutputincludingNPZ (reviewexcludesarrays).
''')]
    for name,setup,execute,markdown in notes:
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),
            cells=[cell('markdown',markdown),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Worker active;').replace('Smoke wall cap exceeded','Phase cap exceeded')),cell('code',execute)])
        p=HERE/name;p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8');print(name,p.stat().st_size,'bytes, unexecuted')


if __name__=='__main__':
    import sys
    make(sys.argv[1] if len(sys.argv)>1 else 'HEAD')
