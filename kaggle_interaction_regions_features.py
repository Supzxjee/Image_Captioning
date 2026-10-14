import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '558b518'
MODEL_ID, MODEL_REVISION = 'openai/clip-vit-base-patch16', '57c216476eefef5ab752ec549e440a49ae4ae5f3'
MODEL_DIR = Path('/tmp/clip_interaction_regions')
OUTPUT = Path('/kaggle/working/interaction_regions_v1')
PROTOCOL = REPO / 'audit/interaction_region_protocol.json'

print('SHARED UNION INTERACTION REGIONS: bootstrap started.',flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
os.environ.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1',OMP_NUM_THREADS='2')


def run(args,cwd=None):
    args=list(map(str,args));print('\nRunning:', ' '.join(args),flush=True)
    started=time.monotonic()
    with subprocess.Popen(args,cwd=cwd) as process:
        while True:
            try:code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:print(f'Process alive: {time.monotonic()-started:.0f}s',flush=True)
        if code:raise subprocess.CalledProcessError(code,args)


if not REPO.exists():run(['git','clone','https://github.com/Supzxjee/Image_Captioning.git',REPO])
assert (REPO/'.git').is_dir(), REPO
run(['git','fetch','--depth','30','origin','main'],cwd=REPO)
run(['git','checkout','--detach',COMMIT],cwd=REPO)
run([sys.executable,'-m','pip','install','-q','transformers>=4.51,<5','huggingface_hub>=0.34,<1','Pillow','numpy'],cwd=REPO)
assert hashlib.sha256(COCO_JSON.read_bytes()).hexdigest() == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d', 'COCO metadata changed.'
OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'data').mkdir(exist_ok=True)
protocol=json.loads(PROTOCOL.read_text())
configuration={'code_commit':COMMIT,'training':False,'variant':protocol['variant'],
    'model':MODEL_ID,'model_revision':MODEL_REVISION,'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),'gpu_workers':2}
manifest=OUTPUT/'manifest.json'
if manifest.is_file():assert json.loads(manifest.read_text())['configuration']==configuration, 'Changed protocol: choose a new output.'
manifest.write_text(json.dumps({'configuration':configuration},indent=2),encoding='utf-8')
for split in ('train','val','transfer','full100'):
    shutil.copyfile(REPO/'audit'/f'interaction_region_{split}_pairs.json',OUTPUT/'data'/f'{split}_pairs.json')
shutil.copyfile(PROTOCOL,OUTPUT/'extraction_protocol.json')
print('Pair counts:',{s:v['pairs'] for s,v in protocol['splits'].items()},flush=True)
download=('from huggingface_hub import snapshot_download; '
    f'snapshot_download(repo_id={MODEL_ID!r},revision={MODEL_REVISION!r},local_dir={str(MODEL_DIR)!r},max_workers=2, '
    "allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])")
run([sys.executable,'-u','-c',download])
common=[sys.executable,'-u','extract_interaction_regions.py','--images',COCO_IMAGES,'--model',MODEL_DIR,'--output',OUTPUT,'--shards','2']
workers=[];started=time.monotonic()
try:
    for index in range(2):
        environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(index)
        workers.append(subprocess.Popen(list(map(str,common+['--shard',index])),cwd=REPO,env=environment))
    while True:
        statuses=[worker.poll() for worker in workers]
        if any(code is not None and code != 0 for code in statuses):raise RuntimeError(f'Extraction failed: {statuses}')
        if all(code==0 for code in statuses):break
        print(f'GPU workers {statuses};elapsed {time.monotonic()-started:.0f}s',flush=True)
        time.sleep(30)
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
run([sys.executable,'-u','validate_interaction_regions.py','--output',OUTPUT],cwd=REPO)
archive=shutil.make_archive(str(OUTPUT)+'_review','zip',OUTPUT)
print('\nREGION FEATURES READY; no model training performed.',flush=True)
print('Send ZIP:',archive,flush=True)
