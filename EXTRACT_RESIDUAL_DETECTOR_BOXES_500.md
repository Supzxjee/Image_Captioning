# Detector-box diagnostic trên cohort500 đã khóa

Giữ scorer R50seed42,scale0.1,thresholds đã học. Không train hoặc refit.
Cohort là500cặp/388ảnh của R53; manifest GT nằm trong repo, không cần ZIP cũ input.
Từ cache detector:confidence>=0.5,same-class NMS0.8,max100boxes. Match GT instances
được tham chiếu với detection cùngclass,IoU>=0.5,greedy IoU-descending one-to-one.
GT chỉ dùng match/labels; feature crop dùng tọa độ detector. Greedy không phải optimal assignment.

Báo missingsubject/missingobject/both,missingcache record vàcoveragepositive từngrelation.
Thiếu cache record không tự xem là detection failure. Detection extra không tự thành falsepositive.
Đây làconditional matched-pair evaluation, không fullHOI detection AP.

## Chạy Kaggle

T4 x2,Internet on. Input COCO và objectdetectioncache.json. Các đường dẫn theo lần trước
đã điền. Cache trước dùng pixel xyxy; chỉ đổi COORDINATE_FORMAT nếu biết cache normalized.
Không tự đoán format. Không cầncheckpoint,promptPT,groundedcache,V-COCOannotations mới.
Chờ Adding data xong. Hai GPU trích feature cho matchedpairs. Không train scorer.

Copy cell hoặc import RESIDUAL_DETECTOR_BOXES_500_FEATURES.ipynb:

```python
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COORDINATE_FORMAT = 'pixel_xyxy'
MIN_CONFIDENCE, MIN_IOU = 0.5, 0.5
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '26de41a'
MODEL_ID = 'openai/clip-vit-base-patch16'
MODEL_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
LOCAL_MODEL_PATH = ''  # Optional standard Hugging Face CLIP folder; not a Q-Former checkpoint.
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/clip_instance_alignment')
OUTPUT = Path('/kaggle/working/residual_detector_boxes_500_features')
DATA = OUTPUT / 'data'
PAIR_COUNT, SELECTION_SEED = 500, 20261013
EXCLUSIONS = REPO / 'audit/instance_alignment_seen_images.json'
PROTOCOL = REPO / 'audit/residual_frozen_evaluation_protocol.json'

print('Frozen residual detector-box feature extraction started.', flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert DETECTIONS.is_file(), DETECTIONS
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
os.environ.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                  HF_HUB_DISABLE_XET='1', OMP_NUM_THREADS='2')


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    started = time.monotonic()
    with subprocess.Popen(args, cwd=cwd) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'Process running: {time.monotonic()-started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)


REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.51,<5',
     'huggingface_hub>=0.34,<1', 'Pillow', 'numpy'], cwd=REPO)
OUTPUT.mkdir(parents=True, exist_ok=True)
configuration = {'code_commit': COMMIT, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                 'evaluation_pairs': PAIR_COUNT, 'selection_seed': SELECTION_SEED,
                 'exclusion_sha256': hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
                 'training': False, 'variant': 'detector_matched_features_v1',
                 'detections': str(DETECTIONS), 'min_confidence': MIN_CONFIDENCE,
                 'min_iou': MIN_IOU, 'coordinate_format': COORDINATE_FORMAT, 'gpu_workers': 2, 'seed': 42,
                 'caption_json_sha256': hashlib.sha256(COCO_JSON.read_bytes()).hexdigest(),
                 'coco_images': str(COCO_IMAGES), 'local_model_path': LOCAL_MODEL_PATH}
manifest = OUTPUT / 'manifest.json'
if manifest.is_file():
    assert json.loads(manifest.read_text())['configuration'] == configuration, 'Use a new OUTPUT for changed config.'
manifest.write_text(json.dumps({'configuration': configuration}, indent=2), encoding='utf-8')

# The reference cohort is already locked in the repo; no annotation download.
run([sys.executable, '-u', 'prepare_detector_box_pairs.py',
     '--pairs', REPO / 'audit/residual_new_image_pairs_500.json', '--detections', DETECTIONS,
     '--images', COCO_IMAGES, '--output', DATA, '--min-confidence', MIN_CONFIDENCE,
     '--min-iou', MIN_IOU, '--coordinate-format', COORDINATE_FORMAT], cwd=REPO)
shutil.copyfile(PROTOCOL, OUTPUT / 'evaluation_protocol.json')
if not LOCAL_MODEL_PATH:
    assert shutil.disk_usage('/tmp').free > 2 * 1024**3, 'Need free /tmp space for CLIP.'
    download_code = ('from huggingface_hub import snapshot_download; '
        f'snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, '
        f'local_dir={str(MODEL_DIR)!r}, max_workers=2, '
        "allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])")
    run([sys.executable, '-u', '-c', download_code])
assert (MODEL_DIR / 'config.json').is_file(), MODEL_DIR

common = [sys.executable, '-u', 'run_instance_alignment_diagnostic.py',
          '--data', DATA, '--images', COCO_IMAGES, '--model', MODEL_DIR,
          '--output', OUTPUT, '--shards', '2', '--evaluation-only']
workers = []
started = time.monotonic()
try:
    for index in range(2):
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(index)
        workers.append(subprocess.Popen(list(map(str, common + ['--shard', index])), cwd=REPO, env=environment))
    while True:
        statuses = [worker.poll() for worker in workers]
        if any(code is not None and code != 0 for code in statuses):
            raise RuntimeError(f'Feature worker failed: {statuses}; see preceding log.')
        if all(code == 0 for code in statuses):
            break
        print(f'Feature workers: {statuses} | elapsed {time.monotonic()-started:.0f}s', flush=True)
        time.sleep(30)
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()

summary = json.loads((DATA / 'detector_pair_summary.json').read_text())
assert summary['reference_pairs'] == 500
print('DETECTOR BOX FEATURE EXTRACTION COMPLETE; no scorer training performed.', flush=True)
print(json.dumps(summary['evaluation'], indent=2), flush=True)
archive = shutil.make_archive(str(OUTPUT) + '_review', 'zip', OUTPUT)
print('Send review ZIP:', archive, flush=True)

```

Gửi `residual_detector_boxes_500_features_review.zip` và Kaggle version.
Nếu0matchedpairs, script báocoverage vàdừng:kiểm tra classnames/format/cachefile,không chỉnhIoU
sau khi xemscore đểđẩymetrics. Config mới cầnghi riêng, không trộn outputs.

## Đánh giá CPU tại máy

```python
import subprocess, sys
subprocess.run([sys.executable, 'evaluate_frozen_residual.py',
    '--source', 'PATH_TO/residual_detector_boxes_500_features_review.zip',
    '--checkpoint', 'PATH_TO/clip_prior_residual_v1/residual_head.pt',
    '--detector-matched',
    '--reference-predictions', 'PATH_TO/residual_new_images_500_evaluation/predictions.json',
    '--output', 'PATH_TO/residual_detector_boxes_500_evaluation'], check=True)
```

Báo CLIP/residual trênmatchedpairs và GTboxscores trên chínhsubset đó; tách ảnh hưởng
mấtcặp vớiảnh hưởngcropbox. Maskunknown vẫn giữ. Không so trực tiếpAPmatchedsubset
vớiAP70.01 của toàn500 như cùngmẫu. Checksum cohort/model giữstrict,LF/CRLF compatible.

Validation:4matchingtests PASS +16supervised/residual/protocoltests PASS;syntax checked.
Chưa có cache detector trênmáy vàchưa chạy realdetectorfeatures nên chưa metrics.
Wearing/wrong-object labels/captionfusion chưa triển khai.
