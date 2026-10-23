"""Native/detail-degraded geometry and locked sampling; no torch/model imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_ID='openai/clip-vit-base-patch16'
REVISION='57c216476eefef5ab752ec549e440a49ae4ae5f3'
MEAN=[.48145466,.4578275,.40821073]
STD=[.26862954,.26130258,.27577711]
QUADRANTS=((0.,0.,.5,.5),(.5,0.,1.,.5),(0.,.5,.5,1.),(.5,.5,1.,1.))
ARCHIVE_SHA='ebf880fcc0ae23388057abc84ce796b64ee1f0f6c16f804d1cdbf98e73fb78fd'
PAYLOAD_PER_IMAGE=2*196*512*2
FULL_IMAGES=15000


def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def select_sample(train):
    if len(train)!=10000 or len({r['coco_id'] for r in train})!=10000:raise ValueError('Expected locked TRAIN10k.')
    order=sorted(train,key=lambda r:hashlib.sha256(('native-detail-benchmark-v1:'+str(r['coco_id'])).encode()).hexdigest())
    return dict(warmup=order[:16],measure=order[16:272])


def make_tiles(image,condition):
    if condition not in ('H','L'):raise ValueError(condition)
    if image.mode!='RGB':raise ValueError('Decode/convert RGB before tiling.')
    source=image if condition=='H' else image.resize((224,224),Image.Resampling.BILINEAR)
    width,height=source.size
    # Fractional resize boxes preserve the same normalized half-boundaries for odd image sizes.
    return [source.resize((224,224),Image.Resampling.BILINEAR,box=(x0*width,y0*height,x1*width,y1*height)) for x0,y0,x1,y1 in QUADRANTS]


def tile_centers():
    return np.asarray([((x1-x0)*(x+.5)/7+x0,(y1-y0)*(y+.5)/7+y0)
        for x0,y0,x1,y1 in QUADRANTS for y in range(7) for x in range(7)],dtype=np.float32)


def resize_global(image):
    return image.resize((224,224),Image.Resampling.BILINEAR)


def estimates(workers):
    if len(workers)!=2 or any(w['status']!='COMPLETE' or w['measured_images']!=128 for w in workers):raise ValueError('Both benchmark workers must complete128 images.')
    start=min(w['measure_start_ns'] for w in workers);end=max(w['measure_end_ns'] for w in workers)
    wall=(end-start)/1e9
    if wall<=0:raise ValueError('Invalid measurement clock.')
    return dict(sample_images=256,sample_views=2048,measured_wall_seconds=wall,images_per_wall_second=256/wall,
        full_images=FULL_IMAGES,full_views=FULL_IMAGES*8,full_tensor_payload_bytes=FULL_IMAGES*PAYLOAD_PER_IMAGE,
        cache_only_linear_wall_hours=wall*FULL_IMAGES/256/3600,
        qualification='Linear projection of this two-GPU reference pipeline; not an ETA guarantee or budget approval. Excludes model download/setup, full-stage I/O, training and caption evaluation.')
