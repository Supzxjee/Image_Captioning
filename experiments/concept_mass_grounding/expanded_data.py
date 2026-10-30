"""Locked evaluation identities and metadata checks; no neural imports."""
import gzip
import json
from pathlib import Path
from concept_data import sha,save,DATA_SHA,LEGACY_SHA,META_SHA,CHAIR_SHA

HERE=Path(__file__).resolve().parent
MANIFEST_SHA='31559d3f68a133fe9d923489875df21b641b84b8238d3a805ec54de9a99af661'
CHECKPOINTS={
 'A':dict(sha256='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901',digest='552c15d6bd72087328b25ba5852aecbfddd5893d301a7700e2643bc185f82f97'),
 'C':dict(version=111,sha256='99f7707f73fe5006592ea9ccbf04bce28a53a65a022cddc5c24ba6a78923c2f4',digest='0a1cb1b43aee77cf84a24ab97eb0d75a8bfead78128915b56a3efe865a752ab7'),
 'C0':dict(version=112,sha256='5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d',digest='969345610f41fd61efc40db59e7665d85ebb01c746146b2c1997a3e866813ee4')}
EVAL_POLICY=dict(cohort='val_extra4000_cg',count=4000,models=['A','C0','C'],batch_per_rank=8,max_len=30,
 seed=42,bootstrap_draws=20000,bootstrap_seed=20261030,cider_margin=.01,recall_margin=.005,
 setup_seconds=900,prepare_seconds=3600,gpu_seconds=1800,evaluate_seconds=1800,parent_seconds=8400,
 training=False,semantic_diagnostic=False,test_used=False,history='Historically reused validation; post-pilot locked evaluation')

def rows_and_sentinels(manifest,old):
 rows=manifest['rows'];dev=old['groups']['val']
 training=HERE/'TRAIN_PRESENCE.json.gz'
 if sha(training)!='dd888949a1a8cc363efd760eb46012a7a6f736ac6b167d98be5303393b614117':raise ValueError('Wrong TRAIN10k inventory')
 train=json.loads(gzip.decompress(training.read_bytes()))['train_rows']
 if len(train)!=10000:raise ValueError('Wrong TRAIN inventory size')
 ids={r['coco_id'] for r in rows};dev_ids={r['coco_id'] for r in dev}
 if len(rows)!=4000 or len(ids)!=4000 or len(dev_ids)!=1000:raise ValueError('Wrong cohort sizes')
 if ids&dev_ids or ids&{r['coco_id'] for r in train}:raise ValueError('Evaluation/train/dev overlap')
 if any(r['split']!='val' for r in rows):raise ValueError('Non-val evaluation row')
 if len({r['image_id'] for r in rows+dev})!=5000:raise ValueError('Duplicate corpus indices')
 return rows,dev[:16]

def checked_rows(rows,corpus):
 result=[]
 for r in rows:
  original=corpus[r['image_id']]
  if original['filename']!=r['filename'] or original['cocoid']!=r['coco_id'] or original['split']!='val':raise ValueError('Corpus mapping mismatch')
  refs=[s['raw'] for s in original['sentences'][:5]]
  if len(refs)!=5:raise ValueError('Missing references')
  result.append(dict(r,references=refs))
 return result

def validate_predictions(records,rows):
 if len(records)!=len(rows):raise ValueError('Incomplete prediction cohort')
 mapping={r['image_id']:r['coco_id'] for r in rows}
 if len({r['image_id'] for r in records})!=len(records) or {r['image_id']:r['coco_id'] for r in records}!=mapping:raise ValueError('Prediction IDs changed')
 if any(not isinstance(r['caption'],str) for r in records):raise ValueError('Invalid caption')
 return sorted(records,key=lambda r:r['image_id'])

def pinned_metadata(path):
 if sha(path)!=META_SHA:raise ValueError('Wrong original metadata')
 return json.loads(gzip.decompress(Path(path).read_bytes()))
