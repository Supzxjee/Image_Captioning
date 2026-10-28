"""Reuse verified prepared tensors in a fresh attempt; never resume model updates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from readout_data import *

PREPARATION_SOURCES=('experiments/shared_pair_readout/readout_data.py',
    'experiments/shared_pair_readout/prepare_readout.py','experiments/shared_pair_readout/DEV_DIMENSIONS.json',
    'experiments/relation_frame_pretrain/frame_data.py','experiments/relation_frames_inventory/V2_MANIFEST.json.gz',
    'experiments/semantic_grounding_method/data/gain_metadata.json.gz','captioning/visual_cache.py',
    'captioning/semantic_prompts.py','captioning/object_reward.py','captioning/object_consistency.py')
PAYLOADS={'inputs.h5','metadata.json','object_targets.json'}


def reuse(cfg):
    started=time.monotonic();source=Path(cfg['previous_prepared']).resolve();out=Path(cfg['output']).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Use separate non-nested output; preserve old attempt')
    if out.exists():raise FileExistsError('New attempt output already exists')
    old=json.loads((source/'protocol.json').read_bytes());receipt=json.loads((source/'preparation.json').read_bytes())
    if old['policy']!=POLICY or cfg['policy']!=POLICY:raise ValueError('Different preparation policy; no automatic reuse')
    for name in PREPARATION_SOURCES:
        if name not in old['source_assets'] or old['source_assets'][name]!=cfg['source_assets'][name]:raise ValueError('Preparation source changed: '+name)
    expected=dict(cache=FILES,checkpoint=CHECKPOINT_SHA,data=DATA_SHA,detections=DETECTION_SHA,legacy=LEGACY_SHA,dimensions=DIMENSIONS_SHA)
    if receipt['status']!='PREPARATION_COMPLETE' or receipt['input_hashes']!=expected:raise ValueError('Unrecognized preparation receipt')
    if set(receipt['payload_sha256'])!=PAYLOADS or not receipt.get('tensor_content_sha256'):raise ValueError('Preparation payload manifest differs')
    for name,h in receipt['payload_sha256'].items():
        print('Verifying existing preparation:',name,flush=True)
        if not (source/name).is_file() or sha(source/name)!=h:raise ValueError('Changed/missing preparation: '+name)
    out.mkdir(parents=True)
    methods={}
    for name,h in receipt['payload_sha256'].items():
        partial=out/(name+'.partial')
        try:os.link(source/name,partial);methods[name]='hardlink'
        except OSError:shutil.copyfile(source/name,partial);methods[name]='verified_copy'
        if sha(partial)!=h:raise ValueError('Reused payload changed: '+name)
        partial.replace(out/name)
    lineage=dict(source=str(source),source_commit=old['source_commit'],
        original_protocol_sha256=sha(source/'protocol.json'),original_preparation_sha256=sha(source/'preparation.json'),
        methods=methods,new_source_commit=cfg['source_commit'],model_updates_resumed=False)
    updated=dict(receipt,reused_preparation=lineage,original_prepare_seconds=receipt['seconds'],seconds=time.monotonic()-started)
    save(out/'protocol.json',cfg);save(out/'preparation.json',updated);save(out/'reuse_lineage.json',lineage)
    print('Verified stage reused. Old results untouched; new model starts from A.',flush=True)
