"""NumPy/data-only staging and fixed 64k-event schedule for matched CSA arms."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import hashlib
import numpy as np
from real_bank_data import read,save,sha
from full_cache_data import stage_full,collate

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950'


def exact_checkpoint(root):
    matches=[p for p in Path(root).rglob('checkpoint.bin') if sha(p)==CHECKPOINT_SHA]
    if len(matches)!=1:raise ValueError('Need exact CSA epoch10 checkpoint.bin (not best.bin/epoch15)')
    return matches[0]


def schedule(events_by_image,updates=2000):
    if updates!=2000 or not events_by_image or any(not len(e) for e in events_by_image):raise ValueError('Fixed pilot schedule required')
    result=[];epoch=0
    while len(result)<updates*32:
        order=np.random.default_rng(42+epoch).permutation(len(events_by_image))
        result.extend(e for i in order for e in events_by_image[i]);epoch+=1
    return np.asarray(result[:updates*32],np.int64).reshape(updates,32)


def locate_audit(root):
    expected=sha(HERE/'grounding_audit_v2_review_text/summary.json')
    matches=[p.parent for p in Path(root).rglob('summary.json') if sha(p)==expected]
    if len(matches)!=1:raise ValueError('Attach saved audit V2 output150: expected unique exact summary.json')
    return matches[0]


def prepare(cfg):
    out=Path(cfg['output']);stage=Path(cfg['stage']);audit=locate_audit(cfg['audit_root'])
    for name in ('mentions.jsonl','train10k_instances.json','cohort.json','lexicon.json','source_receipt.json'):
        if not (audit/name).is_file():raise FileNotFoundError('Need entire audit folder, not small review ZIP: '+name)
    summary=read(audit/'summary.json');cohort=read(audit/'cohort.json')
    if sha(audit/'cohort.json')!=summary['cohort_sha256'] or sha(audit/'lexicon.json')!=summary['lexicon_sha256']:raise ValueError('Audit identity differs')
    if len(cohort)!=10000 or {r['split'] for r in cohort}-{'train','restval'}:raise ValueError('Wrong TRAIN population')
    ck=exact_checkpoint(cfg['checkpoint_root']) # check missing input before expensive stage
    data,vocab,stage_receipt=stage_full(cfg['full_root'],stage)
    index={int(i):j for j,i in enumerate(data['ids'])};images=[index[r['coco_id']] for r in cohort]
    if any(data['group'][i]!=0 for i in images):raise ValueError('TRAIN/VALID overlap')
    event_map={i:list(range(*np.searchsorted(data['caption_image'],[i,i+1]).tolist())) for i in images}
    image_events=[event_map[i] for i in images];wanted={r['coco_id'] for r in cohort}
    if sum(map(len,image_events))!=50022:raise ValueError('Audit caption event coverage differs')
    labels=defaultdict(list);seen=set();counts=Counter();lookup=read(audit/'lexicon.json')
    with (audit/'mentions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);key=r['mention_id']
            if key in seen or r['coco_id'] not in wanted:raise ValueError('Duplicate/foreign mention')
            seen.add(key);counts[r['status']]+=1
            if lookup[r['phrase']]!=r['category_id']:raise ValueError('Mention/category mismatch')
            if r['status']!='eligible':continue
            image=index[r['coco_id']];events=event_map[image]
            if not 0<=r['caption_index']<len(events):raise ValueError('Bad reference index')
            event=int(events[r['caption_index']]);position=r['target_position'];regions=r['positive_regions']
            if not 0<=position<52 or int(data['sequences'][event,position+1])!=r['target_id'] or r['target_id']<4:raise ValueError('Grounding next-token mismatch')
            n=int(data['offsets'][image+1]-data['offsets'][image])
            if not regions or len(set(regions))!=len(regions) or any(type(j)!=int or not 0<=j<n for j in regions):raise ValueError('Bad positive region indices')
            labels[str(event)].append([position,regions])
    if len(seen)!=58575 or counts['eligible']!=52956 or any(counts[k]!=summary['counts'].get(k,0) for k in counts):raise ValueError('Incomplete label file')
    for event,rows in labels.items():
        if len({r[0] for r in rows})!=len(rows):raise ValueError('Duplicate target mention')
    batches=schedule(image_events);np.save(stage/'batches.npy',batches)
    save(stage/'grounding_labels.json',dict(labels));save(out/'vocabulary.json',vocab)
    from full_xe_data import validation_references
    refs=validation_references(cfg['full_root']);save(out/'validation_references.json',refs)
    receipt=dict(status='GROUNDING_PILOT_STAGE_COMPLETE',checkpoint=str(ck),checkpoint_sha256=CHECKPOINT_SHA,
        labels_sha256=sha(stage/'grounding_labels.json'),batches_sha256=sha(stage/'batches.npy'),source_stage=stage_receipt['files_sha256'],
        audit_payload_sha256={n:sha(audit/n) for n in ('summary.json','cohort.json','lexicon.json','mentions.jsonl','train10k_instances.json')},
        caption_events=50022,eligible_mentions=52956,updates=2000,events_seen=64000,reviewed_by_human=False)
    save(out/'pilot_stage.json',receipt)
    print('Grounding pilot staged:64,000 matched events/2,000 updates; weak labels unreviewed.',flush=True)


def batch(data,events,labels):
    x,v,seq=collate(data,events)
    positive=np.zeros((len(events),seq.shape[1]-1,v.shape[1]),bool)
    for j,event in enumerate(events):
        for pos,regions in labels.get(str(int(event)),[]):positive[j,pos,regions]=True
    return x,v,seq,positive
