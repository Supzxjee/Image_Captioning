"""Verify CPU audit output150 and copy byte-for-byte; no model or feature load."""
from pathlib import Path
import shutil
from real_bank_data import read,save,sha,HERE

NAMES=['summary.json','cohort.json','lexicon.json','source_receipt.json','run_status.json',
       'annotation_provenance.json','mentions.jsonl','train10k_instances.json',
       'review_100.html','review_100.json','review_eligible_100.html','review_eligible_100.json']


def names_for(study_images=10000):
    if study_images not in (10000,30000):raise ValueError('Expected locked10k/30k study')
    return [n if n!='train10k_instances.json' or study_images==10000 else 'train_instances.json' for n in NAMES]


def package(download,destination,study_images=10000,source_version=150):
    if (study_images,source_version) not in ((10000,150),(30000,155)):raise ValueError('Only exact approved output150/155 supported')
    folder='grounding_audit_v2_review_text' if study_images==10000 else 'grounding_audit_30k_review_text'
    expected=sha(HERE/folder/'summary.json');names=names_for(study_images)
    roots=[p.parent for p in Path(download).rglob('summary.json') if sha(p)==expected]
    if len(roots)!=1:raise ValueError('Expected one exact reviewed audit summary; do not substitute latest')
    root=roots[0];summary=read(root/'summary.json');status=read(root/'run_status.json')
    mapped,eligible=(58575,52956) if study_images==10000 else (174465,158197)
    if not status['complete'] or summary['mapped_mentions']!=mapped or summary['counts']['eligible']!=eligible:raise ValueError('Wrong/incomplete audit')
    if sha(root/'cohort.json')!=summary['cohort_sha256'] or sha(root/'lexicon.json')!=summary['lexicon_sha256']:raise ValueError('Changed cohort/lexicon')
    for name in names:
        if not (root/name).is_file():raise FileNotFoundError('Full audit output required: '+name)
    destination=Path(destination)
    if destination.exists():raise ValueError('Preserve existing Dataset payload; choose fresh destination')
    destination.mkdir(parents=True)
    hashes={}
    for name in names:
        h=sha(root/name);shutil.copyfile(root/name,destination/name)
        if sha(destination/name)!=h:raise ValueError('Transport copy differs')
        hashes[name]=h
    save(destination/'transport_receipt.json',dict(source_version='ducanh2403/2-vi-cai-kia-qua-nhieu/'+str(source_version),files=hashes,byte_preserving=True,model_executed=False,human_review_pending=True))
    print('Verified full audit',source_version,'copied,12 files; no model execution.',flush=True)
