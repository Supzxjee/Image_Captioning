"""Verify CPU audit output150 and copy byte-for-byte; no model or feature load."""
from pathlib import Path
import shutil
from real_bank_data import read,save,sha,HERE

NAMES=['summary.json','cohort.json','lexicon.json','source_receipt.json','run_status.json',
       'annotation_provenance.json','mentions.jsonl','train10k_instances.json',
       'review_100.html','review_100.json','review_eligible_100.html','review_eligible_100.json']


def package(download,destination):
    expected=sha(HERE/'grounding_audit_v2_review_text/summary.json')
    roots=[p.parent for p in Path(download).rglob('summary.json') if sha(p)==expected]
    if len(roots)!=1:raise ValueError('Output150 must contain one exact reviewed V2 summary; do not substitute latest')
    root=roots[0];summary=read(root/'summary.json');status=read(root/'run_status.json')
    if not status['complete'] or summary['mapped_mentions']!=58575 or summary['counts']['eligible']!=52956:raise ValueError('Wrong/incomplete audit')
    if sha(root/'cohort.json')!=summary['cohort_sha256'] or sha(root/'lexicon.json')!=summary['lexicon_sha256']:raise ValueError('Changed cohort/lexicon')
    for name in NAMES:
        if not (root/name).is_file():raise FileNotFoundError('Full audit output required: '+name)
    destination=Path(destination)
    if destination.exists():raise ValueError('Preserve existing Dataset payload; choose fresh destination')
    destination.mkdir(parents=True)
    hashes={}
    for name in NAMES:
        h=sha(root/name);shutil.copyfile(root/name,destination/name)
        if sha(destination/name)!=h:raise ValueError('Transport copy differs')
        hashes[name]=h
    save(destination/'transport_receipt.json',dict(source_version='ducanh2403/2-vi-cai-kia-qua-nhieu/150',files=hashes,byte_preserving=True,model_executed=False,human_review_pending=True))
    print('Verified full audit150 copied,12 files; no model execution.',flush=True)
