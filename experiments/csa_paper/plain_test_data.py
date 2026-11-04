"""Exact baseline/cache transport contracts; metadata only, no models or network."""
from pathlib import Path
import shutil
from real_bank_data import HERE,read,save,sha

PLAIN_SHA='a75d7bad5a0d8dc976546b70145a29d76748142ad82190f4f04180418987135f'
PLAIN_VERSION='ducanh3/notebook7bc83c4c76/45'
TEST_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/143'


def exact_file(root,name,digest):
    found=[p for p in sorted(Path(root).rglob(name)) if p.is_file() and sha(p)==digest]
    if not found:raise ValueError('Missing exact version payload: '+name+'; no latest/version substitution')
    return found[0]


def stage_inputs(plain_download,test_download,output,expected=None,plain_sha=PLAIN_SHA):
    expected=expected or read(HERE/'csa10_test_review_text/test_receipt.json');output=Path(output)
    checkpoint=exact_file(plain_download,'checkpoint.bin',plain_sha)
    for name,h in expected['files'].items():
        source=exact_file(test_download,name,h);dest=output/name
        if dest.exists() and sha(dest)!=h:raise ValueError('Preserve changed existing TEST payload: '+name)
        if not dest.exists():shutil.copy2(source,dest)
        if sha(dest)!=h:raise ValueError('TEST copy differs: '+name)
    save(output/'test_receipt.json',expected)
    save(output/'test_lock.json',dict(arm='plain',epoch=10,updates=177110,checkpoint_sha256=plain_sha,checkpoint=str(checkpoint),
        plain_version=PLAIN_VERSION,test_version=TEST_VERSION,beam=5,max_new_tokens=20,selection='exact10 prelocked'))
    return checkpoint
