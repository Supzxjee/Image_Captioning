"""Select failures from saved scores, copy available raw images; no training/inference."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/semantic_evidence_probe'))
from probe_data import digest,identity
from oracle_data import select_cases
from review_html import render


def main():
    source=Path('C:/Users/leduc/Downloads/semantic_evidence_verified_pairs_F_review.zip')
    with zipfile.ZipFile(source) as z:
        read=lambda n:json.loads(z.read(n))
        partitions={key:read('matcher_after_'+key+'.json') for key in ('calibration','probe')}
        cfg=read('protocol.json');cases=select_cases(partitions)
        dataset=json.loads(Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json').read_text(encoding='utf-8'))
        by_id={r['cocoid']:r for r in dataset['images']}
        for row in cases:row['filepath']=by_id[row['coco_id']]['filepath']
        manifest=dict(schema_version=1,cases=cases,cases_sha256=identity(cases),source_review_sha256=digest(source),
            checkpoint_sha256=read('run_summary.json')['checkpoint_sha256'],checkpoint_protocol_sha256=read('run_summary.json')['protocol_sha256'],
            source_checkpoint_sha256=cfg['source_checkpoint_sha256'],cohort_sha256=cfg['cohort_sha256'],
            expected_input_hashes={k:cfg['expected_hashes'][k] for k in ('dataset_json','legacy_prompt','upt_prompt')},
            selection='Post-hoc F boxed wrong-predicate failures; stable hash selection, one case/image, 20 of 21 eligible unique images; NOT representative evaluation',
            no_training=True,test_used=False,gt_scope='Annotation positive versus explicit actor/action zero; verify human plausibility before interpreting semantic correctness',
            conditions=['off','upt','annotation_positive','annotation_negative'])
    (HERE/'cases_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    names={r['filename'] for r in cases};found={}
    listing=subprocess.run(['rg','--files','C:/KhoaLuanTotNghiep/Code/output','-g','*.jpg'],capture_output=True,text=True,check=True).stdout
    for line in listing.splitlines():
        p=Path(line)
        if p.name in names and p.name not in found:found[p.name]=p
    images=HERE/'local_review_images';images.mkdir(exist_ok=True)
    for name,path in found.items():shutil.copy2(path,images/name)
    render([r for r in cases if r['filename'] in found],HERE/'case_review_available.html',image_folder='local_review_images')
    print(json.dumps(dict(cases=len(cases),unique_images=len(names),local_images=len(found),missing_images=sorted(names-found.keys())),indent=2))


if __name__=='__main__':main()
