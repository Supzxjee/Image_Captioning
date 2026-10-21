"""Read public reference sources/prototype files and the user PDF. No reference code execution or torch.load."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path,PurePosixPath
import urllib.request

HERE=Path(__file__).resolve().parent
OUT=HERE/'review';OUT.mkdir(exist_ok=True)
REPO='CrossmodalGroup/CSA-Net'


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'caption-research-source-audit'})
    with urllib.request.urlopen(request,timeout=60) as response:return response.read()


def main():
    ref=json.loads(get(f'https://api.github.com/repos/{REPO}/git/ref/heads/main'))
    commit=ref['object']['sha']
    tree=json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))
    (OUT/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    paths=[r['path'] for r in tree['tree'] if r['type']=='blob']
    blobs={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    initial=['README.md','requirements.txt','train_caption.py','models/caption/csa_net.py']
    initial += ['models/caption/__init__.py','models/caption/transformer.py',
        'models/caption/cap_generator.py','models/caption/detector.py','models/caption/base.py',
        'datasets/caption/field.py','engine/caption_engine.py',
        'coco_caption/coco_nouns_emb.pth','coco_caption/coco_verbs_emb.pth']
    initial += [p for p in paths if p.startswith('configs/caption/') and p.endswith(('.yaml','.yml'))]
    def fetch(name):
        if '..' in PurePosixPath(name).parts or name.startswith('/'):raise ValueError(name)
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/{name}'
        existing=OUT/'reference'/name
        raw=existing.read_bytes() if existing.is_file() else None
        if raw is None or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blobs[name]:
            raw=get(url)
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blobs[name]:
            raise ValueError('Reference blob hash differs: '+name)
        return name,url,raw
    manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,url,raw in pool.map(fetch,initial):
            dest=OUT/'reference'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            manifest.append(dict(path=name,url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    paper=Path('C:/KhoaLuanTotNghiep/Paper/Cascade-Semantic-Prompt-Alignment-Network-for-Image-Captioning.pdf')
    from pypdf import PdfReader
    reader=PdfReader(paper)
    pages=[dict(page=i+1,text=p.extract_text()) for i,p in enumerate(reader.pages)]
    (OUT/'paper_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(repo=REPO,commit=commit,tree_complete=not tree.get('truncated',False),files=manifest,
        paper_path=str(paper),paper_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),paper_pages=len(pages),
        status='REFERENCE_SOURCES_AND_PROTOTYPES_NO_CODE_EXECUTION_NO_TORCH_LOAD')
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
