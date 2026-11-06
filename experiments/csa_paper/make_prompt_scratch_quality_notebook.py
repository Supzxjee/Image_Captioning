"""Author CPU quality evaluation for sealed saved captions, without API or model calls."""
import argparse
import ast
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['score_prompt_scratch_quality.py','saved_quality_data.py','test_prompt_scratch_quality_data.py',
    'make_prompt_scratch_quality_notebook.py','prompt_scratch_valid5000_saved.json.gz','PROMPT_SCRATCH_QUALITY_INPUTS.json']


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
OUTPUT=Path('/kaggle/working/csa_prompt_scratch_quality_valid5000')
if OUTPUT.exists():raise RuntimeError('Preserve prior output; use fresh path/session')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='prompt-scratch-quality-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
DEADLINE=time.monotonic()+3600 # CPU STOPcap, not ETA.
REVISION={revision!r}
HASHES={hashes!r}
for n,h in HASHES.items():
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/{PREFIX}'+n,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source/payload:'+n)
    (SOURCE/n).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
'''
    runner=RUNNER.replace("print('Running:',","timeout=min(timeout,max(.01,DEADLINE-time.monotonic()))\n    print('Running:',").replace('Smoke active;','CPU saved-caption metrics active;').replace('Smoke wall cap exceeded','CPU metric cap exceeded')
    execute='''complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_scratch_quality_data.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    CHAIR=SOURCE/'chair'
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    if hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    run([sys.executable,'-u',SOURCE/'score_prompt_scratch_quality.py','--payload',SOURCE/'prompt_scratch_valid5000_saved.json.gz',
        '--manifest',SOURCE/'PROMPT_SCRATCH_QUALITY_INPUTS.json','--chair',CHAIR,'--output',OUTPUT],timeout=1800)
    result=json.loads((OUTPUT/'comparison.json').read_text(encoding='utf-8'))
    complete=result['status']=='PROMPT_SCRATCH_QUALITY_COMPLETE' and len(result['metrics'])==3
    if not complete:raise RuntimeError('Incomplete quality evaluation')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,model_executed=False,new_generation=False,test_used=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',REVIEW,flush=True)
    print('QUALITY COMPLETE' if complete else 'INCOMPLETE; preserve diagnostics; no partial-model ranking.',flush=True)
'''
    intro='''# Chấm caption đã lưu — CONTROL10, PROMPT9 và PROMPT10
**Accelerator None, Internet ON. Không cần Add Input hoặc API Secret.** Không model/checkpoint/GPU, không sinh lại caption, không TEST hoặc train.
Đọc bộ caption5000VALID đã khóa bằngSHA từ đúng hai reviewZIP, giữ nguyênreferencePTB của lượttrain. CIDEr phải replay lệch≤1e-7 trước khi nhận kết quả; không nới tolerance. Sau đó tínhBLEU4/ROUGE-L/CHAIRs/CHAIRi/Recall vàpairedCI.
CONTROLbest10 vàPROMPTbest9 đã khóa bằngCIDEr trướcCHAIR. PROMPT10 chỉ là endpointcùng10epoch; không tự chọn lại theoCHAIR. CHAIR chỉ chấm đối tượng, không phải độ đúng quan hệ/thuộc tính. One seed/reusedVALID, chưa là kết quảTEST độc lập.
Chạy Save Version → Save & Run All vớiCPU và gửi `csa_prompt_scratch_quality_valid5000_review.zip`. Cap1h là giới hạn dừng, không phảiETA.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execute)])
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(c['source'])
    path=HERE/'CSA_PROMPT_SCRATCH_QUALITY_VALID5000_CPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size,'bytes; source',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);a=p.parse_args();make(a.revision)
