# Add a new Code cell in the SAME live runtime; no setup/download/tokenization repeat.
import os,sys,json,hashlib,urllib.request,subprocess,time,queue,threading,signal,zipfile,traceback
from pathlib import Path
FIX_REVISION='9a95d2cfe4546438af8621ab7ca06b99d1917b28'
FIX_HASHES={'score_spice_subsets.py': '1899a3f5050e3ae4e7795f748d466dd5e6f5ec4a3694a8c0f44382aedb709825', 'test_spice_subsets_data.py': 'a8db32fa671f18bb058c0bf37d3146b19aac66e005f67fdda90138b61ef8e1b1', 'compare_spice_subsets.py': '2bde2dcef1ab28bd074e9dd912c84bd034158450699a54a39c2f3c4d1aba48b6'}

fix_source=Path(globals().get('SOURCE','/tmp/prompt-test-extra-e4jhdd9w'))
fix_output=Path(globals().get('OUTPUT','/kaggle/working/csa_prompt_TEST_spice_subsets_CONTROL_v1'))
if not (fix_source/'deps').is_dir() or not (fix_output/'environment.json').is_file():
    raise RuntimeError('Old runtime /tmp is gone. Use the fresh V2 notebook; this cell cannot restore the Java parser cache.')
fix_receipt=json.loads((fix_output/'input_receipt.json').read_text())
fix_arm=fix_receipt['arm']
if fix_arm not in ('CONTROL','PROMPT'):raise ValueError('Unknown arm')
for name,digest in FIX_HASHES.items():
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+FIX_REVISION+'/experiments/csa_paper/'+name,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Changed fix source: '+name)
    (fix_source/name).write_bytes(raw)
(fix_output/'subset_fix_receipt.json').write_text(json.dumps(dict(commit=FIX_REVISION,files=FIX_HASHES,original_input=fix_receipt),indent=2))
fix_env=os.environ.copy()
for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):fix_env.pop(key,None)
fix_env.update(PYTHONPATH=str(fix_source/'deps')+os.pathsep+str(fix_source),CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
def run_fix(args,cap):
    start=time.monotonic();beat=start;q=queue.Queue()
    process=subprocess.Popen(list(map(str,args)),env=fix_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    def reader():
        for line in process.stdout:q.put(line)
        q.put(None)
    threading.Thread(target=reader,daemon=True).start()
    try:
        with (fix_output/'recovery.log').open('a',encoding='utf-8') as log:
            while True:
                try:
                    line=q.get(timeout=1)
                    if line is None:break
                    print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                now=time.monotonic()
                if now-start>cap:raise TimeoutError('Recovery phase cap exceeded')
                if now-beat>=30:print('SPICE recovery active;',round(now-start),'s',flush=True);beat=now
        if process.wait():raise subprocess.CalledProcessError(process.returncode,args)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
fix_complete=False
try:
    run_fix([sys.executable,'-u',fix_source/'test_spice_subsets_data.py'],120)
    run_fix([sys.executable,'-u',fix_source/'score_spice_subsets.py','spice','--arm',fix_arm,'--input',fix_output,'--output',fix_output],9000)
    fix_complete=json.loads((fix_output/'result.json').read_text())['status']=='SPICE_SUBSETS_COMPLETE'
except BaseException:
    (fix_output/'recovery_failure.txt').write_text(traceback.format_exc());raise
finally:
    (fix_output/'run_status.json').write_text(json.dumps(dict(complete=fix_complete,arm=fix_arm,training=False,decoding=False,reranking=False)))
    fix_review=fix_output.with_name(fix_output.name+'_recovered_v2_review.zip')
    with zipfile.ZipFile(fix_review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in fix_output.rglob('*'):
            if p.is_file() and 'assets' not in p.relative_to(fix_output).parts and p.suffix in ('.json','.csv','.txt','.log'):
                z.write(p,str(p.relative_to(fix_output)))
    print('SEND REVIEW:',fix_review,flush=True)
    print('SPICE SUBSETS COMPLETE' if fix_complete else 'INCOMPLETE; preserve outputs.',flush=True)
