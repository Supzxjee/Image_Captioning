"""Package two unexecuted notebooks; no API or model calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import textwrap
from make_notebooks import cell

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def write(name,cells,gpu):
 for c in cells:
  if c['cell_type']=='code':compile(c['source'],name,'exec')
 nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=gpu,accelerator='gpu' if gpu else 'none')))
 target=HERE/name;target.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if target.stat().st_size>=1000000:raise ValueError('Notebook too large')
 return dict(file=name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())

def save_notebook():
 spec=importlib.util.spec_from_file_location('existing_dataset_api_constants',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
 api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 transport=(HERE/'package_eval_checkpoints.py').read_text(encoding='utf-8')
 cells=[cell('markdown','''
  # Lưu checkpoint C/C0 — đúng version111 và112
  **Accelerator=None · Internet ON · CPU. Không chạy model hoặc notebook nguồn.**
  C từ111, C0 từ112 của ducanh2403/2-vi-cai-kia-qua-nhieu.
  Tải chỉ checkpoint, kiểm fullSHA đã khóa, copy nguyên byte thành C.bin và C0.bin.
  Tạo Dataset Private mới, không có source/cache/500files. Cần khoảng2GiB trống.
  Add-ons → Secrets: bật KAGGLE_API_TOKEN; nếu secret tên codex thì sửa SECRET_NAME.
  Không dán token vào code/chat. OWNER phải đúng tài khoản sở hữu token/quyền output.
  '''),cell('code','''
  from pathlib import Path
  OWNER='ducanh2403'
  SLUG='concept-mass-c-c0-v111-v112'
  TITLE='Concept mass C C0 v111 v112 weights'
  SECRET_NAME='KAGGLE_API_TOKEN'  # Đổi thành 'codex' nếu đó là tên secret của bạn
  DESTINATION=Path('/kaggle/working/concept_mass_C_C0_weights_dataset')
  '''),cell('code','''
  import os,re,subprocess,sys
  from kaggle_secrets import UserSecretsClient
  if not re.fullmatch(r'[a-zA-Z0-9_-]+',OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',SLUG):raise ValueError('Invalid OWNER/SLUG')
  if not 6<=len(TITLE)<=50 or not 6<=len(SLUG)<=50:raise ValueError('TITLE/SLUG need6–50characters')
  try:
      token=UserSecretsClient().get_secret(SECRET_NAME).strip()
      if not token:raise ValueError('empty')
  except Exception:raise RuntimeError('Enable SECRET_NAME in Add-ons > Secrets. Do not paste token in code/chat.') from None
  def run_cli(args,authenticated=True):
      env=os.environ.copy()
      for key in ('KAGGLE_API_TOKEN','KAGGLE_KEY','KAGGLE_USERNAME'):env.pop(key,None)
      if authenticated:env['KAGGLE_API_TOKEN']=token
      env['PYTHONUNBUFFERED']='1'
      with subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace') as process:
          for line in process.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
          code=process.wait()
      if code:raise RuntimeError('Command failed; see redacted log. No latest/version fallback or dataset overwrite.')
  ''')]
 cells[-1]['source']+="\nrun_cli([sys.executable,'-m','pip','install','--upgrade','--force-reinstall','--quiet','https://github.com/Kaggle/kaggle-cli/archive/"+api.KAGGLE_COMMIT+".zip'],authenticated=False)\n"
 for version in [111,112]:
  preflight=api.VERSION_PREFLIGHT.replace('/58','/'+str(version)).replace('v58','v'+str(version))
  cells[-1]['source']+='\nrun_cli([sys.executable,"-u","-c",'+repr(preflight)+'])\n'
 cells.append(cell('code','import hashlib,json\nTRANSPORT_SOURCE='+repr(transport)+'\nassert hashlib.sha256(TRANSPORT_SOURCE.encode()).hexdigest()=='+repr(hashlib.sha256(transport.encode()).hexdigest())+"\nexec(compile(TRANSPORT_SOURCE,'package_eval_checkpoints.py','exec'))\n"))
 cells.append(cell('code',r'''
  import tempfile,shutil
  if DESTINATION.exists():raise FileExistsError('Preserve existing staging; check dataset page before retrying')
  if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*1024**3:raise RuntimeError('Need2GiB free')
  SOURCES={}
  for name,info in EXPECTED.items():
      version=info['version'];kernel='ducanh2403/2-vi-cai-kia-qua-nhieu/'+str(version)
      download=Path(tempfile.mkdtemp(prefix='concept-weights-v'+str(version)+'-',dir='/tmp'))
      pattern=r'(^|/)concept_mass_'+name+r'_v2/checkpoint\.bin$'
      print('Download EXACT checkpoint:',name,kernel,flush=True)
      run_cli([sys.executable,'-m','kaggle','kernels','output',kernel,'-p',str(download),'--file-pattern',pattern])
      matches=[p for p in download.rglob('checkpoint.bin') if digest(p)==info['sha256']]
      if len(matches)!=1:raise RuntimeError(f'Version{version}: {len(matches)} matching checkpoints. STOP; no latest fallback. Check version output files.')
      SOURCES[name]=matches[0]
  receipt=package(SOURCES,DESTINATION)
  print('Verified both full checkpoint hashes; ready for private dataset upload.',flush=True)
  '''))
 # The regex is a Python raw literal: use one backslash before the dot.
 cells[-1]['source']=cells[-1]['source'].replace(r'checkpoint\\.bin',r'checkpoint\.bin')
 cells.append(cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+textwrap.dedent('''
  metadata=DESTINATION/'dataset-metadata.json'
  if metadata.exists():raise FileExistsError('Preserve existing metadata')
  try:
      metadata.write_text(json.dumps(dict(title=TITLE,id=OWNER+'/'+SLUG,licenses=[dict(name='other')],description='Private research trained weights. Exact C endpoint500 Output111 and C0 endpoint500 Output112, byte-preserving bin files. Upstream data/model rights retained. See transport_receipt.json. No new training or inference.'),indent=2),encoding='utf-8')
      run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
  finally:metadata.unlink(missing_ok=True)
  print('Creation accepted; processing may still be pending. Do not repeatedly create.',flush=True)
  print('https://www.kaggle.com/datasets/'+OWNER+'/'+SLUG,flush=True)
  try:run_cli([sys.executable,'-m','kaggle','datasets','status',OWNER+'/'+SLUG])
  except RuntimeError:print('Status not confirmed; check the dataset page before retrying.',flush=True)
  ''')))
 cells.append(cell('markdown','''
  Đợi dataset xử lý xong, có C.bin, C0.bin và transport_receipt.json ở gốc.
  Add dataset vào notebook đánh giá. Default path:
  /kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112
  Giữ weights; notebook này chưa được assistant chạy API.
  '''))
 return write('SAVE_CONCEPT_C_C0_V111_V112_DATASET.ipynb',cells,False)

def eval_notebook(core):
 commit=subprocess.check_output(['git','rev-parse',core],cwd=ROOT,text=True).strip()
 old=json.loads((HERE/'CONCEPT_MASS_C0_2GPU.ipynb').read_text(encoding='utf-8'));prior=json.loads((HERE/'NOTEBOOK_PACKAGE_C0.json').read_text())
 names=set(prior['source_assets'])
 for name in ['expanded_data.py','run_expanded_eval.py','expanded_metrics.py','test_expanded_eval.py','package_eval_checkpoints.py','EXPANDED_EVAL_MANIFEST.json','EXPANDED_EVAL_SENTINELS.json','EXPANDED_EVAL_PROTOCOL.md']:names.add('experiments/concept_mass_grounding/'+name)
 assets={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest() for n in sorted(names)}
 cells=[cell('markdown','''
  # A/C0/C — đánh giá khóa trên val_extra4000_cg · T4×2
  Không training, không rerank, không thêm semantic probe. Cùng4000ảnhval ngoài currentdev1000.
  Đây là validation đã dùng trong lịch sử, không phải test độc lập mới.
  Chạy notebook SAVE_CONCEPT_C_C0_V111_V112_DATASET trước (CPU) để có2checkpoint.
  Add Input: dataset weights vừa tạo, pair-source-fullframe-ce(A), prompt-cache,
  visual-cache và mscoco-2014(vuthetam). Không cần detector/RGB/UPT cache.
  Internet ON, GPU T4×2, Save & Run All. Paths ở cell đầu.
  User báo14hGPU; ETA50–100phút nếuI/O tương tự, parentcap2h20, không guarantee.
  GPU bật trong CPU phases vẫn có thể tính quota. Không tự tăngcap/đổiảnh khi thiếuinput.
  16devsentinels chỉ kiểm parity caption, không thuộc4000ảnhmetrics.
  '''),cell('code','''
  from pathlib import Path
  ARM='EXPANDED_EVAL'
  CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
  WEIGHTS=Path('/kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112')
  CHECKPOINT_PATHS={'A':CHECKPOINT,'C':WEIGHTS/'C.bin','C0':WEIGHTS/'C0.bin'}
  COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
  LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
  VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
  OUTPUT=Path('/kaggle/working/concept_mass_expanded_eval_4000')
  REPORTED_REMAINING_GPU_HOURS=14
  ''')]
 bootstrap=old['cells'][2]['source'].split('\n',2)[2]
 bootstrap=bootstrap.replace('(CHECKPOINT,COCO_JSON,LEGACY_PROMPT,DETECTIONS)','(COCO_JSON,LEGACY_PROMPT,*CHECKPOINT_PATHS.values())')
 bootstrap=bootstrap.replace('def run(args,cwd=None,timeout=900):','def run(args,cwd=None,timeout=900):\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic(),PHASE_DEADLINE-time.monotonic())\n    if timeout<=0:raise TimeoutError("Parent/phase budget exhausted")')
 bootstrap=bootstrap.replace('REPO=Path(tempfile.mkdtemp', 'PARENT_DEADLINE=time.monotonic()+8400;PHASE_DEADLINE=min(PARENT_DEADLINE,time.monotonic()+900)\nREPO=Path(tempfile.mkdtemp')
 cells.append(cell('code','SOURCE_COMMIT='+repr(commit)+'\nSOURCE_ASSETS='+repr(assets)+'\n'+bootstrap))
 setup=old['cells'][3]['source']
 setup=setup.replace("HERE/'concept_data.py'","HERE/'expanded_data.py'").replace('POLICY=policy_module.POLICY','POLICY=policy_module.EVAL_POLICY')
 start=setup.index('    cfg=dict(');end=setup.index('    CONFIG.write_text',start)
 setup=setup[:start]+'''    cfg=dict(checkpoints={k:str(v) for k,v in CHECKPOINT_PATHS.items()},dataset_json=str(COCO_JSON),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),metadata=str(REPO/'experiments/semantic_grounding_method/data/gain_metadata.json.gz'),chair=str(CHAIR),output=str(OUTPUT),policy=POLICY,source_commit=SOURCE_COMMIT,source_assets=SOURCE_ASSETS,reported_remaining_gpu_hours=REPORTED_REMAINING_GPU_HOURS)
'''+setup[end:]
 cells.append(cell('code',setup))
 cells.append(cell('code','''
  success=False
  try:
      PHASE_DEADLINE=min(PARENT_DEADLINE,time.monotonic()+3720)
      run([sys.executable,'-u',HERE/'test_expanded_eval.py'],cwd=REPO,timeout=120)
      run([sys.executable,'-u',REPO/'experiments/shared_pair_readout/metric_preflight.py'],cwd=REPO,timeout=180)
      run([sys.executable,'-u',HERE/'run_expanded_eval.py','prepare','--config',CONFIG],cwd=REPO,timeout=3720)
      PHASE_DEADLINE=min(PARENT_DEADLINE,time.monotonic()+1800)
      run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',HERE/'run_expanded_eval.py','worker','--config',CONFIG],cwd=REPO,timeout=1800)
      PHASE_DEADLINE=min(PARENT_DEADLINE,time.monotonic()+1800)
      run([sys.executable,'-u',HERE/'run_expanded_eval.py','evaluate','--config',CONFIG],cwd=REPO,timeout=1800)
      result=json.loads((OUTPUT/'summary.json').read_text())
      if result['status']!='EXPANDED_EVALUATION_COMPLETE' or result['count']!=4000:raise RuntimeError('Incomplete evaluation')
      success=True
  finally:
      with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
          if LOG.is_file():z.write(LOG,'notebook.log')
          z.write(CONFIG,'notebook_config.json')
          for name in SOURCE_ASSETS:
              if name.endswith('.py'):z.write(REPO/name,'source/'+name)
          if OUTPUT.is_dir():
              for p in OUTPUT.iterdir():
                  if p.suffix in ('.json','.jsonl','.log'):z.write(p,p.name)
          z.writestr('execution_status.json',json.dumps(dict(success=success,source_commit=SOURCE_COMMIT,assistant_launched_job=False,training=False)))
      print('SEND REVIEW ZIP:',REVIEW,flush=True)
      if not success:print('INCOMPLETE: send log/ZIP; do not score a subset as complete.',flush=True)
  if success:print('DONE: locked evaluation only; do not tune on this cohort.',flush=True)
  '''))
 cells.append(cell('markdown','''
  Gửi concept_mass_expanded_eval_4000_review.zip. Không có checkpoint mới vì khôngtraining.
  Báo mọi contrasts/gates đã khóa; nếu không hỗ trợ, dừngclaim đợt này, khôngtuning trên4000ảnh.
  Notebook chưa được assistant chạy GPU/API/model; actual parity/metrics chờ lượt của bạn.
  '''))
 return write('CONCEPT_MASS_EXPANDED_EVAL_4000_2GPU.ipynb',cells,True),dict(source_commit=commit,source_assets=assets)

def main():
 p=argparse.ArgumentParser();p.add_argument('--core',required=True);a=p.parse_args();save=save_notebook();evaluation,source=eval_notebook(a.core)
 report=dict(source,notebooks=[save,evaluation],local_neural_execution=False,kaggle_api_called=False,kaggle_launched=False)
 (HERE/'EXPANDED_NOTEBOOK_PACKAGE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report['notebooks'],indent=2))

if __name__=='__main__':main()
