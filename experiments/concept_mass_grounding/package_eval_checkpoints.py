"""Copy only two hash-verified checkpoint files into a new flat dataset folder."""
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED={
 'C':dict(version=111,sha256='99f7707f73fe5006592ea9ccbf04bce28a53a65a022cddc5c24ba6a78923c2f4'),
 'C0':dict(version=112,sha256='5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d')}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def package(sources,destination,expected=EXPECTED):
 destination=Path(destination)
 if destination.exists():raise FileExistsError('Preserve existing dataset staging')
 if set(sources)!=set(expected):raise ValueError('Need exactly C and C0 sources')
 for name,p in sources.items():
  if not Path(p).is_file() or digest(p)!=expected[name]['sha256']:raise ValueError('Missing/changed checkpoint '+name)
 destination.mkdir(parents=True);files={}
 for name,p in sources.items():
  target=destination/(name+'.bin');shutil.copyfile(p,target)
  if digest(target)!=expected[name]['sha256']:raise ValueError('Copy changed checkpoint '+name)
  files[target.name]=dict(expected[name],bytes=target.stat().st_size)
 receipt=dict(schema='concept_mass_fixed_checkpoints_v1',source_kernel='ducanh2403/2-vi-cai-kia-qua-nhieu',files=files,byte_preserving=True,model_loaded=False)
 (destination/'transport_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 return receipt
