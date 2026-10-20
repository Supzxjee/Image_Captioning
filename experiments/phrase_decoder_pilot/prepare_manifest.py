import json
from pathlib import Path
from phrase_data import build_manifest,identity

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    bundle=json.loads((HERE.parent/'phrase_pair_feasibility/FEASIBLE_GROUPS_21.json').read_text(encoding='utf-8'))
    m=build_manifest(bundle);m['manifest_content_sha256']=identity(m)
    (HERE/'locked_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(groups=21,fit=14,adaptation_holdout=7,manifest=m['manifest_content_sha256']),indent=2))
