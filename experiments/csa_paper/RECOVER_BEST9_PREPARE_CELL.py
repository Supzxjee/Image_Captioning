# Paste ONE new cell in the CURRENT failed session, then rerun the last execution cell.
import hashlib,json,urllib.request
from pathlib import Path
if not all(n in globals() for n in ('SOURCE','OUTPUT','run')):raise RuntimeError('Use the corrected full notebook in a fresh session; current globals missing.')
SOURCE=Path(SOURCE);OUTPUT=Path(OUTPUT)
if (OUTPUT/'prepared.json').exists() or list(OUTPUT.glob('greedy_rank*.json')):raise RuntimeError('This recovery is only for the missing-review prepare failure; preserve later progress.')
COMMIT='4c680ae60faa1073ddebc2af9b02e9f23bb3f9e6'
HASHES={'run_best9_decode.py': 'fdc6b027ac13d510be72aa25a963eb89e5467e09f179bfaf0756469aef77a648', 'model.py': '139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09', 'real_bank_data.py': '2c791f8dd856198ec1954a2edd1f9e46520d422f20eff07220e462fc2c9bd9a0', 'saved_quality_data.py': 'f8a092506e3adf96292203190ea2f83de397d3028a3ddc6293d58fd360199a1e', 'SAVE_XE139_PACKAGE.json': '452ee35118a3d0285bc7f522bb8c8fc7a0e3e0dac70cac9c2333e7583cd4e423', 'SAVE_FULL_ROI_V134_PACKAGE.json': 'a73636ca75806d6b1d7b4d88f80f3ce1c59b3061eb3e9a5e730097ddd89c505c', 'comparison_baseline4000.json.gz': '8360e82244995afb12fc4eb0ede17d4a092d7f7b846105c7fcd59cb15cc7e046', 'saved_quality_review_text/comparison.json': '43519017eaaf9006264011f9994255d030d817dc9b44d49553ef346f302051e9', 'DECODE_BEST9_PROTOCOL.md': '2e20f183b651f04ec24f2da237a5a9d1e750da937782ac297b6974d53f896b33', 'best9_saved_text.json.gz': '42594e90ec2f47646e7e68347a17471213038bc182814ca9d29ea374e8010861', 'BEST9_SAVED_TEXT_PACKAGE.json': '898d67c3eead66c89f5d2c091b24e5e3eb44d884e86550b2b26f45d6708032e5'}
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{COMMIT}/experiments/csa_paper/{n}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
for name in ('failure.txt','run_status.json','source_receipt.json'):
    p=OUTPUT/name;previous=OUTPUT/('before_prepare_fix_'+name)
    if p.exists() and not previous.exists():p.rename(previous)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('Fixed source and exact saved text ready. Rerun only the last execution cell; do not rerun setup.')
