"""Reuse exact126 transport and add verified corrected-word metadata; no API calls."""
import hashlib
import json
import subprocess

from make_smoke_notebook import HERE, ROOT, cell


def make():
    revision = subprocess.check_output(['git', 'rev-parse', '27d884d'], cwd=ROOT, text=True).strip()
    overlay = {}
    for name in ('bank_words_corrected.json', 'correction_receipt.json'):
        path = 'experiments/csa_paper/lemma_correction_review_text/'+name
        payload = subprocess.check_output(['git', 'show', revision+':'+path], cwd=ROOT)
        overlay[name] = dict(url=f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{revision}/{path}', sha256=hashlib.sha256(payload).hexdigest())
    nb = json.loads((HERE/'SAVE_CSA_REAL_PREPARATION_V126_DATASET.ipynb').read_text(encoding='utf-8'))
    nb['cells'][0] = cell('markdown', '''
        # Lưu ROI126 và bank đã sửa lemma thành một Dataset

        **CPU/Accelerator None, Internet ON**, không cần Add Input.
        Bật secret KAGGLE_API_TOKEN; nếu tên secret là codex, sửa SECRET_NAME ở cell đầu.
        OWNER phải có quyền xem Output126 và tạo Dataset; không dán token vào code/chat.

        Tải đúng8file gốc từ version126, kiểm fullSHA cả regions.npz. Thêm bank đã sửa
        và correctionreceipt từ Gitcommit cố định, kiểmSHA và tái tạo bảng theo delta.
        Giữ nguyên mọi file gốc; file để mã hóa CLIP text là **bank_words_corrected.json**.
        Không chạy lại POS/ROI/model và không tạo embeddings ở đây.

        Tạo **Private Dataset mới**, không ghi đè dataset cũ. Nếu slug đã có, kiểm tra
        trangDataset trước khi thử lại. GiữOutput126 đến khi xử lýDataset hoàn tất.
        ''')
    replacements = {
        "SLUG='csa-paper-real-preparation-v126'": "SLUG='csa-real-v126-lemma-fixed'",
        "TITLE='CSA real preparation v126 dataset'": "TITLE='CSA real v126 lemma fixed assets'",
        "DESTINATION=Path('/kaggle/working/csa_real_preparation_dataset')": "DESTINATION=Path('/kaggle/working/csa_real_corrected_dataset')"}
    for old, new in replacements.items():
        assert old in nb['cells'][1]['source']
        nb['cells'][1]['source'] = nb['cells'][1]['source'].replace(old, new)
    nb['cells'][4]['source'] = nb['cells'][4]['source'].replace(
        "print('Eight unchanged files verified/copied. Lexical warning retained:',report['lexical_warning'],flush=True)",
        "print('Eight original126 files verified/copied unchanged; next add corrected words.',flush=True)")
    rule = (HERE/'lemma_correction.py').read_text(encoding='utf-8')
    overlay_code = 'OVERLAY='+repr(overlay)+'\nRULE_SOURCE='+repr(rule)+'\n'
    overlay_code += 'assert hashlib.sha256(RULE_SOURCE.encode()).hexdigest()=='+repr(hashlib.sha256(rule.encode()).hexdigest())+'\n'
    overlay_code += '''
exec(compile(RULE_SOURCE,'lemma_correction.py','exec'))
import urllib.request
for name, item in OVERLAY.items():
    if Path(name).name!=name: raise ValueError('Invalid overlay name')
    payload=urllib.request.urlopen(item['url'],timeout=120).read()
    if hashlib.sha256(payload).hexdigest()!=item['sha256']: raise ValueError('Corrected overlay hash differs: '+name)
    target=DESTINATION/name
    if target.is_symlink() or (target.exists() and target.read_bytes()!=payload): raise ValueError('Changed destination preserved: '+name)
    if not target.exists(): target.write_bytes(payload)
receipt=json.loads((DESTINATION/'correction_receipt.json').read_bytes())
old_words=json.loads((DESTINATION/'bank_words.json').read_bytes())
new_words=json.loads((DESTINATION/'bank_words_corrected.json').read_bytes())
if receipt['status']!='BANK_LEMMA_CORRECTION_COMPLETE' or receipt['preserved_roi_source_version']!=126: raise ValueError('Wrong correction receipt')
if receipt['source_bank_sha256']!=INVENTORY['files']['bank_words.json']: raise ValueError('Corrected bank belongs to another source')
if receipt['corrected_bank_sha256']!=OVERLAY['bank_words_corrected.json']['sha256']: raise ValueError('Receipt/new bank SHA differs')
expected_words, delta=corrected_bank(old_words,receipt['delta']['moved_tokens'])
expected_words['lemma_amendment']['source_bank_sha256']=receipt['source_bank_sha256']
if new_words!=expected_words or delta!=receipt['delta']: raise ValueError('Corrected count/noun policy differs')
for name, expected in INVENTORY['files'].items():
    if sha(DESTINATION/name)!=expected: raise ValueError('Original126 file changed: '+name)
merged=dict(status='ORIGINAL126_AND_CORRECTED_WORDS_READY',original_kernel_version=KERNEL_VERSION,
    original_files_sha256=INVENTORY['files'],corrected_overlay_sha256={n:v['sha256'] for n,v in OVERLAY.items()},
    selected_bank_file='bank_words_corrected.json',selected_bank_sha256=receipt['corrected_bank_sha256'],
    original_files_unchanged=True,semantic_embeddings_created=False,model_executed=False,
    note='Raw bank and historical transport warning preserved; selected bank has the measured narrow lemma correction.')
(DESTINATION/'corrected_assets_receipt.json').write_text(json.dumps(merged,indent=2)+chr(10),encoding='utf-8')
print('Verified corrected bank selected; originals unchanged. No POS/ROI/model rerun.',flush=True)
'''
    nb['cells'].insert(5, cell('code', overlay_code))
    api_cell = nb['cells'][-2]
    phrase = 'Source word list linguistic review pending (rid vs ride)'
    assert phrase in api_cell['source']
    api_cell['source'] = api_cell['source'].replace(phrase,
        'Original word list preserved; separate selected bank has measured riding/rides lemma correction')
    nb['cells'][-1] = cell('markdown', '''
        Gửi path mount thực tế sau khi Dataset xử lý xong và được attach.
        Mặc định dự kiến `/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed`.
        Dataset có8file gốc +transportreceipt +2file sửa +corrected_assets_receipt.
        Loader tiếp theo dùng bank_words_corrected.json; chưa cóCLIPtextembeddings.
        Chỉ được kiểm local metadata/mock/static; chưa chạy API/notebook bởi agent.
        ''')
    target=HERE/'SAVE_CSA_CORRECTED_ASSETS_DATASET.ipynb'
    target.write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    assert target.stat().st_size<1_000_000
    (HERE/'SAVE_CORRECTED_ASSETS_PACKAGE.json').write_text(json.dumps(dict(
        source_kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/126',overlay_commit=revision,overlay=overlay,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),notebook_bytes=target.stat().st_size,
        api_executed=False,model_executed=False),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes;unexecuted; exact126+correctedmetadata')


if __name__=='__main__':
    make()
