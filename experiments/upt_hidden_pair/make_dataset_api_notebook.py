"""Build the unexecuted cache upload notebook; no Kaggle calls here."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
KAGGLE_COMMIT = '6894864ff4095cf98e9e36d3a694ff5fc9ba3818'
CREATE_DATASET_SCRIPT = '''import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
result = api.dataset_create_new(sys.argv[1], public=False, quiet=False,
                                convert_to_csv=False, dir_mode='skip')
if result is None or str(getattr(result, 'status', '')).lower() != 'ok' or getattr(result, 'error', None):
    raise RuntimeError('Dataset creation rejected: ' + str(getattr(result, 'error', 'Missing success response')))
print('Dataset creation accepted by API:', result.url, flush=True)
'''
VERSION_PREFLIGHT = '''import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

# Exercise the installed download implementation without contacting Kaggle.
api = KaggleApi()
with tempfile.TemporaryDirectory() as directory, patch.object(api, 'build_kaggle_client') as factory:
    endpoint = factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
    endpoint.return_value = SimpleNamespace(files=[], next_page_token=None, log=None)
    api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/58', path=directory)
    request = endpoint.call_args.args[0]
    assert getattr(request, 'version_label', None) == 'v58', 'Installed CLI ignores the requested version; STOP before download.'
    assert request.user_name == 'ducanh2403'
    assert request.kernel_slug == '2-vi-cai-kia-qua-nhieu'
print('Version routing preflight PASS: request.version_label=v58; no network call.', flush=True)
'''


def build():
    cells = []

    def add(kind, source):
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Lưu UPT hidden-pair cache thành Kaggle Dataset bằng API token

CPU, Internet **ON**. Tự tải đúng **Output version 58** của
`ducanh2403/2-vi-cai-kia-qua-nhieu` bằng API token; không cần gắn Input.
Khóa đúng version 58, không tự lấy latest, không chạy lại notebook nguồn.
Bản CLI 2.2.4 bỏ qua version khi tải Output; notebook cài source chính thức đã sửa,
khóa commit và thử request giả lập phải có `version_label=v58` trước khi tải thật.

1. Lấy token mới tại https://www.kaggle.com/settings/api (không dùng nội dung kaggle.json).
2. Add-ons → Secrets: thêm và bật secret tên **KAGGLE_API_TOKEN**.
3. Sửa OWNER đúng tài khoản sở hữu token. Chạy các cell theo thứ tự.

Chỉ tạo dataset **Private mới**, không sửa/xóa dataset đã có. Nếu slug đã tồn tại,
mở dataset kiểm tra trước; đổi SLUG khi thực sự muốn tạo bản khác.
Token chỉ nằm trong bộ nhớ và môi trường tiến trình con; không ghi vào Output.
Không chạy UPT, caption model hay training.
''')
    add('code', '''from pathlib import Path
KERNEL_VERSION = 'ducanh2403/2-vi-cai-kia-qua-nhieu/58'
OWNER = 'ducanh2403'  # Đổi nếu token thuộc tài khoản khác
SLUG = 'upt-hidden-pairs-caption11k'
TITLE = 'UPT hidden pairs caption 11k'
DESTINATION = Path('/kaggle/working/upt_hidden_pairs_api_dataset')
''')
    add('code', '''import os
import re
import subprocess
import sys
from kaggle_secrets import UserSecretsClient

if not re.fullmatch(r'[a-zA-Z0-9_-]+', OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', SLUG):
    raise ValueError('OWNER/SLUG không hợp lệ.')
if not 3 <= len(SLUG) <= 50 or not 6 <= len(TITLE) <= 50:
    raise ValueError('SLUG cần 3–50 ký tự, TITLE cần 6–50 ký tự.')
try:
    token = UserSecretsClient().get_secret('KAGGLE_API_TOKEN').strip()
    if not token:
        raise ValueError('empty')
except Exception:
    raise RuntimeError('Thêm và bật secret KAGGLE_API_TOKEN tại Add-ons > Secrets. Lấy token tại https://www.kaggle.com/settings/api; bật Internet.') from None

def run_cli(args, authenticated=True):
    environment = os.environ.copy()
    for key in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        environment.pop(key, None)
    if authenticated:
        environment['KAGGLE_API_TOKEN'] = token
    environment['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(args, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line.replace(token, '[REDACTED]'), end='', flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f'Lệnh thất bại (exit {code}); xem log đã ẩn token phía trên. Kiểm tra Internet, token, OWNER và slug đã tồn tại.')

# Install before importing kaggle; no credential files are created.
run_cli([sys.executable, '-m', 'pip', 'install', '--upgrade', '--quiet',
         'https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'], authenticated=False)
print('Secret loaded; CLI ready. Token is not displayed.', flush=True)
''' + '\nVERSION_PREFLIGHT = ' + repr(VERSION_PREFLIGHT)
        + "\nrun_cli([sys.executable, '-u', '-c', VERSION_PREFLIGHT])\n")
    add('code', r'''import tempfile

if KERNEL_VERSION != 'ducanh2403/2-vi-cai-kia-qua-nhieu/58':
    raise ValueError('Notebook này khóa Output version 58; không tự chuyển sang latest.')
# Fresh staging prevents stale files from another version being reused.
DOWNLOAD = Path(tempfile.mkdtemp(prefix='upt-hidden-v58-', dir='/tmp'))
print('Downloading only six cache payload files from:', KERNEL_VERSION, flush=True)
run_cli([sys.executable, '-m', 'kaggle', 'kernels', 'output', KERNEL_VERSION,
         '-p', str(DOWNLOAD), '--file-pattern',
         r'(^|/)(cache_receipt\.json|cohort\.json|features_rank[01]\.h5|pairs_rank[01]\.jsonl)$'])
receipts = list(DOWNLOAD.rglob('cache_receipt.json'))
if len(receipts) != 1:
    raise RuntimeError(f'Version 58 trả về {len(receipts)} cache receipt; dừng, không tự lấy version khác thay thế.')
SOURCE = receipts[0].parent
print('Downloaded version 58. Next: verify receipt and every payload hash.', flush=True)
''')
    source = (ROOT / 'package_cache_dataset.py').read_text(encoding='utf-8')
    add('code', 'import hashlib\nimport json\nSOURCE_CODE = ' + repr(source) + '\n'
        + 'assert hashlib.sha256(SOURCE_CODE.encode()).hexdigest() == ' + repr(hashlib.sha256(source.encode()).hexdigest())
        + '\nexec(compile(SOURCE_CODE, "package_cache_dataset.py", "exec"))\n'
        + 'source = locate(SOURCE)\nreport = package(source, DESTINATION)\n'
        + "report['source_kernel_version'] = KERNEL_VERSION\n"
        + "report['next'] = 'Upload verified flat payload to a private Kaggle Dataset.'\n"
        + "(DESTINATION / 'transport_receipt.json').write_text(json.dumps(report, indent=2) + '\\n', encoding='utf-8')\n"
        + "print('Verified cache ready for API upload.', flush=True)\n")
    add('code', 'CREATE_DATASET_SCRIPT = ' + repr(CREATE_DATASET_SCRIPT) + '\n' + '''metadata_path = DESTINATION / 'dataset-metadata.json'
metadata = {
    'title': TITLE,
    'id': f'{OWNER}/{SLUG}',
    'licenses': [{'name': 'other'}],
    'description': 'Other license: private research cache; original data/model rights and upstream terms retained. Frozen UPT hidden interaction features for caption train10000/dev1000. No original images. See cache receipt for provenance. Not a caption quality result.',
}
if metadata_path.exists():
    raise RuntimeError('Có metadata cũ trong staging; dùng DESTINATION mới để tránh ghi đè.')
# Keep raw HDF5/JSONL unchanged; only the flat verified payload is uploaded.
try:
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    run_cli([sys.executable, '-u', '-c', CREATE_DATASET_SCRIPT, str(DESTINATION)])
finally:
    metadata_path.unlink(missing_ok=True)

print('API confirmed creation; Kaggle may still be processing the dataset.', flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}', flush=True)
run_cli([sys.executable, '-m', 'kaggle', 'datasets', 'status', f'{OWNER}/{SLUG}'])
''')
    add('markdown', '''## Sau upload

Mở link dataset và đợi xử lý hoàn tất, kiểm tra đủ `features_rank0.h5`,
`features_rank1.h5`, `pairs_rank0.jsonl`, `pairs_rank1.jsonl`, `cohort.json`,
`cache_receipt.json` (có thêm transport receipt). Gắn dataset này vào A và H.

Cache nằm **ngay gốc dataset**, vì vậy sửa `HIDDEN_CACHE` theo đường dẫn thực tế, ví dụ:
`/kaggle/input/datasets/ducanh2403/upt-hidden-pairs-caption11k`
(không thêm `/upt_hidden_pairs_11k`). A/H vẫn kiểm tra hash trước training.

Upload được API nhận chưa đồng nghĩa dataset đã xử lý xong. Nếu status chưa ready,
kiểm tra trên trang dataset; không chạy lại cell create liên tục.

Tài liệu chính thức: [token](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication),
[dataset create/status](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python'), kaggle=dict(isInternetEnabled=True, accelerator='none')),
                    nbformat=4, nbformat_minor=5)
    for index, cell in enumerate(cells):
        cell['id'] = f'cache-api-{index}'
    target = ROOT / 'SAVE_UPT_HIDDEN_CACHE_KAGGLE_API.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
