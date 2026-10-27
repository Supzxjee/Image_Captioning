from urllib.request import urlopen
import hashlib
helper = urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/f87c864e0a5b34855676b61142d530ec6f2dfca3/experiments/semantic_grounding_method/restore_gain_metadata.py', timeout=60).read()
assert hashlib.sha256(helper).hexdigest() == '22c52ce678c57463891a314a0606f15a07b94ae86ac1eca9b5bbfdab8f09de75'
exec(compile(helper, 'restore_gain_metadata.py', 'exec'))
PREPARED_ROOT = restore_metadata_view(PREPARED_ROOT, METADATA, METADATA_SHA)
from gain_data import checked_cache
checked_cache(PREPARED_ROOT, METADATA_SHA, hash_payloads=False)
print('Metadata ready. Rerun the training cell; no repeat preparation needed.')
