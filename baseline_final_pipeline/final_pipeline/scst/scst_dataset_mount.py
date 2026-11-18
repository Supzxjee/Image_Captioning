import hashlib, json, shutil, zipfile
from pathlib import Path
from real_bank_data import read, sha

def materialize_part0(dataset, local, expected_parts, expected_ids):
    dataset, local = Path(dataset).resolve(), Path(local)
    manifest = read(dataset/'cache_manifest.json')
    if (manifest['schema']!='csa_scst_full_dataset_v1' or manifest['train_images']!=113287
            or manifest['valid_images']!=5000 or manifest['test_used']
            or manifest['ordered_train_ids_sha256']!=expected_ids or len(manifest['parts'])!=3):
        raise ValueError('Wrong full cache Dataset manifest')
    for part, expected in zip(manifest['parts'],expected_parts,strict=True):
        if any(part.get(key)!=value for key,value in expected.items()):
            raise ValueError('Wrong source part identity in Dataset manifest')
    part = manifest['parts'][0]
    bundle = dataset/'cache_metadata_zip.bin'
    if sha(bundle)!=manifest['bundle_sha256']:
        raise ValueError('Changed cache metadata bundle')
    patch_names = {'train/pilot_patches.npy':'part0_train_pilot_patches.npy',
                   'valid/patches.npy':'part0_valid_patches.npy'}
    links = {}
    with zipfile.ZipFile(bundle) as archive:
        name = 'part0/scst_cache_receipt.json'
        if part['receipt_member']!=name:
            raise ValueError('Unexpected original receipt path')
        raw = archive.read(name)
        if hashlib.sha256(raw).hexdigest()!=expected_parts[0]['receipt_sha256']:
            raise ValueError('Changed original Part 0 receipt')
        files=json.loads(raw)['files_sha256']
        if set(files)!=set(part['files']):
            raise ValueError('Incomplete Part 0 file map')
        local.mkdir()
        (local/'scst_cache_receipt.json').write_bytes(raw)
        for name,digest in files.items():
            entry=part['files'][name]
            target=local/name
            if not target.resolve().is_relative_to(local.resolve()) or entry['sha256']!=digest:
                raise ValueError('Wrong mapped file: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            if name in patch_names:
                if entry.get('file')!=patch_names[name]:
                    raise ValueError('Wrong mounted patch file')
                source=(dataset/patch_names[name]).resolve(strict=True)
                if not source.is_relative_to(dataset) or not source.is_file():
                    raise ValueError('Wrong mounted patch path')
                target.symlink_to(source)
                links[name]=source
            else:
                member='part0/'+name
                if entry.get('bundle')!=bundle.name or entry.get('member')!=member:
                    raise ValueError('Wrong bundled member')
                with archive.open(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source,dest,length=8*1024**2)
    return links
