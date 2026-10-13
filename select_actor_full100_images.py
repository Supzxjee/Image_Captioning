"""Deterministic image metadata selection, independent of detections and scores."""
import argparse,hashlib,json,random,urllib.request
from pathlib import Path


def select_images(caption_images, official_ids, exclusions, count, seed):
    indexed={int(r['cocoid']):r for r in caption_images}
    eligible=sorted(i for i in set(official_ids)-set(exclusions)
                    if i in indexed and indexed[i]['split'] in ('train','restval'))
    if len(eligible)<count:raise ValueError('Insufficient eligible new images; no automatic count change.')
    shuffled=eligible.copy();random.Random(seed).shuffle(shuffled)
    images=[{'image_id':i,'filename':indexed[i]['filename'],'filepath':indexed[i].get('filepath',''),
             'caption_split':indexed[i]['split']} for i in shuffled[:count]]
    return images,eligible


def hashes(path):
    raw=Path(path).read_bytes();lf=raw.replace(b'\r\n',b'\n')
    return {hashlib.sha256(r).hexdigest() for r in (raw,lf,lf.replace(b'\n',b'\r\n'))}


def main():
    p=argparse.ArgumentParser()
    for name in ('caption-json','exclusions','protocol','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();protocol=json.loads(Path(args.protocol).read_text())
    if protocol['exclusions_sha256'] not in hashes(args.exclusions):raise ValueError('Exclusion file changed.')
    raw=Path(args.caption_json).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=protocol['caption_json_sha256']:raise ValueError('Caption metadata changed.')
    url=f"https://raw.githubusercontent.com/s-gupta/v-coco/{protocol['vcoco_revision']}/data/vcoco/vcoco_val.json"
    print('Reading official VCOCOval image IDs:',url,flush=True)
    with urllib.request.urlopen(url,timeout=90) as response:official=response.read()
    annotations=json.loads(official)
    ids={i for action in annotations for i in action['image_id']}
    ex=json.loads(Path(args.exclusions).read_text())['image_ids']
    images,eligible=select_images(json.loads(raw)['images'],ids,ex,protocol['images'],protocol['selection_seed'])
    result={'variant':protocol['variant'],'images':images,'eligible_image_ids':eligible,
        'selection_seed':protocol['selection_seed'],'excluded_image_overlap':0,
        'exclusions_sha256':protocol['exclusions_sha256'],'caption_json_sha256':protocol['caption_json_sha256'],
        'official_vcoco_val_sha256':hashlib.sha256(official).hexdigest(),
        'selection_scope':'100 images new relative to frozen exclusion manifest; official VCOCOval metadata only. No scores, detectors or GT relation labels used for selection.'}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Selected {len(images)} of {len(eligible)} eligible images; excluded overlap0.',flush=True)


if __name__=='__main__':main()
