"""Transfer existing labels to class/IoU-matched detector train/dev boxes."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
from select_actor_full100_images import hashes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('detections','images','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--protocol',default='audit/triplet_detector_training_protocol.json')
    args=p.parse_args();protocol=json.loads(Path(args.protocol).read_text())
    if sha(args.detections)!=protocol['detection_cache_sha256']:raise ValueError('Detector snapshot changed.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    combined={'variant':protocol['variant'],'protocol_sha256':sha(args.protocol),'splits':{}}
    matched_splits={}
    for split in ('train','val'):
        source=Path(f'audit/triplet_detector_{split}_reference_pairs.json')
        if protocol[split+'_pairs_sha256'] not in hashes(source):raise ValueError('GT reference changed.')
        command=[sys.executable,'-u','prepare_detector_box_pairs.py','--pairs',str(source),
            '--detections',args.detections,'--images',args.images,'--output',str(output/split),
            '--reference-count',str(protocol[split+'_reference_pairs']),'--max-objects',str(protocol['max_objects']),
            '--min-confidence',str(protocol['min_confidence']),'--min-iou',str(protocol['min_iou']),
            '--coordinate-format',protocol['coordinate_format']]
        subprocess.run(command,check=True)
        rows=json.loads((output/split/'evaluation_pairs.json').read_text())
        original=json.loads(source.read_text());lookup={(r['image_id'],r['subject_id'],r['object_id']):r for r in original}
        for row in rows:
            key=(row['image_id'],row['subject_id'],row['object_id']);reference=lookup[key]
            if any(row[k]!=reference[k] for k in ('labels','filename','subject','object','caption_split')):
                raise ValueError('Matching changed labels or identity.')
            if row['gt_subject_box']!=reference['subject_box'] or row['gt_object_box']!=reference['object_box']:
                raise ValueError('GT coordinates changed.')
        matched_splits[split]=rows
        destination=output/f'{split}_pairs.json';destination.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        stats=json.loads((output/split/'detector_pair_summary.json').read_text())
        combined['splits'][split]=stats
        combined[split]={'pair_json_sha256':sha(destination),'matched_pairs':len(rows),
                         'reference_pairs':len(original),'source_pairs_sha256':protocol[split+'_pairs_sha256']}
    if {r['image_id'] for r in matched_splits['train']}&{r['image_id'] for r in matched_splits['val']}:
        raise ValueError('Train/dev image overlap.')
    combined['train_dev_image_overlap']=0
    combined['notes']='Conditional class/IoU matched supervision; unknown labels preserved. Unmatched proposals receive no negative labels. GT used for identity matching only; detector boxes used for all feature crops. No training or threshold fit.'
    (output/'pair_summary.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    print('DETECTOR TRAIN/DEV PAIRS READY',json.dumps(combined,indent=2),flush=True)


if __name__=='__main__':main()
