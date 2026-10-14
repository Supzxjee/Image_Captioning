"""Prepare human review of unknown target alternatives; never assign negatives."""
import argparse,hashlib,json,random,urllib.request
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import torch
from captioning.supervised_relations import RELATIONS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--region-directory',required=True)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True);images=output/'images';images.mkdir(exist_ok=True)
    path=Path('audit/interaction_region_train_pairs.json');rows=json.loads(path.read_text());groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['image_id'],row['subject_id'])].append(i)
    candidates=[]
    for indices in groups.values():
        for positive in indices:
            for c,label in enumerate(rows[positive]['labels']):
                if label==1:
                    for alternate in indices:
                        if rows[alternate]['object_id']!=rows[positive]['object_id'] and rows[alternate]['labels'][c]==-1:
                            candidates.append((positive,alternate,c))
    if len(candidates)!=125:raise ValueError('Target-alternative cohort changed.')
    candidates.sort(key=lambda x:(rows[x[0]]['object']!=rows[x[1]]['object'],rows[x[0]]['image_id'],x))
    rng=random.Random(42);public=[];private=[]
    for positive,alternate,c in candidates:
        anchor=rows[positive];other=rows[alternate]
        identity=json.dumps([anchor['image_id'],anchor['subject_id'],anchor['object_id'],other['object_id'],c],separators=(',',':'))
        case='target_'+hashlib.sha256(identity.encode()).hexdigest()[:12]
        order=[positive,alternate];rng.shuffle(order)
        targets=[{'label':rows[index]['object'],'bbox':rows[index]['object_box']} for index in order]
        public.append({'case_id':case,'filename':anchor['filename'],'filepath':anchor['filepath'],
                       'subject_box':anchor['subject_box'],'relation':RELATIONS[c],'targets':targets,
                       'same_class':anchor['object']==other['object']})
        private.append({'case_id':case,'positive_row_index':positive,'alternate_row_index':alternate,
                        'relation_index':c,'positive_side':'A' if order[0]==positive else 'B'})
    directory=Path(args.region_directory)/'validated_source';reported=json.loads((directory/'extraction_summary.json').read_text())
    sizes={}
    for i in range(2):
        path_=directory/f'regions_{i}.pt'
        if sha(path_)!=reported['feature_files_sha256'][path_.name]:raise ValueError('Region metadata source changed.')
        for record in torch.load(path_,map_location='cpu',weights_only=True):
            if record['split']=='train':sizes[rows[record['split_index']]['filename']]=record['image_size']
    def fetch(item):
        filename,filepath=item
        if filepath not in ('train2014','val2014') or Path(filename).name!=filename:raise ValueError('Unexpected image path.')
        target=images/filename;url=f'http://images.cocodataset.org/{filepath}/{filename}'
        if not target.exists():
            with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
            temporary=target.with_suffix('.jpg.partial');temporary.write_bytes(raw);temporary.replace(target)
        with Image.open(target) as image:
            if list(image.size)!=sizes[filename]:raise ValueError('Image size differs from Kaggle extraction: '+filename)
            image.verify()
        return {'filename':filename,'source_url':url,'sha256':sha(target),'image_size':sizes[filename]}
    entries=sorted({(case['filename'],case['filepath']) for case in public})
    with ThreadPoolExecutor(max_workers=4) as pool:
        provenance=[]
        for record in pool.map(fetch,entries):
            provenance.append(record)
            if len(provenance)%10==0:print(f'COCO images verified: {len(provenance)}/{len(entries)}',flush=True)
    for case in public:case['image_size']=sizes[case['filename']]
    manifest={'schema_version':1,'cases':public,'notes':'Training candidates only. Model scores and source labels hidden from UI. No annotation assigned.'}
    (output/'cases.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest_hash=sha(output/'cases.json')
    (output/'private_mapping.json').write_text(json.dumps({'manifest_sha256':manifest_hash,'source_pairs_sha256':sha(Path('audit/interaction_region_train_pairs.json')),'cases':private},indent=2),encoding='utf-8')
    (output/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    template=Path('target_annotation_template.html').read_text(encoding='utf-8')
    payload=json.dumps({'manifest_sha256':manifest_hash,'cases':public},ensure_ascii=False).replace('<','\\u003c')
    (output/'index.html').write_text(template.replace('__CASE_DATA__',payload),encoding='utf-8')
    summary={'cases':len(public),'images':len(entries),'same_class_cases':sum(c['same_class'] for c in public),
             'relation_counts':dict(Counter(c['relation'] for c in public)),'manifest_sha256':manifest_hash,
             'annotation_performed':False,'training_performed':False,'negative_labels_assigned':0,
             'notes':'Official public COCO images, JPEG decoding and dimensions match extraction metadata. Pixel hashes recorded; byte identity with Kaggle images not established.'}
    (output/'preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
