"""Build only metadata from verified local annotations; no ML imports."""
import hashlib,json,math
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent
SOURCE=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=SOURCE/'grounded_links_v1';summary=json.loads((root/'box_join_summary.json').read_text(encoding='utf-8'))
 for name,expected in summary['output_sha256'].items():assert sha(root/name)==expected,name
 instances=SOURCE/'sources/instances_vcoco_trainval2014.json'
 assert sha(instances)=='2b26ef46c5a06201d863e3f888fc9cac28f60b16a39ff9a66364800f94458680'
 data=json.loads(instances.read_text(encoding='utf-8'));cats={r['id']:r['name'] for r in data['categories']};images={r['id']:r for r in data['images']};by=defaultdict(list)
 for a in data['annotations']:by[a['image_id']].append(a)
 fit=json.loads((root/'relation_fit_images.json').read_text(encoding='utf-8'));rows=[]
 for r in fit:
  iid=r['image_id'];im=images[iid];nodes=[];people_complete=True
  for a in by[iid]:
   label=cats[a['category_id']];x,y,w,h=a['bbox'];valid=not a.get('iscrowd',0) and all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0
   if not valid:
    if label=='person':people_complete=False
    continue
   nodes.append([a['id'],label,[round(x/im['width'],7),round(y/im['height'],7),round((x+w)/im['width'],7),round((y+h)/im['height'],7)]])
  ids={n[0] for n in nodes}
  assert all(e['subject_id'] in ids and e['object_id'] in ids for e in r['relations'])
  rows.append(dict(coco_id=iid,filename=r['filename'],filepath=r['filepath'],caption_split=r['caption_split'],instances=nodes,
    actor_inventory_complete=people_complete,positive_edges=[[e['subject_id'],e['predicate'],e['object_id']] for e in r['relations']],
    actor_labels=[[e['subject_id'],e['predicate'],e['label']] for e in r['actor_action_labels']]))
 reserved={split:[r['image_id'] for r in json.loads((root/(split+'_images.json')).read_text(encoding='utf-8'))] for split in ('relation_calibration','relation_dev')}
 assert not ({r['coco_id'] for r in rows}&set(sum(reserved.values(),[])))
 payload=dict(schema='role-interval-fit-metadata-v1',fit=rows,reserved=reserved,source_sha256=dict(summary=sha(root/'box_join_summary.json'),instances=sha(instances),**summary['output_sha256']),
  sources=summary['sources'],scope='holding/riding only; full annotated COCO instance inventory joined; annotation completeness is conditional, not proof all visible people are annotated.')
 text=json.dumps(payload,separators=(',',':'),ensure_ascii=False);(HERE/'role_metadata.json').write_text(text,encoding='utf-8')
 print('Metadata rows',len(rows),'bytes',len(text.encode()),'actors_inventory_complete',sum(r['actor_inventory_complete'] for r in rows))
if __name__=='__main__':main()
