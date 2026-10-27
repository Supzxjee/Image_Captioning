"""Check case/source/image binding and offline viewer; no model dependencies."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

class Images(HTMLParser):
    def __init__(self):super().__init__();self.sources=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':self.sources.append(attrs['src'])
        if tag=='article':self.ids.append(attrs['id'])

class Checks(unittest.TestCase):
    def test_source_caption_and_reference_join(self):
        native=load(ROOT/'experiments/native_detail_pilot/review/loaded.json')['files']
        original={r['image_id']:r['caption'] for r in json.loads(native['A_predictions.json'])}
        group={r['image_id']:r['coco_id'] for r in json.loads(native['protocol.json'])['groups']['val']}
        dataset=load('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')['images']
        focus=load(HERE/'cases20.json');control=load(HERE/'controls10.json')
        self.assertEqual(len(focus),20);self.assertEqual(len(control),10)
        self.assertEqual(len({r['coco_id'] for r in focus+control}),30)
        for r in focus+control:
            self.assertEqual(r['coco_id'],group[r['image_id']])
            self.assertEqual(r['caption_A'],original[r['image_id']])
            self.assertEqual(r['references'],[s['raw'] for s in dataset[r['image_id']]['sentences']])
        summary=load(HERE/'RESULT.json');self.assertFalse(summary['human_verified']);self.assertFalse(summary['training_labels'])
        self.assertEqual(sum(summary['focus_status_counts'].values()),20)

    def test_image_hash_and_offline_links(self):
        for name,count in [('index.html',20),('controls.html',10)]:
            parser=Images();parser.feed((HERE/'review'/name).read_text(encoding='utf-8'))
            self.assertEqual(len(parser.ids),count);self.assertEqual(len(set(parser.ids)),count)
            self.assertEqual(len(parser.sources),count*2)
            for source in parser.sources:
                p=(HERE/'review'/source).resolve();self.assertTrue(p.is_relative_to((HERE/'review').resolve()));self.assertTrue(p.is_file())
                if 'preview224_' in p.name:
                    with Image.open(p) as image:self.assertEqual(image.size,(224,224))
        for r in load(HERE/'cases20.json')+load(HERE/'controls10.json'):
            raw=(HERE/'review/images'/r['filename']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),r['image']['sha256'])

    def test_correction_not_gold_and_reference_disagreement_retained(self):
        rows={r['coco_id']:r for r in load(HERE/'cases20.json')}
        self.assertEqual(rows[447237]['review']['status'],'under_specified')
        self.assertEqual(rows[261068]['review']['status'],'under_specified')
        self.assertEqual(sum('lemon' in ref.lower() for ref in rows[185901]['references']),4)
        controls={r['coco_id']:r for r in load(HERE/'controls10.json')}
        self.assertEqual(controls[100811]['review']['status'],'acceptable_control')

if __name__=='__main__':unittest.main(verbosity=2)
