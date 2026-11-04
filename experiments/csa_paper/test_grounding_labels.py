"""Small CPU fixtures; no actual features, annotations or models are loaded."""
import ast
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import json
import numpy as np
from grounding_labels import cohort,lexicon,mentions,positive_set,caption_labels
import run_grounding_audit as audit
from run_grounding_audit import review_html,sample


class Checks(unittest.TestCase):
    def setUp(self):
        self.lookup=lexicon([dict(id=1,name='person'),dict(id=2,name='dog'),dict(id=3,name='hot dog'),dict(id=4,name='knife')])
        self.boxes=[[0,0,10,10],[20,0,30,10],[40,0,50,10]]
        self.gt=[dict(category_id=2,bbox=[0,0,10,10],iscrowd=0),dict(category_id=2,bbox=[20,0,10,10],iscrowd=0)]

    def test_train_only_deterministic_cohort(self):
        rows=[dict(cocoid=i,filename=str(i),split=s) for i,s in enumerate(('train','restval','val','test'))]
        self.assertEqual({r['cocoid'] for r in cohort(rows,2)},{0,1})
        self.assertEqual(cohort(rows,2),cohort(rows[::-1],2))

    def test_compound_blocks_dog(self):
        rows=list(mentions('a hot dog and two dogs'.split(),self.lookup))
        self.assertEqual([(r['category_id'],r['target_position']) for r in rows],[(3,2),(2,5)])

    def test_irregular_and_person_alias(self):
        self.assertEqual([r['category_id'] for r in mentions(['women','knives'],self.lookup)],[1,4])

    def test_all_same_class_instances_are_positive(self):
        self.assertEqual(positive_set(self.boxes,['predicted_roi']*3,self.gt,2)[0],[0,1])

    def test_fallback_excluded(self):
        self.assertEqual(positive_set(self.boxes,['full_frame_fallback']*3,self.gt,2)[1],'no_positive')

    def test_crowd_category_skipped(self):
        gt=self.gt+[dict(category_id=2,bbox=[0,0,10,10],iscrowd=1)]
        self.assertEqual(positive_set(self.boxes,['predicted_roi']*3,gt,2)[1],'crowd')

    def test_missing_gt_not_negative(self):
        self.assertEqual(positive_set(self.boxes,['predicted_roi']*3,[],2)[1],'missing_gt')

    def test_degenerate_gt_skips_whole_mention(self):
        for bad in ([0,0,0,10],[0,0,10,0],[0,0,-1,10]):
            gt=self.gt+[dict(id=99,category_id=2,bbox=bad,iscrowd=0)]
            p,status,_=positive_set(self.boxes,['predicted_roi']*3,gt,2)
            self.assertEqual((p,status),([],'invalid_gt'))
            row=list(caption_labels(['dog'],[1,4,2],['<pad>','<bos>','<eos>','<unk>','dog'],self.boxes,['predicted_roi']*3,gt,self.lookup))[0]
            self.assertEqual(row['invalid_gt_annotations'],[dict(id=99,bbox=bad)])

    def test_nonfinite_gt_still_fails(self):
        gt=[dict(category_id=2,bbox=[0,0,float('nan'),10],iscrowd=0)]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):positive_set(self.boxes,['predicted_roi']*3,gt,2)

    def test_prediction_offset(self):
        vocab=['<pad>','<bos>','<eos>','<unk>','a','dog']
        row=list(caption_labels(['a','dog'],[1,4,5,2],vocab,self.boxes,['predicted_roi']*3,self.gt,self.lookup))[0]
        self.assertEqual((row['target_position'],row['target_id'],row['status']),(1,5,'eligible'))

    def test_misaligned_sequence_fails(self):
        with self.assertRaisesRegex(ValueError,'alignment'):
            list(caption_labels(['dog'],[1,4,2],['<pad>','<bos>','<eos>','<unk>','a','dog'],self.boxes,['predicted_roi']*3,self.gt,self.lookup))

    def test_unknown_and_truncated_skip(self):
        v=['<pad>','<bos>','<eos>','<unk>','a']
        self.assertEqual(list(caption_labels(['dog'],[1,3,2],v,self.boxes,['predicted_roi']*3,self.gt,self.lookup))[0]['status'],'unknown_token')
        self.assertEqual(list(caption_labels(['a','dog'],[1,4,2],v,self.boxes,['predicted_roi']*3,self.gt,self.lookup))[0]['status'],'truncated')

    def test_corrupt_boxes_fail(self):
        with self.assertRaises(ValueError):positive_set([[1,1,0,0]],['predicted_roi'],self.gt,2)

    def test_review_has_no_preconfirmed_answers(self):
        page=review_html([dict(mention_id='x',caption='</script>')])
        self.assertNotIn('</script>"',page)
        self.assertIn("localStorage.getItem(key)||'{}'",page)
        self.assertEqual(sample([dict(mention_id='1'),dict(mention_id='2')],'x'),sample([dict(mention_id='2'),dict(mention_id='1')],'x'))

    def test_audit_does_not_access_feature_array(self):
        tree=ast.parse(Path(__file__).with_name('run_grounding_audit.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):self.assertNotEqual(node.slice.value,'features')
        imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) for n in imports))

    def test_complete_fixture_join_without_network_or_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir();roi=root/'roi';roi.mkdir()
            save=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')
            vocab=['<pad>','<bos>','<eos>','<unk>','a','dog'];save(roi/'vocabulary.json',vocab)
            image=dict(cocoid=1,filename='COCO_train2014_000000000001.jpg',split='train',sentences=[dict(raw='a dog',tokens=['a','dog'])])
            save(root/'dataset.json',dict(images=[image]))
            meta=dict(coco_id=1,filename=image['filename'],source_split='train',width=50,height=20,
                      region_kinds=['predicted_roi']*3,captions=['a dog'],sequences=[[1,4,5,2]+[0]*50])
            save(roi/'train_0000.json',dict(images=[meta]))
            # No features key: audit must succeed without opening a feature tensor.
            np.savez(roi/'train_0000.npz',boxes=np.asarray(self.boxes),offsets=[0,3],ids=[1])
            files={n:audit.sha(roi/n) for n in ('train_0000.json','train_0000.npz','vocabulary.json')}
            save(roi/'manifest.json',dict(shards=[dict(group='train',files={k:files[k] for k in ('train_0000.json','train_0000.npz')})]))
            save(roi/'receipt.json',dict(status='fixture'))
            files.update({n:audit.sha(roi/n) for n in ('manifest.json','receipt.json')});save(root/'SAVE_FULL_ROI_V134_PACKAGE.json',dict(files=files))
            gt=({1:dict(id=1,file_name=image['filename'],width=50,height=20)},{1:self.gt},[dict(id=2,name='dog')],dict(fixture=True))
            with patch.object(audit,'HERE',root),patch.object(audit,'DATA_SHA',audit.sha(root/'dataset.json')),patch.object(audit,'cohort',lambda rows:rows),patch.object(audit,'annotations_for',return_value=gt):
                audit.run(dict(output=str(out),work=str(root/'work'),full_root=str(roi),dataset_json=str(root/'dataset.json')))
            summary=audit.read(out/'summary.json');self.assertEqual(summary['counts']['eligible'],1)
            self.assertFalse(summary['training_ready']);self.assertEqual(summary['positive_coverage'],1)
            row=json.loads((out/'mentions.jsonl').read_text());self.assertEqual(row['positive_regions'],[0,1])


if __name__=='__main__':unittest.main(verbosity=2)
