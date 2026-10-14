import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image


class DetectorTrainingPreparationTests(unittest.TestCase):
    def test_explicit_nonlegacy_cohort_preserves_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(100,100)).save(root/'test.jpg')
            row={'image_id':1,'filename':'test.jpg','filepath':'','caption_split':'train',
                 'subject_id':1,'object_id':2,'subject':'person','object':'bicycle',
                 'subject_box':[0,0,.4,1],'object_box':[.3,.3,1,1],'labels':[-1,1,0,0,-1,0]}
            (root/'pairs.json').write_text(json.dumps([row]))
            (root/'cache.json').write_text(json.dumps({'test.jpg':{'objects':[
                {'label':'person','bbox':[1,0,39,100],'conf':.9},
                {'label':'bicycle','bbox':[31,30,100,100],'conf':.9}]}}))
            args=[sys.executable,'prepare_detector_box_pairs.py','--pairs',str(root/'pairs.json'),
                  '--detections',str(root/'cache.json'),'--images',str(root),'--output',str(root/'output')]
            failed=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)  # Default500 cohort remains locked.
            passed=subprocess.run(args+['--reference-count','1','--max-objects','10'],capture_output=True,text=True)
            self.assertEqual(passed.returncode,0,passed.stderr)
            selected=json.loads((root/'output'/'evaluation_pairs.json').read_text())
            self.assertEqual(selected[0]['labels'],row['labels'])
            self.assertEqual(selected[0]['gt_subject_box'],row['subject_box'])
            self.assertEqual(selected[0]['subject_box'],[.01,0,.39,1])
            report=json.loads((root/'output'/'detector_pair_summary.json').read_text())
            self.assertEqual(report['reference_pairs'],1);self.assertEqual(report['max_objects'],10)


if __name__=='__main__':unittest.main()
