"""Local metadata/NumPy fixtures only; Torch tests are separate and user-run."""
import ast
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch
import numpy as np
from grounding_pilot_data import schedule,batch
import package_grounding150 as transport


class Checks(unittest.TestCase):
    def test_fixed_schedule_preserves_events_across_subset_boundary(self):
        groups=[[i*5+j for j in range(5)] for i in range(10000)]+[[50000+i for i in range(22)]]
        a=schedule(groups);b=schedule(groups)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2000,32))
        self.assertEqual(set(a.ravel()[:50022]),set(range(50022)))
        self.assertEqual(len(set(a.ravel())),50022)

    def test_positive_mask_target_offset_and_variable_regions(self):
        data=dict(caption_image=np.array([0,1]),offsets=np.array([0,2,5]),features=np.ones((5,768),np.float16),sequences=np.array([[1,4,5,2,0],[1,6,2,0,0]]))
        x,v,s,p=batch(data,[1,0],{'0':[[1,[0,1]]],'1':[[0,[2]]]})
        self.assertEqual(x.shape,(2,3,768));self.assertEqual(p.shape,(2,3,3))
        self.assertEqual(np.flatnonzero(p[0,0]).tolist(),[2]);self.assertTrue(p[1,1,:2].all())
        self.assertFalse(v[1,2]);self.assertFalse(p[1,:,2].any())

    def test_empty_labels_still_keep_xe_batch(self):
        d=dict(caption_image=np.array([0]),offsets=np.array([0,1]),features=np.ones((1,768),np.float16),sequences=np.array([[1,4,2]]))
        x,v,s,p=batch(d,[0],{});self.assertFalse(p.any());self.assertEqual(s.tolist(),[[1,4,2]])

    def test_no_local_neural_import_in_data(self):
        tree=ast.parse(Path(__file__).with_name('grounding_pilot_data.py').read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,(ast.Import,ast.ImportFrom)):self.assertNotIn('torch',ast.unparse(n))

    def test_model_sources_parse_without_import(self):
        for name in ('grounding_model.py','run_grounding_pilot.py','test_grounding_model.py'):
            ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))

    def test_label_transport_copies_bytes_and_rejects_changed_cohort(self):
        for altered in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);download=root/'download';source=root/'source';gold=source/'grounding_audit_v2_review_text'
                download.mkdir();gold.mkdir(parents=True)
                for name in transport.NAMES:(download/name).write_text('{}\n',encoding='utf-8')
                summary=dict(mapped_mentions=58575,counts=dict(eligible=52956),cohort_sha256=transport.sha(download/'cohort.json'),lexicon_sha256=transport.sha(download/'lexicon.json'))
                payload=json.dumps(summary)+'\n';(download/'summary.json').write_text(payload,encoding='utf-8');(gold/'summary.json').write_text(payload,encoding='utf-8')
                (download/'run_status.json').write_text('{"complete":true}',encoding='utf-8')
                if altered:(download/'cohort.json').write_text('[]\n',encoding='utf-8')
                with patch.object(transport,'HERE',source):
                    if altered:
                        with self.assertRaisesRegex(ValueError,'cohort'):transport.package(download,root/'destination')
                    else:
                        transport.package(download,root/'destination')
                        receipt=json.loads((root/'destination/transport_receipt.json').read_text())
                        self.assertTrue(receipt['source_version'].endswith('/150'))
                        for name in transport.NAMES:self.assertEqual((download/name).read_bytes(),(root/'destination'/name).read_bytes())

    def test_generator_sources_parse(self):
        ast.parse(Path(__file__).with_name('make_grounding_pilot_notebooks.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
