"""Small CPU contract checks; no torch/pretrained model imported."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pilot_contract as c


class FakeFile(dict):
    attrs={'complete':True}


class Checks(unittest.TestCase):
    def test_one_pass_main_and_aux_rank_partition(self):
        main,aux=c.schedule()
        self.assertEqual(main.shape,(1250,8));self.assertEqual(aux.shape,(250,4))
        self.assertEqual(sorted(main.ravel().tolist()),list(range(10000)))
        self.assertEqual(sorted(aux.ravel().tolist()),list(range(1000)))
        self.assertFalse(set(main[:,:4].ravel())&set(main[:,4:].ravel()))

    def test_locate_exact_nested_receipt_and_reject_duplicate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);source=root/'nested/cache';source.mkdir(parents=True);(source/'receipt.json').write_bytes(b'right')
            with patch.object(c,'RECEIPT_SHA',c.sha(source/'receipt.json')):
                self.assertEqual(c.locate_cache(root),source)
                (root/'receipt.json').write_bytes(b'wrong');self.assertEqual(c.locate_cache(root),source)
                (root/'receipt.json').write_bytes(b'right')
                with self.assertRaises(ValueError):c.locate_cache(root)

    def test_hash_check_rejects_changed_payload_before_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            for name in ('protocol.json','manifest.json','aux_rank0.h5'):(root/name).write_bytes(b'synthetic')
            receipt=dict(status='PREPARATION_COMPLETE',images=2000,full_pilot_run=False,payload_sha256={n:c.sha(root/n) for n in ('protocol.json','manifest.json','aux_rank0.h5')})
            (root/'receipt.json').write_text(json.dumps(receipt))
            with patch.multiple(c,RECEIPT_SHA=c.sha(root/'receipt.json'),PROTOCOL_SHA=c.sha(root/'protocol.json'),MANIFEST_SHA=c.sha(root/'manifest.json')):
                c.checked_payloads(root);(root/'aux_rank0.h5').write_bytes(b'bad')
                with self.assertRaises(ValueError):c.checked_payloads(root)

    def test_aux_shape_values_and_coco_order(self):
        rows=[dict(coco_id=12),dict(coco_id=24)];f=FakeFile(coco_ids=np.array([12,24]))
        for name,shape in dict(visual_original=(197,768),visual_occluded=(197,768),prompt_original=(20,512),prompt_neutral=(20,512),mask_original=(20,),mask_neutral=(20,)).items():
            f[name]=np.zeros((2,*shape),dtype='u1' if name.startswith('mask') else 'f4')
        c.validate_auxiliary(f,rows)
        f['coco_ids']=np.array([24,12])
        with self.assertRaises(ValueError):c.validate_auxiliary(f,rows)
        f['coco_ids']=np.array([12,24]);f['prompt_neutral'][0,0,0]=np.nan
        with self.assertRaises(ValueError):c.validate_auxiliary(f,rows)


if __name__=='__main__':unittest.main(verbosity=2)
