"""CPU-only synthetic transport/stage joins for30k; no actual features or model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import grounding_pilot_data as data
import package_grounding150 as transport


class Checks(unittest.TestCase):
    def test_transport155_keeps_30k_filename_and_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);download=root/'download';download.mkdir();gold=root/'source/grounding_audit_30k_review_text';gold.mkdir(parents=True)
            for n in transport.names_for(30000):(download/n).write_text('{}\n',encoding='utf-8')
            summary=dict(mapped_mentions=174465,counts=dict(eligible=158197),cohort_sha256=transport.sha(download/'cohort.json'),lexicon_sha256=transport.sha(download/'lexicon.json'))
            raw=json.dumps(summary)+'\n';(download/'summary.json').write_text(raw,encoding='utf-8');(gold/'summary.json').write_text(raw,encoding='utf-8')
            (download/'run_status.json').write_text('{"complete":true}',encoding='utf-8')
            with patch.object(transport,'HERE',root/'source'):transport.package(download,root/'out',study_images=30000,source_version=155)
            receipt=json.loads((root/'out/transport_receipt.json').read_text())
            self.assertTrue(receipt['source_version'].endswith('/155'))
            self.assertIn('train_instances.json',receipt['files']);self.assertNotIn('train10k_instances.json',receipt['files'])
            for n,h in receipt['files'].items():self.assertEqual(transport.sha(root/'out'/n),h)

    def test_unknown_output_version_rejected(self):
        with self.assertRaisesRegex(ValueError,'exact approved'):transport.package('/unused','/unused',study_images=30000,source_version=156)

    def test_prepare30k_schedule_and_label_join_without_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audit=root/'audit';out=root/'out';stage=root/'stage'
            for p in (audit,out,stage):p.mkdir()
            save=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')
            cohort=[dict(coco_id=i,filename=str(i),split='train') for i in range(30000)]
            save(audit/'cohort.json',cohort);save(audit/'lexicon.json',dict(dog=1));save(audit/'source_receipt.json',{})
            save(audit/'train_instances.json',{})
            counts=np.full(30000,5,dtype=np.int64);counts[-76:]=6
            caption_image=np.repeat(np.arange(30000),counts);self.assertEqual(len(caption_image),150076)
            seq=np.zeros(54,dtype=np.int32);seq[:3]=[1,4,2]
            arrays=dict(ids=np.arange(30000),group=np.zeros(30000,dtype=np.uint8),caption_image=caption_image,
                offsets=np.arange(30001),sequences=np.broadcast_to(seq,(150076,54)))
            # Deliberately no features member: this fixture tests joins/schedule, not encoding.
            label=dict(mention_id='0:0:0:1',coco_id=0,phrase='dog',category_id=1,status='eligible',caption_index=0,target_position=0,target_id=4,positive_regions=[0])
            (audit/'mentions.jsonl').write_text(json.dumps(label)+'\n',encoding='utf-8')
            summary=dict(cohort_sha256=data.sha(audit/'cohort.json'),lexicon_sha256=data.sha(audit/'lexicon.json'),mapped_mentions=1,counts=dict(captions=150076,eligible=1))
            save(audit/'summary.json',summary)
            with patch.object(data,'locate_audit',return_value=audit),patch.object(data,'exact_checkpoint',return_value=root/'checkpoint.bin'),patch.object(data,'stage_full',return_value=(arrays,['<pad>','<bos>','<eos>','<unk>','dog'],dict(files_sha256={}))),patch('full_xe_data.validation_references',return_value=[]):
                data.prepare(dict(output=str(out),stage=str(stage),audit_root=str(audit),full_root='fixture',checkpoint_root='fixture',study_images=30000))
            receipt=data.read(out/'pilot_stage.json')
            self.assertEqual(receipt['train_images'],30000);self.assertEqual(receipt['caption_events'],150076)
            self.assertEqual(receipt['image_exposure_histogram'],{'2':26000,'3':4000})
            self.assertEqual(receipt['unique_images_exposed'],30000)
            self.assertEqual(data.read(stage/'grounding_labels.json'),{'0':[[0,[0]]]})


if __name__=='__main__':unittest.main(verbosity=2)
