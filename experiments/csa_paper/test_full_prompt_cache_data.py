"""Safe full/pilot cohort, stream, reuse and mocked-encoding fixtures; no neural execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import full_prompt_cache_data as data
from run_prompt_mapping_audit import mapping_cohort
from run_full_prompt_cache import incremental_vectors,cache
from recover_full_prompt_mapping import preserve


def row(i,text='a person',group='train'):
    return dict(coco_id=i,filename=str(i)+'.jpg',group=group,roi_count=1,source_mode='object_only',
        objects=[dict(roi=0,bbox=[0,0,1,1],text=text)],relations=[],counts={})


def vectors(n):
    a=np.zeros((n,512),np.float32)
    for i in range(n):a[i,i]=1
    return a


class Checks(unittest.TestCase):
    def test_full_cohort_includes_all_train_and_default_remains_hash10k(self):
        train=list(range(113287));val=list(range(200000,205000))
        full=mapping_cohort(train,val,'full');pilot=mapping_cohort(train,val)
        self.assertEqual(full['train'],train);self.assertEqual(len(pilot['train']),10000)
        self.assertTrue(set(pilot['train'])<=set(full['train']));self.assertEqual(full['val'],pilot['val'])
        self.assertEqual(full['test'],[])
        with self.assertRaises(ValueError):mapping_cohort(train,train[:5000],'full')
        with self.assertRaises(ValueError):mapping_cohort(train,val,'unknown')

    def fixture(self):
        rows=[row(1),row(2,group='val')];cohort=dict(train=[1],val=[2],test=[])
        summary=dict(counts={g:dict(images=1,kept_objects=1,mapped_relations=0) for g in ('train','val')})
        return rows,cohort,['a person'],summary

    def test_stream_preserves_existing_instances_and_rejects_changed_or_missing(self):
        rows,cohort,texts,summary=self.fixture();old={1:deepcopy(rows[0])}
        r=data.validate_records(iter(rows),cohort,texts,summary,old)
        self.assertEqual(r['old169_records_exactly_preserved'],1)
        rows[0]['objects'][0]['bbox']=[.1,.1,.9,.9]
        with self.assertRaisesRegex(ValueError,'existing169'):data.validate_records(iter(rows),cohort,texts,summary,old)
        with self.assertRaises(ValueError):data.validate_records(iter(rows[1:]),cohort,texts,summary,{})

    def test_inference_payload_rejects_reference_fields_duplicates_and_split_leak(self):
        rows,cohort,texts,summary=self.fixture();rows[0]['captions']=['leaked reference']
        with self.assertRaises(ValueError):data.validate_records(iter(rows),cohort,texts,summary,{})
        rows,cohort,texts,summary=self.fixture()
        with self.assertRaises(ValueError):data.validate_records(iter(rows+[rows[0]]),cohort,texts,summary,{})
        cohort['test']=[3]
        with self.assertRaises(ValueError):data.validate_records(iter(rows),cohort,texts,summary,{})

    def test_incremental_encoding_only_new_strings_preserves_exact_old_vectors(self):
        old=['a person'];a=vectors(1);new_vec=np.zeros((1,512),np.float32);new_vec[0,1]=1
        with patch('run_full_prompt_cache.encode_cpu',return_value=(new_vec,dict(max_tokens=4,seconds=.1))) as encode:
            result,meta=incremental_vectors(['a horse','a person'],old,a,'fixture-hf-cache')
            encode.assert_called_once_with(['a horse'],'fixture-hf-cache')
        np.testing.assert_array_equal(result[1],a[0]);np.testing.assert_array_equal(result[0],new_vec[0])
        self.assertTrue(meta['model_executed'])

    def test_all_vectors_reused_skips_model_and_empty_norm_or_bad_vocabulary_rejected(self):
        a=vectors(1)
        with patch('run_full_prompt_cache.encode_cpu',side_effect=AssertionError('Must not call model')):
            result,meta=incremental_vectors(['a person'],['a person'],a,'unused')
        np.testing.assert_array_equal(result,a);self.assertFalse(meta['model_executed'])
        with self.assertRaises(ValueError):data.assemble(['a horse'],['a person'],a,['a horse'],a)
        with self.assertRaises(ValueError):data.validate_vectors(['a person'],np.zeros((1,512),np.float32))

    def test_payload_coverage_and_path_names_checked(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'review_examples.json').write_bytes(b'fixture')
            receipt=dict(payload_sha256={'review_examples.json':hashlib.sha256(b'fixture').hexdigest()})
            data.checked_payloads(root,receipt);(root/'review_examples.json').unlink()
            with self.assertRaises(ValueError):data.checked_payloads(root,receipt)
            with self.assertRaises(ValueError):data.checked_payloads(root,dict(payload_sha256={'../outside':'x'}))

    def test_cache_receipt_written_only_after_every_payload_copy_no_reencoding(self):
        with tempfile.TemporaryDirectory() as name:
            top=Path(name);mapping=top/'mapping';mapping.mkdir();out=top/'out';out.mkdir()
            for n in data.MAPPING_FILES+('receipt.json',):(mapping/n).write_text('{}',encoding='utf-8')
            manifest=dict(model='fixtureCLIP',revision='fixtureRevision',files_sha256={},max_tokens=3)
            validation=dict(images=118287,old169_records_exactly_preserved=15000)
            cfg=dict(base_root=str(top),mapping_root=str(mapping),output=str(out),hf_cache='unused')
            with patch('run_full_prompt_cache.checked_base',return_value=(top,['a person'],vectors(1),manifest)),patch('run_full_prompt_cache.checked_mapping',return_value=(['a person'],validation)),patch('run_full_prompt_cache.encode_cpu',side_effect=AssertionError('No new text, do not execute model')):
                cache(cfg)
                r=json.loads((out/'cache_receipt.json').read_text(encoding='utf-8'))
                self.assertEqual(r['status'],'FULL_PROMPT_TEXT_CACHE_COMPLETE');self.assertEqual(len(r['payload_sha256']),9)
                data.checked_payloads(out,r)
                (mapping/'review_examples.json').unlink();partial=top/'partial';partial.mkdir();cfg['output']=str(partial)
                with self.assertRaises(FileNotFoundError):cache(cfg)
                self.assertFalse((partial/'cache_receipt.json').exists())

    def test_derived_roundoff_requires_identical_fp32_and_preserves_strict_source_fields(self):
        old=dict(objects=[dict(IoU=.75,roi_bbox=[.1],bbox=[.1],roi=1,confidence=.9)],relations=[dict(geometry=[1.])])
        new=deepcopy(old);new['relations'][0]['geometry'][0]=float(np.nextafter(1.,2.))
        d=data.record_differences(new,old);self.assertEqual(len(d),1);self.assertTrue(d[0]['safe'])
        for path,value in [('bbox',float(np.nextafter(.1,1.))),('roi',True),('confidence',float(np.nextafter(.9,1.)))]:
            new=deepcopy(old);new['objects'][0][path]=[value] if path=='bbox' else value
            self.assertFalse(all(d['safe'] for d in data.record_differences(new,old)))
        new=deepcopy(old);new['relations'][0]['geometry'][0]=1.+1e-5
        self.assertFalse(data.record_differences(new,old)[0]['safe'])
        old['relations'][0]['geometry'][0]=1e-14;new=deepcopy(old);new['relations'][0]['geometry'][0]+=1e-17
        self.assertFalse(data.record_differences(new,old)[0]['safe'])

    def test_diagnostic_reports_all_record_differences_and_stops_semantic_changes(self):
        rows,cohort,texts,summary=self.fixture();rows[0]['objects'][0]['IoU']=.75
        old={1:deepcopy(rows[0])};rows[0]['objects'][0]['IoU']=float(np.nextafter(.75,1.))
        with tempfile.TemporaryDirectory() as name:
            diagnostic=Path(name)/'comparison_diagnostic.json'
            v=data.validate_records(iter(rows),cohort,texts,summary,old,diagnostic)
            self.assertEqual(v['old169_records_exactly_preserved'],0);self.assertEqual(v['old169_records_model_inputs_preserved'],1)
            self.assertEqual(v['old169_records_safe_derived_drift'],1)
            report=json.loads(diagnostic.read_text(encoding='utf-8'));self.assertEqual(report['unsafe_records'],0)
            rows[0]['objects'][0]['text']='different text'
            texts=['a person','different text']
            with self.assertRaisesRegex(ValueError,'existing169'):data.validate_records(iter(rows),cohort,texts,summary,old,diagnostic)
            report=json.loads(diagnostic.read_text(encoding='utf-8'))
            self.assertEqual(report['unsafe_records'],1);self.assertEqual(report['first_unsafe']['coco_id'],1)
            self.assertEqual(report['first_unsafe']['path'],'objects[0].text')

    def test_recovery_checks_original_source_and_every173_payload_preserving_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            top=Path(name);download=top/'download';root=download/'old/mapping';root.mkdir(parents=True)
            for n in data.MAPPING_FILES:(root/n).write_text('{}',encoding='utf-8')
            receipt=dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',payload_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in data.MAPPING_FILES})
            (root/'receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
            source=dict(commit='fixture-source',sha256={'fixture':'a'})
            (download/'source_receipt.json').write_text(json.dumps(source),encoding='utf-8')
            destination=top/'out/mapping';preserve(download,destination,source)
            self.assertTrue(all((destination/n).read_bytes()==(root/n).read_bytes() for n in data.MAPPING_FILES))
            with self.assertRaises(FileExistsError):preserve(download,destination,source)
            with self.assertRaises(ValueError):preserve(download,top/'wrong-source',dict(commit='wrong'))
            (root/'review_examples.json').unlink()
            with self.assertRaises(ValueError):preserve(download,top/'missing-file',source)


if __name__=='__main__':unittest.main(verbosity=2)
