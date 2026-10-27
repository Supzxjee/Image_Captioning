"""Missing compressed-file compatibility checks; synthetic local files only."""
import contextlib
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from restore_gain_metadata import file_sha,restore_metadata_view

HERE=Path(__file__).resolve().parent


class RestorationChecks(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();pinned=root/'pinned.gz';pinned.write_bytes(b'original-pinned-gzip-bytes')
        h=file_sha(pinned)
        for name in ('inputs.h5','object_targets.json','initial_head.bin'):(source/name).write_bytes(b'unchanged:'+name.encode())
        payload={p.name:file_sha(p) for p in source.iterdir()};payload['gain_metadata.json.gz']=h
        (source/'receipt.json').write_text(json.dumps(dict(status='PREPARATION_COMPLETE',metadata_sha256=h,payload_sha256=payload)))
        return source,pinned,h

    def workspace(self):
        p=HERE/'review';p.mkdir(exist_ok=True);return p

    def test_missing_gzip_restored_without_hdf5_copy_or_receipt_change(self):
        parent=self.workspace()
        with tempfile.TemporaryDirectory(dir=parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(parent.resolve()))
            source,pinned,h=self.fixture(root);before=(source/'receipt.json').read_bytes()
            original_copy=shutil.copyfile
            # Windows sandbox cannot create links; mock only the tiny fixture link.
            with patch('restore_gain_metadata.os.link',side_effect=original_copy) as linked,patch('restore_gain_metadata.shutil.copyfile',wraps=original_copy) as copied:
                view=restore_metadata_view(source,pinned,h,root)
                self.assertEqual(linked.call_count,1)
                self.assertFalse(any(Path(c.args[0]).name=='inputs.h5' for c in copied.call_args_list))
            self.assertEqual((view/'gain_metadata.json.gz').read_bytes(),pinned.read_bytes())
            self.assertFalse((source/'gain_metadata.json.gz').exists())
            self.assertEqual((source/'inputs.h5').read_bytes(),(view/'inputs.h5').read_bytes())
            self.assertEqual((view/'receipt.json').read_bytes(),before)
            self.assertEqual(restore_metadata_view(view,pinned,h,root),view)
            r=json.loads((view/'receipt.json').read_text())
            for name,expected in r['payload_sha256'].items():self.assertEqual(file_sha(view/name),expected)

    def test_wrong_pinned_bytes_and_other_missing_files_stop(self):
        parent=self.workspace()
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(parent.resolve()))
            source,pinned,h=self.fixture(root);pinned.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Pinned'):restore_metadata_view(source,pinned,h,root)
            pinned.write_bytes(b'original-pinned-gzip-bytes');(source/'inputs.h5').unlink()
            with self.assertRaisesRegex(ValueError,'also missing'):restore_metadata_view(source,pinned,h,root)
            self.assertFalse(list(root.glob('grounded-gain-restored-*')))

    def test_existing_corrupt_metadata_not_silently_replaced(self):
        parent=self.workspace()
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(parent.resolve()))
            source,pinned,h=self.fixture(root);(source/'gain_metadata.json.gz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrupt'):restore_metadata_view(source,pinned,h,root)
            self.assertEqual((source/'gain_metadata.json.gz').read_bytes(),b'corrupt')

    def test_directory_in_place_of_gzip_is_recoverable(self):
        parent=self.workspace()
        with tempfile.TemporaryDirectory(dir=parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(parent.resolve()))
            source,pinned,h=self.fixture(root);(source/'gain_metadata.json.gz').mkdir()
            with patch('restore_gain_metadata.os.link',side_effect=OSError('cross-device')),patch('restore_gain_metadata.os.symlink',side_effect=shutil.copyfile) as symlink:
                view=restore_metadata_view(source,pinned,h,root)
                self.assertEqual(symlink.call_count,1)
            self.assertTrue((source/'gain_metadata.json.gz').is_dir())
            self.assertEqual(file_sha(view/'gain_metadata.json.gz'),h)


if __name__=='__main__':unittest.main(verbosity=2)
