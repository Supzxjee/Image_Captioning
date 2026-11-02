import ast
from pathlib import Path
import unittest
from prepare_full_regions import select_full,chunks

class Checks(unittest.TestCase):
    def test_full_split_inventory_excludes_test_and_rejects_duplicate(self):
        rows=[];identity=0
        for split,count in [('train',82783),('restval',30504),('val',5000),('test',5000)]:
            for _ in range(count):
                identity+=1;rows.append(dict(split=split,cocoid=identity,filename=f'COCO_train2014_{identity:012d}.jpg'))
        groups=select_full(dict(images=rows))
        self.assertEqual((len(groups['train']),len(groups['val'])),(113287,5000))
        self.assertFalse(set(r['cocoid'] for r in groups['train'])&set(r['cocoid'] for r in groups['val']))
        self.assertEqual(len(chunks(groups['train']))+len(chunks(groups['val'])),119)
        rows[1]=dict(rows[0])
        with self.assertRaises(ValueError):select_full(dict(images=rows))

    def test_shards_exact_coverage_tail(self):
        rows=list(range(113287));parts=chunks(rows)
        self.assertEqual(len(parts),114);self.assertEqual(len(parts[-1]),287)
        self.assertEqual([i for part in parts for i in part],rows)

    def test_runtime_sources_compile_and_do_not_change_decoding(self):
        root=Path(__file__).parent
        for name in ('prepare_full_regions.py','run_decode_check.py','make_pre_full_notebooks.py'):
            ast.parse((root/name).read_text(encoding='utf-8'))
        tree=ast.parse((root/'prepare_full_regions.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            self.assertFalse(any(x.split('.')[0] in ('torch','transformers','spacy') for x in modules))
        decode=(root/'run_decode_check.py').read_text(encoding='utf-8')
        self.assertIn('beams=1,maximum=20',decode);self.assertIn('beams=5,maximum=20',decode)
        self.assertNotIn('optimizer.step',decode);self.assertIn("bid!=expected_row['ids']",decode)

if __name__=='__main__':unittest.main(verbosity=2)
