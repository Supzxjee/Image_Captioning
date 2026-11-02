import ast
from pathlib import Path
import unittest
import numpy as np
from full_cache_data import collate
from sharing_data import grouped_order,rank_events,shared_collate,ddp_loss_multiplier

class Checks(unittest.TestCase):
    def fixture(self):
        return dict(features=np.arange(6*768,dtype=np.float32).reshape(6,768).astype(np.float16),offsets=np.array([0,1,5,6]),
            caption_image=np.array([0,0,0,1,1,2]),group=np.array([0,0,1]),
            sequences=np.array([[1,4,2]+[0]*51,[1,5,6,2]+[0]*50,[1,7,2]+[0]*51,[1,8,2]+[0]*51,[1,9,10,2]+[0]*50,[1,4,2]+[0]*51]))

    def test_grouped_order_covers_train_once_and_keeps_image_refs_contiguous(self):
        data=self.fixture();order=grouped_order(data,5)
        self.assertEqual(sorted(order.tolist()),list(range(5)));self.assertEqual(order.tolist(),grouped_order(data,5).tolist())
        images=data['caption_image'][order];self.assertEqual(len(np.flatnonzero(np.diff(images))),1)
        with self.assertRaises(ValueError):grouped_order(data,6)

    def test_unique_encoder_rows_expand_to_same_data_and_sum_caption_gradients(self):
        data=self.fixture();indices=[1,4,0,3,2];x,v,s=collate(data,indices);u,uv,us,inverse=shared_collate(data,indices)
        self.assertEqual(len(u),2);np.testing.assert_array_equal(u[inverse],x);np.testing.assert_array_equal(uv[inverse],v);np.testing.assert_array_equal(us,s)
        caption_grad=np.arange(1,6,dtype=float);image_grad=np.zeros(2);np.add.at(image_grad,inverse,caption_grad)
        self.assertEqual(image_grad.sum(),caption_grad.sum());self.assertEqual(image_grad.tolist(),[9,6])

    def test_ddp_weighting_with_unequal_tokens_and_tail_partition(self):
        local_grad=np.array([4.,30.]);tokens=np.array([2,6])
        averaged=(local_grad*ddp_loss_multiplier(2,int(tokens.sum()))).mean()
        self.assertAlmostEqual(averaged,local_grad.sum()/tokens.sum())
        self.assertNotAlmostEqual(averaged,float((local_grad/tokens).mean()))
        tail=np.arange(27);a,b=rank_events(tail,0),rank_events(tail,1)
        self.assertEqual((len(a),len(b)),(16,11));np.testing.assert_array_equal(np.r_[a,b],tail)
        with self.assertRaises(ValueError):ddp_loss_multiplier(2,0)

    def test_runtime_sources_compile_and_forward_keeps_graph(self):
        root=Path(__file__).parent
        for n in ('sharing_data.py','run_sharing_benchmark.py','make_sharing_benchmark_notebook.py'):ast.parse((root/n).read_text(encoding='utf-8'))
        tree=ast.parse((root/'run_sharing_benchmark.py').read_text(encoding='utf-8'))
        reader=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Reader')
        forward=next(n for n in reader.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='detach' for n in ast.walk(forward)))

if __name__=='__main__':unittest.main(verbosity=2)
