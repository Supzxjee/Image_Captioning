"""Static integration contracts; no local Accelerate/Torch import or model run."""
import ast
from pathlib import Path
from unittest import TestCase, main

HERE = Path(__file__).resolve().parent


class Checks(TestCase):
    def test_explicit_fp32_two_gpu_backward_and_cleanup(self):
        tree = ast.parse((HERE/'run_csa_sharing_benchmark.py').read_text(encoding='utf-8'))
        constructors = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Accelerator']
        self.assertEqual(len(constructors),1)
        keywords = {k.arg:k.value for k in constructors[0].keywords}
        self.assertEqual(ast.literal_eval(keywords['mixed_precision']),'no')
        self.assertEqual(ast.literal_eval(keywords['gradient_accumulation_steps']),1)
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        attrs = [n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
        self.assertNotIn('DistributedDataParallel',attrs); self.assertNotIn('init_process_group',attrs)
        self.assertGreaterEqual(attrs.count('free_memory'),2)
        self.assertIn('clip_grad_norm_',attrs); self.assertIn('end_training',attrs)
        training_backwards = [n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='accelerator' and n.func.attr=='backward']
        self.assertEqual(len(training_backwards),1)
        self.assertIn('2 / int(total)', ast.unparse(training_backwards[0]))

    def test_no_second_data_sharding_or_extra_prepare(self):
        source=(HERE/'run_csa_sharing_benchmark.py').read_text(encoding='utf-8');tree=ast.parse(source)
        prepare=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='prepare']
        self.assertEqual(len(prepare),1)
        self.assertEqual([ast.unparse(n) for n in prepare[0].args],['model','optimizer'])
        self.assertNotIn('prepare_data_loader',source); self.assertNotIn('DataLoader',source)
        self.assertNotIn('model.module',source)
        self.assertIn('local_events(events, rank)',source)

    def test_neural_imports_remain_user_run_and_reference_backward_plain(self):
        tree=ast.parse((HERE/'run_csa_sharing_benchmark.py').read_text(encoding='utf-8'))
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                self.assertNotIn('accelerate',ast.unparse(n)); self.assertNotIn('torch',ast.unparse(n))
        source=ast.unparse(tree)
        self.assertIn('reference = copy.deepcopy(accelerator.unwrap_model(model)).eval()',source)
        self.assertIn('loss.backward()',source)
        self.assertIn("framework='Accelerate'",source)


if __name__ == '__main__': main(verbosity=2)
