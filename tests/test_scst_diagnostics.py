import json
import os
import random
import unittest
import numpy as np
import torch
from captioning.object_reward import recognized_objects, TrainingObjectReward
from captioning.scst_runtime import configure_strict_runtime, model_digest, tensor_digest, runtime_manifest


class ParserDiagnosisTests(unittest.TestCase):
    def test_two_observed_alias_gaps_fixed_only_in_opt_in_version(self):
        for caption, label in [('a woman putting a turkey into an oven.', 'bird'),
                               ('a cat laying on top of a wooden desk.', 'dining table')]:
            self.assertNotIn(label, recognized_objects(caption))
            self.assertIn(label, recognized_objects(caption, 'aliases_v2'))

    def test_animal_mother_baby_not_blindly_mapped_to_person(self):
        self.assertNotIn('person', recognized_objects('a mother zebra and her baby in a field.', 'aliases_v2'))

    def test_alias_fix_affects_reward_and_legacy_default_remains(self):
        row={'image_id':1,'supported':['person','oven'],'salient':['person','oven'],'detector_neutral':[]}
        legacy=TrainingObjectReward([row]); fixed=TrainingObjectReward([row], 'aliases_v2')
        text=['a woman putting a turkey into an oven.']
        self.assertGreater(float(legacy.score([1],text)[0][0]), float(fixed.score([1],text)[0][0]))

    def test_fraction_dilution_counterexample_is_documented_not_silently_changed(self):
        scorer=TrainingObjectReward([{'image_id':1,'supported':['person','sink'],'salient':['person'],'detector_neutral':[]}])
        scores,terms=scorer.score([1,1],['person motorcycle','person sink motorcycle'])
        self.assertGreater(scores[1],scores[0])
        self.assertEqual(terms[0]['unsupported_classes'],terms[1]['unsupported_classes'])


class RuntimeDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.old={'det':torch.are_deterministic_algorithms_enabled(),
                  'warn':torch.is_deterministic_algorithms_warn_only_enabled(),
                  'bench':torch.backends.cudnn.benchmark,'cdet':torch.backends.cudnn.deterministic,
                  'mha':torch.backends.mha.get_fastpath_enabled(),
                  'flash':torch.backends.cuda.flash_sdp_enabled(),'mem':torch.backends.cuda.mem_efficient_sdp_enabled(),
                  'math':torch.backends.cuda.math_sdp_enabled(),'matmul':torch.backends.cuda.matmul.allow_tf32,
                  'ctf32':torch.backends.cudnn.allow_tf32,'workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
        self.old_cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda,'cudnn_sdp_enabled') else None

    def tearDown(self):
        v=self.old;torch.use_deterministic_algorithms(v['det'],warn_only=v['warn'])
        torch.backends.cudnn.benchmark=v['bench'];torch.backends.cudnn.deterministic=v['cdet']
        torch.backends.mha.set_fastpath_enabled(v['mha'])
        torch.backends.cuda.enable_flash_sdp(v['flash']);torch.backends.cuda.enable_mem_efficient_sdp(v['mem'])
        torch.backends.cuda.enable_math_sdp(v['math']);torch.backends.cuda.matmul.allow_tf32=v['matmul']
        torch.backends.cudnn.allow_tf32=v['ctf32']
        if self.old_cudnn_sdp is not None: torch.backends.cuda.enable_cudnn_sdp(self.old_cudnn_sdp)
        if v['workspace'] is None: os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        else:os.environ['CUBLAS_WORKSPACE_CONFIG']=v['workspace']

    def test_legacy_mode_does_not_change_backends(self):
        configure_strict_runtime(False)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(),self.old['det'])
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),self.old['mha'])

    def test_strict_flags_and_serializable_environment_evidence(self):
        from unittest.mock import patch
        configure_strict_runtime(True)
        # Test metadata without initializing CUDA inside a CPU-only unit test.
        with patch('torch.cuda.is_available', return_value=False): m=runtime_manifest()
        self.assertTrue(m['deterministic_algorithms']);self.assertFalse(m['warn_only'])
        self.assertFalse(m['sdp_flash']);self.assertFalse(m['sdp_mem_efficient']);self.assertTrue(m['sdp_math'])
        self.assertFalse(m['mha_fastpath']);self.assertEqual(m['cublas_workspace'],':4096:8')
        json.dumps(m)

    def test_tensor_identity_detects_dtype_shape_and_value_changes(self):
        a=torch.arange(8)
        self.assertEqual(tensor_digest(a),tensor_digest(a.clone()))
        self.assertNotEqual(tensor_digest(a),tensor_digest(a.reshape(2,4)))
        self.assertNotEqual(tensor_digest(a),tensor_digest(a.float()))
        self.assertNotEqual(tensor_digest(a),tensor_digest(a+1))

    def test_seeded_cpu_sample_and_optimizer_replay_exact(self):
        configure_strict_runtime(True)
        def trial():
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,4))
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3);trace=[]
            for _ in range(32):
                x=torch.randn(3,4);logits=model(x);ids=torch.multinomial(logits.detach().softmax(-1),1)
                loss=-logits.log_softmax(-1).gather(-1,ids).mean()
                opt.zero_grad();loss.backward();opt.step();trace.append(tensor_digest(ids))
            return model_digest(model),trace
        self.assertEqual(trial(),trial())


if __name__=='__main__':unittest.main()
