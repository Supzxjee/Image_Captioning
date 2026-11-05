"""Pure metadata fixtures; never import or execute real Torch."""
import unittest
from cuda_runtime_guard import validate_runtime


class Checks(unittest.TestCase):
    def fixture(self):return dict(cuda_build='fixture12',set_device_symbol=True,cuda_available=True,device_count=2,nccl_available=True)

    def test_cuda_pair_allowed(self):validate_runtime(self.fixture())

    def test_cpu_binary_rejected_with_actionable_error(self):
        for key,value in (('cuda_build',None),('set_device_symbol',False)):
            info=self.fixture();info[key]=value
            with self.assertRaisesRegex(RuntimeError,'lacks CUDA support'):validate_runtime(info)

    def test_invisible_wrong_gpu_count_or_no_nccl_rejected(self):
        for key,value in (('cuda_available',False),('device_count',0),('device_count',1),('nccl_available',False)):
            info=self.fixture();info[key]=value
            with self.assertRaises(RuntimeError):validate_runtime(info)


if __name__=='__main__':unittest.main(verbosity=2)
