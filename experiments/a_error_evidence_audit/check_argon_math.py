"""Small NumPy counterexamples to design guarantees; no neural-model execution."""
import json
import math
from pathlib import Path
import numpy as np

def softmax(x):
    z=np.exp(x-np.max(x));return z/z.sum()

def main():
    memory=np.eye(2)
    read1=softmax(memory@np.array([4.,0.]))@memory
    read2=softmax(memory@np.array([0.,4.]))@memory
    assert not np.allclose(read1,read2)
    e=np.array([2.,3.]);m=np.array([2.,0.]);residual=e-(e@m)/(m@m)*m
    assert abs(residual@m)<1e-12 and np.linalg.norm(residual)>0
    # Full memory spans both axes: orthogonal to ONE vector is not outside memory.
    np.testing.assert_allclose(np.linalg.lstsq(memory.T,residual,rcond=None)[0]@memory,residual)
    # The proposal explicitly includes b_f; GELU(1) != 0 even when evidence is 0.
    gelu_bias=.5*(1.+math.erf(1./math.sqrt(2.)))
    assert gelu_bias>0
    raw_margin_difference=2.-1.
    gamma_negative_actual=math.tanh(-1.)*raw_margin_difference
    varying_gate_actual=.1*2.-.9*1.
    assert raw_margin_difference>0 and gamma_negative_actual<0 and varying_gate_actual<0
    result=dict(scope='Mathematical counterexamples only, not model/feature measurements.',
        fixed_memory_dynamic_attention=dict(first_read=read1.tolist(),second_read=read2.tolist()),
        single_vector_orthogonality_not_information_removal=dict(residual=residual.tolist(),still_in_full_memory_span=True),
        zero_evidence_nonzero_MLP_with_explicit_bias=gelu_bias,
        positive_raw_margin_but_negative_final_contribution=dict(raw_difference=raw_margin_difference,negative_gamma=gamma_negative_actual,different_gates=varying_gate_actual),
        matrix_weights_if_fusion_hidden128=512*128+768*128*2+512*128+256*128+128*512,
        local_neural_execution=False)
    Path(__file__).with_name('ARGON_MATH_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
