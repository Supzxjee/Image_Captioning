"""Run locally without torch: recipe boundaries and resume identity contract."""
import copy
import math
from phase_policy import ADAM, UPDATES, schedule, contract


def main():
    assert UPDATES == 1250
    first = schedule(1,1)
    assert first['base_lr'] == first['scst_weight'] == 0.
    assert math.isclose(first['core_lr'],1e-6,rel_tol=1e-12)
    assert schedule(1,100)['core_lr'] == schedule(1,1250)['core_lr'] == 1e-4
    assert all(schedule(1,s)['scst_weight'] == 0 for s in range(1,1251))
    assert schedule(2,200)['scst_weight'] == 0 and schedule(2,201)['scst_weight'] == .05
    assert all(schedule(2,s)['core_lr'] == 2e-5 and schedule(2,s)['base_lr'] == 5e-6 for s in range(1,1251))
    for phase,step in ((0,1),(3,1),(1,0),(2,1251)):
        try:
            schedule(phase,step)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid phase/update accepted.')
    cfg = dict(csa_commit='csa',project_commit='base',implementation_sha256={'model.py':'hash'},
               cohort_sha256='cohort',input_sha256={'checkpoint':'weight'},output='/a',stage='/a.h5')
    other = copy.deepcopy(cfg)
    other.update(output='/b',stage='/b.h5')
    assert contract(cfg) == contract(other)
    for key in ('implementation_sha256','input_sha256'):
        changed = copy.deepcopy(cfg)
        changed[key]['different'] = 'hash'
        assert contract(cfg) != contract(changed)
    assert ADAM['weight_decay'] == .01 and ADAM['betas'] == [.9,.999] and ADAM['eps'] == 1e-8
    print('Recipe boundaries + path-independent resume/input-sensitive contract PASS')


if __name__=='__main__':
    main()
