from time_propagator0.stationary_states.helper_functions import compute_R0_

from time_propagator0.stationary_states.projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
)


def compute_R0(CCSC):
    import numpy as np

    R0 = np.empty(CCSC.n_states)
    for n in range(CCSC.n_states):
        t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
        R0[n] = compute_R0_(l1, l2, CCSC.R1[n], CCSC.R2[n])

    return R0
