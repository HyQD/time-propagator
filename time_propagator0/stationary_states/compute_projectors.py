from time_propagator0.stationary_states.helper_functions import compute_R0_

from time_propagator0.stationary_states.projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
)

def compute_conventional_EOM_projector(CCSC,t,l,n):
    """ CCStatesContainer object
        t : t amplitudes
        l : l amplitudes
        n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]
    P = conventional_EOM_projector(
        CCSC.L1[n],
        CCSC.L2[n],
        l1_t,
        l2_t,
        t0_t,
        t1_t,
        t2_t,
        t1,
        t2,
        CCSC.R0[n],
        CCSC.R1[n],
        CCSC.R2[n],
    )

    return P

def compute_two_component_EOM_projector(CCSC,t,l,n):
    """ CCStatesContainer object
        t : t amplitudes
        l : l amplitudes
        n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]
    P = two_component_EOM_projector(
        CCSC.L1[n],
        CCSC.L2[n],
        l1_t,
        l2_t,
        t0_t,
        t1_t,
        t2_t,
        t1,
        t2,
        CCSC.R0[n],
        CCSC.R1[n],
        CCSC.R2[n],
    )

    return P

def compute_LR_projector(CCSC,t,l,n):
    """ CCStatesContainer object
        t : t amplitudes
        l : l amplitudes
        n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]
    P = LR_projector(
        CCSC.M1[n],
        CCSC.M2[n],
        CCSC.L1[n],
        CCSC.L2[n],
        l1,
        l2,
        l1_t,
        l2_t,
        t1,
        t2,
        t0_t,
        t1_t,
        t2_t,
        CCSC.R0[n],
        CCSC.R1[n],
        CCSC.R2[n],
    )

    return P


def compute_R0(CCSC):
    import numpy as np

    R0 = np.empty(CCSC.n_states)
    for n in range(CCSC.n_states):
        t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
        R0[n] = compute_R0_(l1, l2, CCSC.R1[n], CCSC.R2[n])

    return R0
