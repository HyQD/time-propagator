import numpy as np

from time_propagator0.stationary_states.helper_functions import compute_R0_

from time_propagator0.stationary_states.projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
)


def compute_expectation_value(tdcc, t, y, M):
    n = len(M.shape)

    if n == 2:
        return tdcc.compute_one_body_expectation_value(t, y, M)

    ret = np.zeros(n, dtype=np.complex128)
    for i in np.arange(n):
        ret[i] = tdcc.compute_one_body_expectation_value(t, y, M[i])
    return ret


def compute_CI_projectors(CISC, t, y):
    """CIStatesContainer object
    t : t amplitudes
    l : l amplitudes
    n : int, nth excited state
    """
    P = np.zeros(CISC.n_states)

    for n in np.arange(CISC.n_states):
        P[n] = np.abs(self.tdcc.compute_overlap(t, CISC.C[:, n], y)) ** 2

    return P


def compute_conventional_EOM_projectors(CCSC, t, l):
    """CCStatesContainer object
    t : t amplitudes
    l : l amplitudes
    n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(CCSC.n_states)

    for n in np.arange(CCSC.n_states):
        P[n] = conventional_EOM_projector(
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


def compute_two_component_EOM_projectors(CCSC, t, l):
    """CCStatesContainer object
    t : t amplitudes
    l : l amplitudes
    n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(CCSC.n_states)

    for n in np.arange(CCSC.n_states):
        P[n] = two_component_EOM_projector(
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


def compute_LR_projectors(CCSC, t, l):
    """CCStatesContainer object
    t : t amplitudes
    l : l amplitudes
    n : int, nth excited state
    """
    t0, t1, t2, l1, l2 = CCSC.t[0][0], CCSC.t[1], CCSC.t[2], CCSC.l[0], CCSC.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(CCSC.n_states)

    for n in np.arange(CCSC.n_states):
        P[n] = LR_projector(
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


def compute_F(t, rho_qp, pulses, pwi):
    """computes the D_{pq}Z_{pq,j,m}"""
    F = np.zeros((2, 2, pulses.n_pulses), dtype=np.complex128)

    Ru = pulses.Ru
    Iu = pulses.Iu

    Rg = pulses.Rg
    Ig = pulses.Ig

    RuRu = pulses.RuRu
    RuIu = pulses.RuIu
    IuRu = pulses.IuRu
    IuIu = pulses.IuIu

    # F00,m and F01,m
    for m in np.arange(pulses.n_pulses):
        # cos (i=1)
        Z00_l = -1j * np.tensordot(Ru[m], pwi[f"cosp,{m}"], axes=(0, 0))

        Z00_q = 0
        for n in np.arange(pulses.n_pulses):
            Z00_q00 = 0.5 * (pwi[f"cos-,{n}{m}"] + pwi[f"cos+,{n}{m}"]) * Rg[n](t)
            Z00_q11 = 0.5 * (pwi[f"sin+,{n}{m}"] + pwi[f"sin-,{n}{m}"]) * Ig[n](t)
            Z00_q10 = 0.5 * (pwi[f"sin+,{n}{m}"] + pwi[f"sin-,{n}{m}"]) * Rg[n](t)
            Z00_q01 = 0.5 * (pwi[f"cos-,{n}{m}"] + pwi[f"cos+,{n}{m}"]) * Ig[n](t)

            Z00_q += -1j * (
                RuRu(m, n) * Z00_q00
                + RuRu(m, n) * Z00_q11
                + RuIu(m, n) * Z00_q10
                + RuIu(m, n) * Z00_q01
            )

        Z00 = Z00_l + Z00_q
        F[0, 0, m] = np.einsum("qp,pq ->", rho_qp, Z00)

        Z01_l = -1j * np.tensordot(Iu[m], pwi[f"cosp,{m}"], axes=(0, 0))
        Z01_q = 0

        Z01_q += -1j * (
            IuRu(m, n) * Z00_q00
            + IuRu(m, n) * Z00_q11
            + IuIu(m, n) * Z00_q10
            + IuIu(m, n) * Z00_q01
        )

        Z01 = Z01_l + Z01_q
        F[0, 1, m] = np.einsum("qp,pq ->", rho_qp, Z01)

        # F11,m and F10,m
        # sin (i=2)
        Z11_l = -1j * np.tensordot(Ru[m], pwi[f"sinp,{m}"], axes=(0, 0))

        Z11_q = 0
        for n in np.arange(pulses.n_pulses):
            Z11_q00 = 0.5 * (pwi[f"sin+,{n}{m}"] - pwi[f"sin-,{n}{m}"]) * Rg[n](t)
            Z11_q11 = 0.5 * (pwi[f"cos-,{n}{m}"] - pwi[f"cos+,{n}{m}"]) * Ig[n](t)
            Z11_q10 = 0.5 * (pwi[f"cos-,{n}{m}"] - pwi[f"cos+,{n}{m}"]) * Rg[n](t)
            Z11_q01 = 0.5 * (pwi[f"sin+,{n}{m}"] - pwi[f"sin-,{n}{m}"]) * Ig[n](t)

            Z11_q += -1j * (
                RuRu(m, n) * Z11_q00
                + RuRu(m, n) * Z11_q11
                + RuIu(m, n) * Z11_q10
                + RuIu(m, n) * Z11_q01
            )

        Z11 = Z11_l + Z11_q
        F[1, 1, m] = np.einsum("qp,pq ->", rho_qp, Z11)

        Z10_l = -1j * np.tensordot(Iu[m], pwi[f"sinp,{m}"], axes=(0, 0))
        Z10_q = 0

        Z10_q += -1j * (
            IuRu(m, n) * Z11_q00
            + IuRu(m, n) * Z11_q11
            + IuIu(m, n) * Z11_q10
            + IuIu(m, n) * Z11_q01
        )

        Z10 = Z10_l + Z10_q
        F[1, 0, m] = np.einsum("qp,pq ->", rho_qp, Z10)

    return F


def compute_F_dipole(self, t, rho_qp):
    p = self.time_transformed_operator(self.QS.momentum)

    F = np.zeros((1, pulses.n_pulses), dtype=np.complex128)
    for m in np.arange(pulses.n_pulses):
        Z = -1j * np.tensordot(p, self.polarization[m], axes=(0, 0))
        F[:, m] = np.einsum("qp,pq ->", rho_qp, Z)
        if self.inputs("gauge") == "velocity":
            F[:, m] += -1j * self.inputs("n_electrons") * self.Rpulses[m](t)

    return F
