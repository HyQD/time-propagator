import numpy as np

from time_propagator0.stationary_states.projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
)


def laser_pulse(tp):
    return tp.pulses.pulses(tp.r.t)


def expectation_value_hamiltonian(tp):
    return tp.tdcc.compute_energy(tp.r.t, tp.r.y)


def expectation_value_dipole_moment(tp):
    M = tp.system.dipole_moment
    ret = np.empty_like(M[:, 0, 0], dtype=np.complex128)
    ret[0] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[0])
    ret[1] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[1])
    ret[2] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[2])
    return ret


def expectation_value_quadrupole_moment(tp):
    M = tp.system.quadrupole_moment
    ret = np.empty_like(M[:, 0, 0], dtype=np.complex128)
    ret[0] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[0])
    ret[1] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[1])
    ret[2] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[2])
    ret[3] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[3])
    ret[4] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[4])
    ret[5] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[5])
    return ret


def expectation_value_momentum(tp):
    M = tp.system.momentum
    ret = np.empty_like(M[:, 0, 0], dtype=np.complex128)
    ret[0] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[0])
    ret[1] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[1])
    ret[2] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[2])
    return ret


def expectation_value_kinetic_momentum(tp):
    M = tp.system.momentum + vector_potential(tp)
    ret = np.empty_like(M[:, 0, 0], dtype=np.complex128)
    ret[0] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[0])
    ret[1] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[1])
    ret[2] = tp.tdcc.compute_one_body_expectation_value(tp.r.t, tp.r.y, M[2])
    return ret


def CI_projectors(tp):
    sc = tp.states_container
    P = np.empty(sc.n_states)

    for n in np.arange(sc.n_states):
        P[n] = np.abs(tp.tdcc.compute_overlap(tp.r.t, sc.C[:, n], tp.r.y)) ** 2

    return P


def conventional_EOM_projectors(tp):
    sc = tp.states_container

    t, l = tp.cc.get_amplitudes(get_t_0=True).from_array(tp.r.y)

    t0, t1, t2, l1, l2 = sc.t[0][0], sc.t[1], sc.t[2], sc.l[0], sc.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(sc.n_states)

    for n in np.arange(sc.n_states):
        P[n] = conventional_EOM_projector(
            sc.L1[n],
            sc.L2[n],
            l1_t,
            l2_t,
            t0_t,
            t1_t,
            t2_t,
            t1,
            t2,
            sc.R0[n],
            sc.R1[n],
            sc.R2[n],
        )

    return P


def two_component_EOM_projectors(tp):
    sc = tp.states_container

    t, l = tp.cc.get_amplitudes(get_t_0=True).from_array(tp.r.y)

    t0, t1, t2, l1, l2 = sc.t[0][0], sc.t[1], sc.t[2], sc.l[0], sc.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(sc.n_states)

    for n in np.arange(sc.n_states):
        P[n] = two_component_EOM_projector(
            sc.L1[n],
            sc.L2[n],
            l1_t,
            l2_t,
            t0_t,
            t1_t,
            t2_t,
            t1,
            t2,
            sc.R0[n],
            sc.R1[n],
            sc.R2[n],
        )

    return P


def LR_projectors(tp):
    sc = tp.states_container

    t, l = tp.cc.get_amplitudes(get_t_0=True).from_array(tp.r.y)

    t0, t1, t2, l1, l2 = sc.t[0][0], sc.t[1], sc.t[2], sc.l[0], sc.l[1]
    t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

    P = np.zeros(sc.n_states)

    for n in np.arange(sc.n_states):
        P[n] = LR_projector(
            sc.M1[n],
            sc.M2[n],
            sc.L1[n],
            sc.L2[n],
            l1,
            l2,
            l1_t,
            l2_t,
            t1,
            t2,
            t0_t,
            t1_t,
            t2_t,
            sc.R0[n],
            sc.R1[n],
            sc.R2[n],
        )

    return P


def auto_correlation(tp):
    return tp.tdcc.compute_overlap(tp.r.t, tp.y0, tp.r.y)


def compute_F(tp):
    """computes the D_{pq}Z_{pq,j,m}"""
    F = np.zeros((2, 2, tp.pulses.n_pulses), dtype=np.complex128)

    Ru = tp.pulses.Ru
    Iu = tp.pulses.Iu

    Rg = tp.pulses.Rg
    Ig = tp.pulses.Ig

    RuRu = tp.pulses.RuRu
    RuIu = tp.pulses.RuIu
    IuRu = tp.pulses.IuRu
    IuIu = tp.pulses.IuIu

    t = tp.r.t
    y = tp.r.y

    rho_qp = tp.tdcc.compute_one_body_density_matrix(t, y)
    pwi = tp.pwi_container

    cross_terms = tp.inputs("cross_terms")

    # F00,m and F01,m
    for m in np.arange(tp.pulses.n_pulses):
        # cos (i=1)
        Z00_l = -1j * np.tensordot(Ru[m], pwi[f"cosp,{m}"], axes=(0, 0))

        Z00_q = 0
        if cross_terms:
            for n in np.arange(tp.pulses.n_pulses):
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
        else:
            n = m
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
        if cross_terms:
            for n in np.arange(tp.pulses.n_pulses):
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
        else:
            n = m
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


## helper functions
def dipole_vector_potential(tp):
    A = np.zeros((3, tp.system.l, tp.system.l), dtype=np.complex128)

    for i in np.arange(3):
        A[i, :, :] = np.eye(tp.system.l)

    pulse = np.zeros(3)
    for m in np.arange(tp.pulses.n_pulses):
        pulse += tp.pulses.Ru[m, :] * tp.pulses.Rg[m](tp.r.t) + tp.pulses.Iu[
            m, :
        ] * tp.pulses.Ig[m](tp.r.t)

    for i in np.arange(3):
        A[i, :, :] *= pulse[i]

    return A


def plane_wave_vector_potential(tp):
    A = np.zeros((3, tp.system.l, tp.system.l), dtype=np.complex128)
    for m in np.arange(tp.pulses.n_pulses):
        for i in np.arange(3):
            A[i] += tp.pulses.Ru[m, i] * (
                tp.pwi_container[f"cos,{m}"] * tp.pulses.Rg[m](tp.r.t)
                + tp.pwi_container[f"sin,{m}"] * tp.pulses.Ig[m](tp.r.t)
            )
            A[i] += tp.pulses.Iu[m, i] * (
                tp.pwi_container[f"cos,{m}"] * tp.pulses.Ig[m](tp.r.t)
                - tp.pwi_container[f"sin,{m}"] * tp.pulses.Rg[m](tp.r.t)
            )
    return A


def vector_potential(tp):
    if tp.inputs("gauge") == "length":
        return 0
    if tp.inputs("laser_approx") == "dipole":
        return dipole_vector_potential(tp)
    else:
        return plane_wave_vector_potential(tp)
