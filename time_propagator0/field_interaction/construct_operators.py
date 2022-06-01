from quantum_systems.time_evolution_operators import CustomOneBodyOperator
import numpy as np


def construct_linear_dipole_operators(QS, pulses, gauge):
    dipole_operator = QS.position if gauge == "length" else QS.momentum
    Ru = pulses.Ru
    Rg = pulses.Rg
    Iu = pulses.Iu
    Ig = pulses.Ig

    operators = []

    for m in range(pulses.n_pulses):
        if pulses.Ru_is_nonzero(m):
            Rop = np.tensordot(
                Ru[m],
                dipole_operator,
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Rg[m], Rop)
            operators.append(o0)

        if pulses.Iu_is_nonzero(m):
            Iop = np.tensordot(
                Iu[m],
                dipole_operator,
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Ig[m], Iop)
            operators.append(o0)

    return operators


def construct_quadratic_dipole_operators(QS, pulses):
    quadratic_dipole_operator = 0.5 * np.eye(QS.l)

    Rg = pulses.Rg
    Ig = pulses.Ig

    operators = []

    for m in range(pulses.n_pulses):
        if pulses.RuRu_is_nonzero(m, m):
            RuRu = pulses.RuRu(m, m)
            pulse = lambda x: RuRu * Rg[m](x) ** 2
            o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
            operators.append(o0)
        if pulses.RuIu_is_nonzero(m, m):
            RuIu = pulses.RuIu(m, m)
            pulse = lambda x: 2 * RuIu * Rg[m](x) * Ig[m](x)
            o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
            operators.append(o0)
        if pulses.IuIu_is_nonzero(m, m):
            IuIu = pulses.IuIu(m, m)
            pulse = lambda x: IuIu * Ig[m](x) ** 2
            o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
            operators.append(o0)

    return operators


def construct_cross_dipole_operators(QS, pulses):
    cross_dipole_operator = np.eye(QS.l)

    operators = []

    Rg = pulses.Rg
    Ig = pulses.Ig

    pulse_nums = np.arange(pulses.n_pulses)
    for m in pulse_nums:
        for n in pulse_nums[pulse_nums > m]:
            if pulses.RuRu_is_nonzero(m, n):
                RuRu = pulses.RuRu(m, n)
                pulse = lambda x: RuRu * Rg[m](x) * Rg[n](x)
                o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                operators.append(o0)
            if pulses.RuIu_is_nonzero(m, n):
                RuIu = pulses.RuIu(m, n)
                pulse = lambda x: RuIu * Rg[m](x) * Ig[n](x)
                o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                operators.append(o0)
            if pulses.IuRu_is_nonzero(m, n):
                IuRu = pulses.IuRu(m, n)
                pulse = lambda x: IuRu * Ig[m](x) * Rg[n](x)
                o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                operators.append(o0)
            if pulses.IuIu_is_nonzero(m, n):
                IuIu = pulses.IuIu(m, n)
                pulse = lambda x: IuIu * Ig[m](x) * Ig[n](x)
                o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                operators.append(o0)

    return operators


def construct_linear_plane_wave_operators(pwi, pulses):
    operators = []

    Ru = pulses.Ru
    Rg = pulses.Rg
    Iu = pulses.Iu
    Ig = pulses.Ig

    for m in np.arange(pulses.n_pulses):
        if pulses.Ru_is_nonzero(m):
            Rop = np.tensordot(
                Ru[m],
                pwi[f"cosp,{m}"],
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Rg[m], Rop)
            operators.append(o0)

            Rop = np.tensordot(
                Ru[m],
                pwi[f"sinp,{m}"],
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Ig[m], Rop)
            operators.append(o0)

        if pulses.Iu_is_nonzero(m):
            Iop = np.tensordot(
                Iu[m],
                pwi[f"cosp,{m}"],
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Ig[m], Iop)
            operators.append(o0)

            Iop = np.tensordot(
                Iu[m],
                pwi[f"sinp,{m}"],
                axes=(0, 0),
            )
            o0 = CustomOneBodyOperator(Ig[m], -Iop)
            operators.append(o0)

    return operators


def construct_quadratic_plane_wave_operators(pwi, pulses):
    operators = []

    gg = pulses.gg
    uu = pulses.uu

    pulse_nums = np.arange(pulses.n_pulses)

    for m in pulse_nums:
        p_cos_pl = lambda x: 0.25 * (
            (uu(1, 1, m, 1, 1, m) - uu(2, 1, m, 2, 1, m)) * gg(1, m, 1, m)(x)
            + (uu(1, 1, m, 1, 2, m) - uu(2, 1, m, 2, 2, m)) * gg(1, m, 2, m)(x)
            + (uu(1, 2, m, 1, 1, m) - uu(2, 2, m, 2, 1, m)) * gg(2, m, 1, m)(x)
            + (uu(1, 2, m, 1, 2, m) - uu(2, 2, m, 2, 2, m)) * gg(2, m, 2, m)(x)
        )

        p_sin_pl = lambda x: 0.25 * (
            (uu(1, 1, m, 2, 1, m) + uu(2, 1, m, 1, 1, m)) * gg(1, m, 1, m)(x)
            + (uu(1, 1, m, 2, 2, m) + uu(2, 1, m, 1, 2, m)) * gg(1, m, 2, m)(x)
            + (uu(1, 2, m, 2, 1, m) + uu(2, 2, m, 1, 1, m)) * gg(2, m, 1, m)(x)
            + (uu(1, 2, m, 2, 2, m) + uu(2, 2, m, 1, 2, m)) * gg(2, m, 2, m)(x)
        )

        p_cos_mi = lambda x: 0.25 * (
            (uu(1, 1, m, 1, 1, m) + uu(2, 1, m, 2, 1, m)) * gg(1, m, 1, m)(x)
            + (uu(1, 1, m, 1, 2, m) + uu(2, 1, m, 2, 2, m)) * gg(1, m, 2, m)(x)
            + (uu(1, 2, m, 1, 1, m) + uu(2, 2, m, 2, 1, m)) * gg(2, m, 1, m)(x)
            + (uu(1, 2, m, 1, 2, m) + uu(2, 2, m, 2, 2, m)) * gg(2, m, 2, m)(x)
        )

        o0 = CustomOneBodyOperator(
            p_cos_pl,
            pwi[f"cos+,{m}{m}"],
        )
        o1 = CustomOneBodyOperator(
            p_sin_pl,
            pwi[f"sin+,{m}{m}"],
        )
        o2 = CustomOneBodyOperator(
            p_cos_mi,
            np.eye(pwi.l),
        )
        operators.append(o0)
        operators.append(o1)
        operators.append(o2)

    return operators


def construct_cross_plane_wave_operators(pwi, pulses):
    operators = []

    Rg = pulses.Rg
    Ig = pulses.Ig

    pulse_nums = np.arange(pulses.n_pulses)

    for m in pulse_nums:
        for n in pulse_nums[pulse_nums > m]:
            g1m_g1n = pulses.RgRg(m, n)
            g1m_g2n = pulses.RgIg(m, n)
            g2m_g1n = pulses.IgRg(m, n)
            g2m_g2n = pulses.IgIg(m, n)

            RuRu = pulses.RuRu(m, n)
            RuIu = pulses.RuIu(m, n)
            IuRu = pulses.IuRu(m, n)
            IuIu = pulses.IuIu(m, n)

            p_cos_pl = lambda x: 0.5 * (
                (uu(1, 1, m, 1, 1, n) - uu(2, 1, m, 2, 1, n)) * gg(1, m, 1, n)(x)
                + (uu(1, 1, m, 1, 2, n) - uu(2, 1, m, 2, 2, n)) * gg(1, m, 2, n)(x)
                + (uu(1, 2, m, 1, 1, n) - uu(2, 2, m, 2, 1, n)) * gg(2, m, 1, n)(x)
                + (uu(1, 2, m, 1, 2, n) - uu(2, 2, m, 2, 2, n)) * gg(2, m, 2, n)(x)
            )

            p_sin_pl = lambda x: 0.5 * (
                (uu(1, 1, m, 2, 1, n) + uu(2, 1, m, 1, 1, n)) * gg(1, m, 1, n)(x)
                + (uu(1, 1, m, 2, 2, n) + uu(2, 1, m, 1, 2, n)) * gg(1, m, 2, n)(x)
                + (uu(1, 2, m, 2, 1, n) + uu(2, 2, m, 1, 1, n)) * gg(2, m, 1, n)(x)
                + (uu(1, 2, m, 2, 2, n) + uu(2, 2, m, 1, 2, n)) * gg(2, m, 2, n)(x)
            )

            p_cos_mi = lambda x: 0.5 * (
                (uu(1, 1, m, 1, 1, n) + uu(2, 1, m, 2, 1, n)) * gg(1, m, 1, n)(x)
                + (uu(1, 1, m, 1, 2, n) + uu(2, 1, m, 2, 2, n)) * gg(1, m, 2, n)(x)
                + (uu(1, 2, m, 1, 1, n) + uu(2, 2, m, 2, 1, n)) * gg(2, m, 1, n)(x)
                + (uu(1, 2, m, 1, 2, n) + uu(2, 2, m, 2, 2, n)) * gg(2, m, 2, n)(x)
            )

            p_sin_mi = lambda x: 0.5 * (
                (uu(2, 1, m, 1, 1, n) - uu(1, 1, m, 2, 1, n)) * gg(1, m, 1, n)(x)
                + (uu(2, 1, m, 1, 2, n) - uu(1, 1, m, 2, 2, n)) * gg(1, m, 2, n)(x)
                + (uu(2, 2, m, 1, 1, n) - uu(1, 2, m, 2, 1, n)) * gg(2, m, 1, n)(x)
                + (uu(2, 2, m, 1, 2, n) - uu(1, 2, m, 2, 2, n)) * gg(2, m, 2, n)(x)
            )

            o0 = CustomOneBodyOperator(
                p_cos_pl,
                pwi[f"cos+,{m}{n}"],
            )
            o1 = CustomOneBodyOperator(
                p_sin_pl,
                pwi[f"sin+,{m}{n}"],
            )
            o2 = CustomOneBodyOperator(
                p_cos_mi,
                pwi[f"cos-,{m}{n}"],
            )
            o3 = CustomOneBodyOperator(
                p_sin_mi,
                np.sign(m - n) * pwi[f"sin-,{m}{n}"],
            )
            operators.append(o0)
            operators.append(o1)
            operators.append(o2)
            operators.append(o3)

    return operators
