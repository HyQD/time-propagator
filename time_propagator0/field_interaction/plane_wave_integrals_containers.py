import abc

import numpy as np

from time_propagator0.setup_daltonproject import (
    compute_plane_wave_integrals_from_molcas,
)


class IntegralContainer(metaclass=abc.ABCMeta):
    @property
    def l(self):
        return self._l

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        self._C = C

    @property
    def C_tilde(self):
        return self._C_tilde

    @C_tilde.setter
    def C_tilde(self, C_tilde):
        self._C_tilde = C_tilde

    def change_basis(self):
        for i in range(len(self._integrals)):
            self._integrals[i] = self.C_transform(self._integrals[i])
        self.C = np.eye(len(self.C))
        self.C_tilde = np.eye(len(self.C_tilde))

    def C_transform(self, M):
        return self.C_tilde @ M @ self.C

    def __getitem__(self, index):
        pass


class IntegralContainerFixedOrbitals(IntegralContainer):
    def __init__(self, integrals, mapping=None, C=None, C_tilde=None):
        self._integrals = integrals
        self._mapping = mapping
        self._C = C
        self._C_tilde = C_tilde
        self._l = (integrals[0]).shape[-1]

    def __getitem__(self, index):
        return self._integrals[self._mapping[index]]


class IntegralContainerOrbitalAdaptive(IntegralContainer):
    def __init__(self, integrals, mapping=None, C=None, C_tilde=None):
        self._integrals = integrals
        self._mapping = mapping
        self._C = C
        self._C_tilde = C_tilde
        self._l = (integrals[0]).shape[-1]

    def __getitem__(self, index):
        return self.C_transform(self._integrals[self._mapping[index]])


def get_integrals_from_molcas(
    molecule,
    basis,
    omega,
    k_direction,
    verbose=False,
    return_s=False,
    custom_basis=False,
):
    if verbose:
        print("Running Molcas, omega: ", omega)
    ma = compute_plane_wave_integrals_from_molcas(
        molecule, basis, omega, k_direction, custom_basis=custom_basis
    )

    cos2 = ma.cos2
    sin2 = ma.sin2

    l = len(cos2)
    cosp = np.zeros((3, l, l), dtype=complex)
    sinp = np.zeros((3, l, l), dtype=complex)
    cosp[0] = ma.cosp(0)
    cosp[1] = ma.cosp(1)
    cosp[2] = ma.cosp(2)
    sinp[0] = ma.sinp(0)
    sinp[1] = ma.sinp(1)
    sinp[2] = ma.sinp(2)

    return (cosp, sinp, cos2, sin2) if not return_s else (cosp, sinp, cos2, sin2, ma.s)


def setup_plane_wave_integrals_from_molcas(
    pulse_inputs,
    molecule,
    basis,
    quadratic_terms=False,
    cross_terms=False,
    compute_A=False,
    custom_basis=False,
):
    n_pulses = len(pulse_inputs)

    integrals = []
    index_mapping = {}

    i = 0

    for m in range(n_pulses):
        k_direction = pulse_inputs[m]["k_direction"]
        omega = pulse_inputs[m]["omega"]

        cosp, sinp, cos2, sin2, s = get_integrals_from_molcas(
            molecule,
            basis,
            omega,
            k_direction,
            return_s=True,
            custom_basis=custom_basis,
        )

        integrals.append(cosp)
        index_mapping[f"cosp,{m}"] = i
        i += 1
        integrals.append(sinp)
        index_mapping[f"sinp,{m}"] = i
        i += 1

        if quadratic_terms:
            integrals.append(cos2)
            index_mapping[f"cos+,{m}{m}"] = i
            i += 1
            integrals.append(sin2)
            index_mapping[f"sin+,{m}{m}"] = i
            i += 1
            integrals.append(s)
            index_mapping[f"cos-,{m}{m}"] = i
            i += 1
            integrals.append(np.zeros_like(s))
            index_mapping[f"sin-,{m}{m}"] = i
            i += 1

        if compute_A:
            cosp, sinp, cos2, sin2 = get_integrals_from_molcas(
                molecule, basis, omega / 2, k_direction, custom_basis=custom_basis
            )
            integrals.append(cos2)
            index_mapping[f"cos,{m}"] = i
            i += 1
            integrals.append(sin2)
            index_mapping[f"sin,{m}"] = i
            i += 1

    if cross_terms:
        pulse_nums = np.arange(n_pulses)
        for m in pulse_nums:
            for n in pulse_nums[pulse_nums > m]:

                k_direction_m = pulse_inputs[m]["k_direction"]
                omega_m = pulse_inputs[m]["omega"]

                k_direction_n = pulse_inputs[n]["k_direction"]
                omega_n = pulse_inputs[n]["omega"]

                ck_m = omega_m * np.array(k_direction_m)
                ck_n = omega_n * np.array(k_direction_n)
                ck_pl = ck_m + ck_n
                ck_mi = ck_m - ck_n
                omega_pl = (1 / 2) * np.linalg.norm(ck_pl)
                omega_mi = (1 / 2) * np.linalg.norm(ck_mi)
                k_direction_pl = ck_pl / np.linalg.norm(ck_pl)
                k_direction_mi = ck_mi / np.linalg.norm(ck_mi)
                cosp, sinp, cos2, sin2 = get_integrals_from_molcas(
                    molecule, basis, omega_pl, k_direction_pl, custom_basis=custom_basis
                )
                integrals.append(cos2)
                index_mapping[f"cos+,{m}{n}"] = i
                index_mapping[f"cos+,{n}{m}"] = i
                i += 1
                integrals.append(sin2)
                index_mapping[f"sin+,{m}{n}"] = i
                index_mapping[f"sin+,{n}{m}"] = i
                i += 1

                cosp, sinp, cos2, sin2 = get_integrals_from_molcas(
                    molecule, basis, omega_mi, k_direction_mi, custom_basis=custom_basis
                )
                integrals.append(cos2)
                index_mapping[f"cos-,{m}{n}"] = i
                index_mapping[f"cos-,{n}{m}"] = i
                i += 1
                integrals.append(sin2)
                index_mapping[f"sin-,{m}{n}"] = i
                i += 1
                integrals.append(-sin2)
                index_mapping[f"sin-,{n}{m}"] = i
                i += 1

    return integrals, index_mapping


def setup_PlaneWaveIntegralsContainerMemorySaver_from_molcas():
    pass
