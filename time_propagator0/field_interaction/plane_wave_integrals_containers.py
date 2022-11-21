import abc
import numpy as np

from time_propagator0.setup_daltonproject import (
    compute_plane_wave_integrals_from_molcas,
)


class IntegralContainer(metaclass=abc.ABCMeta):
    def __init__(self, integrals, C=None, C_tilde=None):
        self._integrals = integrals
        self._C = C
        self._C_tilde = C_tilde
        self._l = (integrals["cosp,0"]).shape[-1]

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

    def change_basis(self,C=None,C_tilde=None):
        if C is None:
            for el in self._integrals:
                self._integrals[el] = self.C_tilde @ self._integrals[el] @ self.C
            self.C = np.eye(len(self.C))
            self.C_tilde = np.eye(len(self.C_tilde))
        else:
            for el in self._integrals:
                self._integrals[el] = C_tilde @ self._integrals[el] @ C

    @abc.abstractmethod
    def __getitem__(self, index):
        pass


class IntegralContainerFixedOrbitals(IntegralContainer):
    def __getitem__(self, index):
        return self._integrals[index]


class IntegralContainerOrbitalAdaptive(IntegralContainer):
    def __getitem__(self, index):
        return self.C_tilde @ self._integrals[index] @ self.C


def get_integrals_from_molcas(
    molecule,
    basis,
    omega,
    k_direction,
    return_s=False,
    custom_basis=False,
):
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

    integrals = {}

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

        integrals[f"cosp,{m}"] = cosp
        integrals[f"sinp,{m}"] = sinp

        if quadratic_terms:
            integrals[f"cos+,{m}{m}"] = cos2
            integrals[f"sin+,{m}{m}"] = sin2
            integrals[f"cos-,{m}{m}"] = s
            integrals[f"sin-,{m}{m}"] = np.zeros_like(s)

        if compute_A:
            cosp, sinp, cos2, sin2 = get_integrals_from_molcas(
                molecule, basis, omega / 2, k_direction, custom_basis=custom_basis
            )
            integrals[f"cos,{m}"] = cos2
            integrals[f"sin,{m}"] = sin2

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
                integrals[f"cos+,{m}{n}"] = cos2
                integrals[f"cos+,{n}{m}"] = cos2
                integrals[f"sin+,{m}{n}"] = sin2
                integrals[f"sin+,{n}{m}"] = sin2

                cosp, sinp, cos2, sin2 = get_integrals_from_molcas(
                    molecule, basis, omega_mi, k_direction_mi, custom_basis=custom_basis
                )
                integrals[f"cos-,{m}{n}"] = cos2
                integrals[f"cos-,{n}{m}"] = cos2
                integrals[f"sin-,{m}{n}"] = sin2
                integrals[f"sin-,{n}{m}"] = -sin2

    return integrals
