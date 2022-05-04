import numpy as np

from time_propagator0 import lasers

from time_propagator0.setup_daltonproject import (
    setup_plane_wave_integrals_from_molcas,
)

import importlib

from qcelemental import periodictable

################################################################################
class Inputs:
    def __init__(self, a):
        """a: NpzFile object or dict or list (containing dicts)"""
        # self.to_listify = [
        #    "field_strength",
        #    "omega",
        #    "time_after_pulse",
        #    "ncycles",
        #    "phase",
        #    "sigma",
        #    "pulse",
        # ]

        # self.require_2d = ["k_direction", "polarization"]

        if type(a) is dict:
            self.setup_dict(a)
        elif type(a) is list:
            self.setup_list(a)
        elif type(a) is np.lib.npyio.NpzFile:
            self.setup_npz(a)

    def format(self, dict_):
        if "pulses" in dict_.keys():
            for el in dict_["pulses"]:
                if el in dict_:
                    if "polarization" in dict_[el]:
                        dict_[el]["polarization"] = np.array(dict_[el]["polarization"])
                    if "k_direction" in dict_[el]:
                        dict_[el]["k_direction"] = np.array(dict_[el]["k_direction"])
        return dict_

    def setup_dict(self, a):
        self.inputs = self.format(a)

    def setup_list(self, a):
        dict_ = a[0]
        for i in np.arange(1, len(a)):
            dict_ = {**dict_, **a[i]}

        self.inputs = self.format(dict_)

    def setup_npz(self, a):
        if not "laser_inputs" in a:
            dict_ = a["inputs"].item()
            self.inputs = self.format(dict_)
        else:
            dict0 = a["inputs"].item()
            dict1 = a["laser_inputs"].item()
            dict2 = a["laser_classes"].item()
            dict3 = a["sample_settings"].item()

            dict_ = {**dict0, **dict1, **dict2, **dict3}
            self.inputs = self.format(dict_)

    def set_from_file(self, file_name):
        input_module = importlib.import_module(file_name.replace(".py", ""))
        input_attr_names = [el for el in dir(input_module) if not el.startswith("__")]
        for el in input_attr_names:
            input_dict = getattr(input_module, el)
            if type(input_dict) == dict:
                self.set_from_dict(input_dict)

    def set_from_dict(self, input_dict):
        for key in input_dict:
            self.set(key, input_dict[key])

    def set(self, key, value):
        self.inputs[key] = value
        self.inputs = self.format(self.inputs)

    def has_key(self, key):
        return key in self.inputs.keys()

    def check_consistency(self):
        """method to check input consistency"""
        # check that only harmonic pulses have complex polarization vectors

        if self.has_key("pulses"):
            for el in self.inputs["pulses"]:
                s = "pulse must have defined inputs"
                assert self.has_key(el), s

                pulse_inputs = self.inputs[el]

                s = "pulses must have defined polarization"
                assert "polarization" in pulse_inputs.keys(), s
                uI = (pulse_inputs["polarization"]).imag
                if np.max(np.abs(uI)) > 1e-14:
                    s = "only harmonic pulses can have complex polarization vectors"
                    assert "omega" in self.inputs[el], s

    def __call__(self, key):
        """key: str"""
        return self.inputs[key]


class PlaneWaveOperators:
    def __init__(
        self,
        C,
        input_file,
        basis,
        pulses_cos,
        pulses_sin,
        n_basis,
        custom_basis=False,
        change_basis=True,
    ):
        self.C = C
        self.change_basis = change_basis
        self.input_file = input_file
        self.basis = basis
        self.custom_basis = custom_basis
        self.n_pulses = len(pulses_cos)
        self.n_basis = n_basis
        self.create_indices()

        n_pulses = self.n_pulses

        self.Ap_real = np.zeros((n_pulses, 2, n_basis, n_basis), dtype=np.complex128)
        self.Ap_imag = np.zeros((n_pulses, 2, n_basis, n_basis), dtype=np.complex128)
        self.A2 = np.zeros(
            (n_pulses, n_pulses, 4, n_basis, n_basis), dtype=np.complex128
        )
        self.A = np.zeros((n_pulses, 2, n_basis, n_basis), dtype=np.complex128)

        self.set_pulses(pulses_cos, pulses_sin)

    def create_indices(self):
        self.indices = {}
        self.indices["cos+"] = 0
        self.indices["sin+"] = 1
        self.indices["cos-"] = 2
        self.indices["sin-"] = 3
        self.indices["cos"] = 0
        self.indices["sin"] = 1

    def set_pulses(self, pulses_cos, pulses_sin):
        self.pulses = []
        for i in np.arange(self.n_pulses):
            self.pulses.append([pulses_cos[i], pulses_sin[i]])

    def construct_operators(self, inputs, compute_A=True):
        quadratic_terms = inputs("quadratic_terms")
        cross_terms = inputs("cross_terms")

        for i in np.arange(self.n_pulses):
            pulse_name = inputs("pulses")[i]
            pulse_inputs = inputs(pulse_name)
            k_direction = pulse_inputs["k_direction"]
            omega = pulse_inputs["omega"]

            Ru = pulse_inputs["polarization"].real
            Iu = pulse_inputs["polarization"].imag

            cosp, sinp, cos2, sin2 = self.compute_vpi(omega, k_direction)
            self.Ap_real[i, 0, :, :] = np.tensordot(Ru, cosp, axes=(0, 0))
            self.Ap_real[i, 1, :, :] = np.tensordot(Ru, sinp, axes=(0, 0))
            self.Ap_imag[i, 0, :, :] = np.tensordot(Iu, cosp, axes=(0, 0))
            self.Ap_imag[i, 1, :, :] = np.tensordot(Iu, sinp, axes=(0, 0))

            if quadratic_terms:
                self.A2[i, i, 0, :, :] = cos2
                self.A2[i, i, 1, :, :] = sin2
                self.A2[i, i, 2, :, :] = np.eye(self.n_basis)

            if compute_A:
                cosp, sinp, cos2, sin2 = self.compute_vpi(omega / 2, k_direction)
                self.A[i, 0, :, :] = cos2
                self.A[i, 1, :, :] = sin2

        if cross_terms:
            pulse_nums = np.arange(self.n_pulses)
            for i in pulse_nums:
                for j in pulse_nums[pulse_nums != i]:
                    pulse_name_i = inputs("pulses")[i]
                    pulse_name_j = inputs("pulses")[j]

                    pulse_inputs_i = inputs(pulse_name_i)
                    pulse_inputs_j = inputs(pulse_name_j)

                    k_direction_i = pulse_inputs_i["k_direction"]
                    omega_i = pulse_inputs_i["omega"]

                    k_direction_j = pulse_inputs_j["k_direction"]
                    omega_j = pulse_inputs_j["omega"]

                    ck_i = omega_i * np.array(k_direction_i)
                    ck_j = omega_j * np.array(k_direction_j)
                    ck_p = ck_i + ck_j
                    ck_m = ck_i - ck_j
                    omega_p = (1 / 2) * np.linalg.norm(ck_p)
                    omega_m = (1 / 2) * np.linalg.norm(ck_m)
                    k_direction_p = ck_p / np.linalg.norm(ck_p)
                    k_direction_m = ck_m / np.linalg.norm(ck_m)
                    cosp, sinp, cos2, sin2 = self.compute_vpi(omega_p, k_direction_p)
                    self.A2[i, j, 0, :, :] = cos2
                    self.A2[i, j, 1, :, :] = sin2

                    cosp, sinp, cos2, sin2 = self.compute_vpi(omega_m, k_direction_m)
                    self.A2[i, j, 2, :, :] = cos2
                    self.A2[i, j, 3, :, :] = sin2

    def transform_operator(self, a):
        if self.change_basis:
            return np.dot(np.dot(self.C.T, a), self.C)
        else:
            return a

    def compute_vpi(self, omega, k_direction):
        print("Running Molcas, omega: ", omega)
        self.ma = setup_plane_wave_integrals_from_molcas(
            self.input_file,
            self.basis,
            omega,
            k_direction,
            custom_basis=self.custom_basis,
        )

        l = self.n_basis
        cosp = np.zeros((3, l, l), dtype=complex)
        sinp = np.zeros((3, l, l), dtype=complex)
        cosp[0] = self.transform_operator(self.ma.cosp(0))
        cosp[1] = self.transform_operator(self.ma.cosp(1))
        cosp[2] = self.transform_operator(self.ma.cosp(2))
        sinp[0] = self.transform_operator(self.ma.sinp(0))
        sinp[1] = self.transform_operator(self.ma.sinp(1))
        sinp[2] = self.transform_operator(self.ma.sinp(2))

        cos2 = self.transform_operator(self.ma.cos2)
        sin2 = self.transform_operator(self.ma.sin2)

        return cosp, sinp, cos2, sin2

    def linear_operator_is_zero(self, laser_no, component, contraction, eps=1e-14):
        Ap = self.Ap_real if contraction == "real" else self.Ap_imag
        op = Ap[laser_no, self.indices[component], :, :]
        is_zero = True if np.max(np.abs(op)) < eps else False
        return is_zero

    def linear_operator(self, laser_no, component, contraction, C=None, C_tilde=None):
        """component: 'cos' or 'sin'
        contraction: 'real' or 'imag'"""
        Ap = self.Ap_real if contraction == "real" else self.Ap_imag
        if (C is None) or (C_tilde is None):
            return Ap[laser_no, self.indices[component], :, :]
        else:
            return C_tilde @ Ap[laser_no, self.indices[component], :, :] @ C

    def quadratic_operator(self, laser_no1, laser_no2, component, C=None, C_tilde=None):
        """component: 'cos+', 'sin+', 'cos-' or 'sin-'"""
        if (C is None) or (C_tilde is None):
            return self.A2[laser_no1, laser_no2, self.indices[component], :, :]
        else:
            return (
                C_tilde
                @ self.A2[laser_no1, laser_no2, self.indices[component], :, :]
                @ C
            )

    def A_operator(self, laser_no, component, C=None, C_tilde=None):
        """component: 'cos' or 'sin'"""
        if (C is None) or (C_tilde is None):
            return self.A[laser_no, self.indices[component], :, :]
        else:
            return C_tilde @ self.A[laser_no, self.indices[component], :, :] @ C

    def linear_pulse(self, laser_no, component):
        """component: 'cos' or 'sin'"""
        return self.pulses[laser_no][self.indices[component]]

    def old_quadratic_pulse(self, laser_no1, laser_no2, component):
        """component: 'cos+', 'sin+', 'cos-' or 'sin-'"""
        g11 = self.pulses[laser_no1][self.indices["cos"]]
        g21 = self.pulses[laser_no1][self.indices["sin"]]
        g12 = self.pulses[laser_no2][self.indices["cos"]]
        g22 = self.pulses[laser_no2][self.indices["sin"]]
        if component == "cos+":
            return lambda x: g11(x) * g12(x) - g21(x) * g22(x)
        if component == "sin+":
            return lambda x: g21(x) * g12(x) + g11(x) * g22(x)
        if component == "cos-":
            return lambda x: g11(x) * g12(x) + g21(x) * g22(x)
        if component == "sin-":
            return lambda x: g21(x) * g12(x) - g11(x) * g22(x)

    def quadratic_pulse(self, laser_no1, laser_no2, component1, component2):
        """component: 'cos', 'sin'"""
        g1 = self.pulses[laser_no1][self.indices[component1]]
        g2 = self.pulses[laser_no2][self.indices[component2]]
        return lambda x: g1(x) * g2(x)

    def A0(self, laser_no):
        return self.pulses[laser_no][0].A0(0)


def symbols2atomicnumbers(symbols):
    n = len(symbols)
    atomicnumbers = []
    for i in np.arange(n):
        atomicnumbers.append(periodictable.to_atomic_number(symbols[i]))
    return atomicnumbers


def symbols2nelectrons(symbols):
    return sum(symbols2atomicnumbers(symbols))
