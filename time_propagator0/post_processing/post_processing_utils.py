import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from time_propagator0 import Inputs, setup_Pulses
from numba import jit


def parse_arg(a):
    if type(a) == str:
        return np.load(a, allow_pickle=True)
    else:
        return a


def convert_old_input_params(inp_dict):
    key = "dt"
    if key in inp_dict:
        inp_dict["time_step"] = inp_dict[key]
        del inp_dict[key]

    key = "pulse_class"
    for el in inp_dict["pulses"]:
        if key in inp_dict[el]:
            inp_dict[el]["laser_class"] = inp_dict[el][key]
            del inp_dict[el][key]

    key = "initial_state"
    if key in inp_dict:
        del inp_dict[key]

    key = "verbose"
    if key in inp_dict:
        inp_dict["print_level"] = inp_dict[key]
        del inp_dict[key]

    key = "return_inputs"
    if key in inp_dict:
        del inp_dict[key]

    key = "return_C"
    if key in inp_dict:
        inp_dict["return_hf_coeff_matrix"] = inp_dict[key]
        del inp_dict[key]


def setup_inputs(a):
    i = (a["inputs"]).item()
    convert_old_input_params(i)
    inputs = Inputs({})
    inputs.set_from_dict(i)
    return inputs


def setup_samples(a):
    s = (a["samples"]).item()
    return s


def setup_arrays(samples, inputs):
    arrays = Arrays(samples, inputs)
    return arrays


def setup_pulses(inputs):
    pulse_inputs = []
    for el in inputs("pulses"):
        pulse_inputs.append(inputs(el))

    pulses = setup_Pulses(pulse_inputs)

    return pulses


def transient_dipole_spectrum_de(a, induced_dipole=True):
    a = parse_arg(a)
    inputs = setup_inputs(a)
    samples = setup_samples(a)
    arrays = setup_arrays(samples, inputs)
    pulses = setup_pulses(inputs)

    # generate arrays
    t0 = arrays(samples["time_points"])
    d0 = arrays(samples["dipole_moment"])

    E0 = pulses.pulses(t0)

    # obtain time-symmetrized version of arrays
    t = arrays.get_symmetrized_time()
    d = arrays.get_symmetrized_3D(d0, induced=induced_dipole)
    E = arrays.get_symmetrized_3D(E0)

    # fourier transform
    time_step = inputs("time_step") if inputs.has_key("time_step") else inputs("dt")
    fftransform = FFTransform(t, time_step)

    d_tilde = fftransform.transform_3D(d)
    E_tilde_conj = fftransform.transform_3D(E).conj()
    dE = np.sum(d_tilde * E_tilde_conj, axis=1)

    freq = fftransform.freq

    return physical_spectrum(freq, -2 * dE.imag)


def transient_dipole_spectrum_pia(a):
    a = parse_arg(a)
    inputs = setup_inputs(a)
    samples = setup_samples(a)
    arrays = setup_arrays(samples, inputs)
    pulses = setup_pulses(inputs)

    # generate arrays
    t0 = arrays(samples["time_points"])
    pi0 = arrays(samples["kinetic_momentum"])

    A0 = pulses.pulses(t0)

    # obtain time-symmetrized version of arrays
    t = arrays.get_symmetrized_time()
    pi = arrays.get_symmetrized_3D(pi0)
    A = arrays.get_symmetrized_3D(A0)

    # fourier transform
    time_step = inputs("time_step") if inputs.has_key("time_step") else inputs("dt")
    fftransform = FFTransform(t, time_step)

    freq = fftransform.freq

    pi_tilde = fftransform.transform_3D(pi)
    A_tilde = fftransform.transform_3D(A)

    A_tilde_conj = A_tilde.conj()

    piA = np.sum(pi_tilde * A_tilde_conj, axis=1)

    return physical_spectrum(freq, 2 * piA.imag)


def transient_plane_wave_spectrum(a):
    a = parse_arg(a)
    inputs = setup_inputs(a)
    samples = setup_samples(a)
    arrays = setup_arrays(samples, inputs)
    pulses = setup_pulses(inputs)

    # generate arrays
    t0 = arrays(samples["time_points"])
    F0 = arrays(samples["general_response"])

    # obtain time-symmetrized version of arrays
    t = arrays.get_symmetrized_time()

    # fourier transform
    time_step = inputs("time_step") if inputs.has_key("time_step") else inputs("dt")
    fftransform = FFTransform(t, time_step)

    freq = fftransform.freq

    S = np.zeros(len(freq))

    pulse_numbers = np.arange(pulses.n_pulses)

    for m in pulse_numbers:
        F_1_cos = arrays.get_symmetrized(F0[:, 0, 0, m])
        F_2_cos = arrays.get_symmetrized(F0[:, 1, 0, m])
        F_1_sin = arrays.get_symmetrized(F0[:, 0, 1, m])
        F_2_sin = arrays.get_symmetrized(F0[:, 1, 1, m])
        g_cos = arrays.get_symmetrized(pulses.g(1)[m](t0))
        g_sin = arrays.get_symmetrized(pulses.g(2)[m](t0))

        F_1_cos_tilde = fftransform.transform(F_1_cos)
        F_2_cos_tilde = fftransform.transform(F_2_cos)
        F_1_sin_tilde = fftransform.transform(F_1_sin)
        F_2_sin_tilde = fftransform.transform(F_2_sin)
        g_cos_tilde = fftransform.transform(g_cos)
        g_sin_tilde = fftransform.transform(g_sin)

        S += (
            F_1_cos_tilde * g_cos_tilde.conj() + F_1_cos_tilde.conj() * g_cos_tilde
        ).real
        S += (
            F_2_cos_tilde * g_cos_tilde.conj() + F_2_cos_tilde.conj() * g_cos_tilde
        ).real
        S += (
            F_1_sin_tilde * g_sin_tilde.conj() + F_1_sin_tilde.conj() * g_sin_tilde
        ).real
        S += (
            F_2_sin_tilde * g_sin_tilde.conj() + F_2_sin_tilde.conj() * g_sin_tilde
        ).real

    pulse_numbers = np.arange(pulses.n_pulses)

    return physical_spectrum(freq, S)


def physical_spectrum(freq, f):
    ind = freq > 0
    return freq[ind], f[ind]


################################################################################
class FFTransform:
    def __init__(self, t, dt):
        self.n = len(t)
        self.t = t
        self.t_max = t[-1]
        self.dt = dt

        self.index = np.ones(self.n, dtype=bool)
        if not self.n % 2:
            self.index[0] = False

        self.freq = (fftshift(fftfreq(self.n)) * (2 * np.pi / self.dt))[self.index]

    def w(self, window):
        if window == "no_window":
            return 1
        if window == "hann":
            return np.cos(np.pi * self.t / (2 * self.t_max)) ** 2
        if window == "exp":
            return np.exp(-0.001 * self.t)

    def transform(self, array, window="hann"):
        return (fftshift(fft(self.w(window) * array)))[self.index]

    def transform_2D(self, array, window="hann"):
        a0 = self.transform(array[:, 0], window=window)
        a1 = self.transform(array[:, 1], window=window)
        return np.concatenate((np.atleast_2d(a0).T, np.atleast_2d(a1).T), axis=1)

    def transform_3D(self, array, window="hann"):
        a0 = self.transform_2D(array[:, 0:2], window=window)
        a1 = self.transform(array[:, 2], window=window)
        return np.concatenate((a0, np.atleast_2d(a1).T), axis=1)


################################################################################
class Arrays:
    def __init__(self, a, inputs):
        self.generate_index(a, inputs)
        self.t = a["time_points"][self.index]
        self.dt = self.t[1] - self.t[0]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]

    def generate_index(self, a, inputs):
        t = a["time_points"]
        if t[-1] != 0:
            ind0 = np.ones(len(t), dtype=bool)
        elif t[-1] == 0 and t[-2] != 0:
            ind0 = np.ones(len(t), dtype=bool)
            ind0[-1] = False
        else:
            ind0 = t != 0
            where = np.where(ind0 == False)
            if where[0][1] - where[0][0] > 1:
                ind0[where[0][0]] = True

        self.index = ind0

    def get_time(self):
        self.t

    def get_symmetrized_time(self):
        ext_t = np.arange(-self.t_max, self.t_min, step=self.dt)
        return np.concatenate((ext_t, self.t))

    def get_symmetrized(self, array, induced=False):
        temp_a = array.copy()
        if induced:
            temp_a = temp_a - temp_a[0]
        n_additional_points = len(np.arange(-self.t_max, self.t_min, step=self.dt))
        ext = np.ones(n_additional_points)
        return np.concatenate((temp_a[0] * ext, temp_a))

    def get_symmetrized_2D(self, array, induced=False):
        a0 = self.get_symmetrized(array[:, 0], induced=induced)
        a1 = self.get_symmetrized(array[:, 1], induced=induced)
        return np.concatenate((np.atleast_2d(a0).T, np.atleast_2d(a1).T), axis=1)

    def get_symmetrized_3D(self, array, induced=False):
        a0 = self.get_symmetrized_2D(array[:, 0:2], induced=induced)
        a1 = self.get_symmetrized(array[:, 2], induced=induced)
        return np.concatenate((a0, np.atleast_2d(a1).T), axis=1)

    def __call__(self, array):
        return array[self.index]
