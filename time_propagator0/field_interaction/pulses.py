import abc
import numpy as np


class Pulses(metaclass=abc.ABCMeta):
    @property
    def n_pulses(self):
        return self._n_pulses

    @property
    def has_real_polarization(self):
        return self._has_real_polarization

    @property
    def has_imaginary_polarization(self):
        return self._has_imaginary_polarization

    @property
    def Rg(self):
        return self._Rg

    @property
    def Ig(self):
        return self._Ig

    @property
    def Ru(self):
        return self._Ru

    @property
    def Iu(self):
        return self._Iu

    def u(self, i, j):
        if i == j:
            return self.Ru
        else:
            return np.sign(i - j) * self.Iu

    def uu(self, i, j, pulse_no1, k, l, pulse_no2):
        return np.dot(self.u(i, j)[pulse_no1], self.u(k, l)[pulse_no2])

    def g(self, i):
        if i == 1:
            return self._Rg
        elif i == 2:
            return self._Ig

    def gg(self, i, pulse_no1, j, pulse_no2):
        return lambda x: self.g(i)[pulse_no1](x) * self.g(j)[pulse_no2](x)

    def RgRg(self, pulse_no1, pulse_no2):
        return lambda x: self.Rg[pulse_no1](x) * self.Rg[pulse_no2](x)

    def RgIg(self, pulse_no1, pulse_no2):
        return lambda x: self.Rg[pulse_no1](x) * self.Ig[pulse_no2](x)

    def IgRg(self, pulse_no1, pulse_no2):
        return lambda x: self.Ig[pulse_no1](x) * self.Rg[pulse_no2](x)

    def IgIg(self, pulse_no1, pulse_no2):
        return lambda x: self.Ig[pulse_no1](x) * self.Ig[pulse_no2](x)

    def RuRu(self, pulse_no1, pulse_no2):
        return np.dot(self.Ru[pulse_no1], self.Ru[pulse_no2])

    def IuRu(self, pulse_no1, pulse_no2):
        return np.dot(self.Iu[pulse_no1], self.Ru[pulse_no2])

    def RuIu(self, pulse_no1, pulse_no2):
        return np.dot(self.Ru[pulse_no1], self.Iu[pulse_no2])

    def IuIu(self, pulse_no1, pulse_no2):
        return np.dot(self.Iu[pulse_no1], self.Iu[pulse_no2])

    def Ru_is_nonzero(self, pulse_no):
        return True if pulse_no in self._nonzero_Ru else False

    def Iu_is_nonzero(self, pulse_no):
        return True if pulse_no in self._nonzero_Iu else False

    def RuRu_is_nonzero(self, pulse_no1, pulse_no2):
        val = np.dot(self.Ru[pulse_no1], self.Ru[pulse_no2])
        return True if np.abs(val) > self._eps else False

    def RuIu_is_nonzero(self, pulse_no1, pulse_no2):
        val = np.dot(self.Ru[pulse_no1], self.Iu[pulse_no2])
        return True if np.abs(val) > self._eps else False

    def IuRu_is_nonzero(self, pulse_no1, pulse_no2):
        val = np.dot(self.Iu[pulse_no1], self.Ru[pulse_no2])
        return True if np.abs(val) > self._eps else False

    def IuIu_is_nonzero(self, pulse_no1, pulse_no2):
        val = np.dot(self.Iu[pulse_no1], self.Iu[pulse_no2])
        return True if np.abs(val) > self._eps else False

    def pulse(self, t, m):
        pass

    def pulses(self, t):
        pass


# class DipolePulsesRealPolarization(Pulses):
#    def __init__(self,Rg,Ru):
#        self._Rg = Rpulses
#        self._Ru = Ru
#
#        self._n_pulses = len(Rpulses)
#
#        self._has_real_polarization = True
#
#    def single_pulse(self,t,m):
#        return np.squeeze((self.Rg[m](t)*self.Ru[m,:,None]).T)


class GeneralPulses(Pulses):
    def __init__(self, Rg, Ig, Ru, Iu, eps=1e-14):
        self._Rg = Rg
        self._Ig = Ig
        self._Ru = Ru
        self._Iu = Iu

        self._n_pulses = len(Rg)

        self._eps = eps

        self._nonzero_Ru = []
        self._nonzero_Iu = []

        for i in range(self._n_pulses):
            max_Ru = np.max(np.abs(Ru[i, :]))
            max_Iu = np.max(np.abs(Iu[i, :]))
            if max_Ru > eps:
                self._nonzero_Ru.append(i)
            if max_Iu > eps:
                self._nonzero_Iu.append(i)

        self._has_real_polarization = True if len(self._nonzero_Ru) == 0 else False
        self._has_imag_polarization = True if len(self._nonzero_Iu) == 0 else False

    def pulse(self, t, m):
        ret = np.squeeze(np.zeros((len(t), 3)))
        if self.Ru_is_nonzero(m):
            ret += np.squeeze((self.Rg[m](t) * self.Ru[m, :, None]).T)
        if self.Iu_is_nonzero(m):
            ret += np.squeeze((self.Ig[m](t) * self.Iu[m, :, None]).T)
        return ret

    def pulses(self, t):
        ret = np.squeeze(np.zeros((len(t), 3)))
        for m in self._nonzero_Ru:
            ret += self.pulse(t, m)
        for m in self._nonzero_Iu:
            ret += self.pulse(t, m)
        return ret


def setup_Pulses(pulse_inputs):
    from time_propagator0.field_interaction import lasers

    n_pulses = len(pulse_inputs)

    Rg = []
    Ig = []

    Ru = np.atleast_2d(np.zeros((n_pulses, 3), dtype=float))
    Iu = np.atleast_2d(np.zeros((n_pulses, 3), dtype=float))

    c = 0
    for m in range(n_pulses):
        inputs = pulse_inputs[m]

        Ru[c, :] = (inputs["polarization"]).real
        Iu[c, :] = (inputs["polarization"]).imag

        pulse_class = inputs["pulse_class"]

        Laser = vars(lasers)[pulse_class]

        Rg.append(Laser(**inputs))

        imag_inputs = inputs.copy()
        if "phase" in imag_inputs:
            imag_inputs["phase"] -= np.pi / 2
        else:
            imag_inputs["phase"] = -np.pi / 2

        Ig.append(Laser(**imag_inputs))

        c += 1

    return GeneralPulses(Rg, Ig, Ru, Iu)
