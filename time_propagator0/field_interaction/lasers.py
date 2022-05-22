import numpy as np
from scipy.special import erf
import abc


class Laser(metaclass=abc.ABCMeta):
    @property
    def t0_at_center(self):
        return False

    def set_t0(self, t0):
        self.t0 = t0

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.sigma2 = sigma ** 2

    def set_tprime(self, tprime):
        self.tprime = tprime


class time_dependent_polarization_vector:
    def __init__(self, p0, p1, t_cut):
        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)
        self.T = t_cut

    def __call__(self, t=0.0):
        # NB! t must be a scalar here
        dt = t - self.T
        pol = self.p0 * np.heaviside(-dt, 1e0) + self.p1 * np.heaviside(dt, 1e0)
        return pol


class time_dependent_phase:
    def __init__(self, phi0=0.0, a=0.0, b=0.0, c=0.0):
        self.phi0 = phi0
        self.a = a
        self.b = b
        self.c = c

    def _frequency(self, t):
        return self.a + self.b * t + self.c * t * t

    def __call__(self, t):
        t2 = t ** 2
        t3 = t2 * t
        return self.phi0 + self.a * t + self.b * t2 + self.c * t3


### delta pulses #########################################
class discrete_delta_pulse:
    def __init__(self, field_strength, dt, **kwargs):
        self.field_strength = field_strength
        self.dt = dt

    def __call__(self, t):
        if t < self.dt:
            return self.field_strength
        else:
            return 0


class gaussian_delta_pulse:
    # https://pubs.acs.org/doi/10.1021/ct200137z
    def __init__(self, field_strength=1e-3, t_c=5, gamma=5.0, **kwargs):

        self.field_strength = field_strength
        self.t_c = t_c
        self.gamma = gamma

    def __call__(self, t):
        return (
            self.field_strength
            * np.sqrt(self.gamma / np.pi)
            * np.exp(-self.gamma * (t - self.t_c) ** 2)
        )


class zero_pulse(Laser):
    def __init__(self, **kwargs):
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        return 0


### sin2 envelope on A #############################################


class square_velocity_plane(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse_cos = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        pulse_sin = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse_cos, pulse_sin


class square_velocity_plane_cos(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def envelope(self, t):
        dt = t - self.t0

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class square_velocity_plane_sin(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class square_velocity_dipole(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class square_length_dipole(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.sin(np.pi * dt / self.tprime)
            * (
                self.omega
                * np.sin(np.pi * dt / self.tprime)
                * np.sin(self.omega * dt + self.phase)
                - (2 * np.pi / self.tprime)
                * np.cos(np.pi * dt / self.tprime)
                * np.cos(self.omega * dt + self.phase)
            )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.A0
        )
        return pulse


#######--#############--#####################--########--########


### gaussian envelope on A ################################


class gaussian_A_plane_cos(Laser):
    def __init__(self, field_strength, omega, sigma, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-(dt ** 2) / (2 * self.sigma2))
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class gaussian_A_plane_sin(Laser):
    def __init__(self, field_strength, omega, tprime, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-(dt ** 2) / (2 * self.sigma2))
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class gaussian_A_velocity(Laser):
    def __init__(self, field_strength, omega, sigma, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-(dt ** 2) / (2 * self.sigma2))
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class gaussian_A_length(Laser):
    def __init__(self, field_strength, omega, sigma, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0
        self.sigma2 = sigma ** 2

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-(dt ** 2) / (2 * self.sigma2))
            * (
                (dt / self.sigma2) * np.cos(self.omega * dt + self._phase(dt))
                + self.omega * np.sin(self.omega * dt + self._phase(dt))
            )
            * self.A0
        )
        return pulse


#######--#############--#####################--########--########

### gaussian envelope on E ################################


class gaussian_E_length(Laser):
    def __init__(self, field_strength, omega, sigma, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def set_t0(self, t0):
        self.t0 = t0

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-(dt ** 2) / (2 * self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = np.exp(-(dt ** 2) / (2 * self.sigma2)) * np.sin(
            self.omega * dt + self._phi(dt)
        )
        return self.field_strength * pulse


class gaussian_E_length_cos(Laser):
    def __init__(self, field_strength, omega, sigma, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def set_t0(self, t0):
        self.t0 = t0

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-(dt ** 2) / (2 * self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = np.exp(-(dt ** 2) / (2 * self.sigma2)) * np.cos(
            self.omega * dt + self._phi(dt)
        )
        return self.field_strength * pulse


class gaussian_E_velocity(Laser):
    """Dysfunctional..."""

    def __init__(
        self, field_strength, omega, sigma, phase=0.0, t0=0.0, N=100.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0
        self.a = self.t0 - N * self.sigma
        self.b = self.t0 + N * self.sigma
        self.N = N

    @property
    def t0_at_center(self):
        return True

    def set_t0(self, t0):
        self.t0 = t0
        self.a = self.t0 - self.N * self.sigma
        self.b = self.t0 + self.N * self.sigma

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-(dt ** 2) / (2 * self.sigma2))

    def __call__(self, t):
        def f(time, sign):
            return erf(
                (self.sigma2 * self.omega + sign * 1j * time)
                / (np.sqrt(2 * self.sigma2))
            )

        dt = t - self.t0
        pulse = (
            0.5
            * np.sqrt(self.sigma * np.pi / 2)
            * np.exp(self.sigma2 * self.omega ** 2 / 2 - 1j * self._phi(t))
            * (
                np.exp(2 * 1j * self._phi(t)) * (f(self.a, -1) - f(t, -1))
                + f(self.a, 1)
                - f(t, 1)
            )
            * np.heaviside(t - self.a, 1.0)
            * np.heaviside(self.b - t, 1.0)
        )
        return self.field_strength * pulse


### sin lasers without envelope ################################


class noenv_velocity_plane_cos(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class noenv_velocity_plane_sin(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class noenv_velocity_dipole(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class noenv_length_dipole(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.omega
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


#######--#############--#####################--########--########

### sin2 envelope on E ##########################################


class sine_square_laser_length(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.field_strength
        )
        return pulse


class sine_square_laser_velocity(Laser):
    """sine square laser in velocity gauge"""

    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0

        def f(t_):
            f0 = (
                self.tprime
                * np.cos(t_ * (self.omega - 2 * np.pi / self.tprime) + self._phase(dt))
                / (self.omega * self.tprime - 2 * np.pi)
            )
            f1 = (
                self.tprime
                * np.cos(t_ * (self.omega + 2 * np.pi / self.tprime) + self._phase(dt))
                / (self.omega * self.tprime + 2 * np.pi)
            )
            f2 = 2 * np.cos(self._phase(dt)) * np.cos(self.omega * t_) / self.omega
            f3 = 2 * np.sin(self._phase(dt)) * np.sin(self.omega * t_) / self.omega
            return (1 / 4.0) * (-f0 - f1 + f2 - f3)

        pulse = (
            self.field_strength
            * (f(dt) - f(0))
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse


class sine_square_A(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            self.A0
            * np.cos(self.omega * dt + self._phase(dt))
            * (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse


class sine_square_E(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            self.A0
            * (
                self.omega
                * np.sin(self.omega * dt + self._phase(dt))
                * np.sin(np.pi * dt / self.tprime) ** 2
                - np.pi
                / self.tprime
                * np.cos(self.omega * dt + self._phase(dt))
                * np.sin(2 * np.pi * dt / self.tprime)
            )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse


class sine_square_derivative(Laser):
    def __init__(self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.field_strength
        )
        return pulse


class sin_sin2(Laser):
    def __init__(self, amplitude, omega, ncycles, phase=0.0, t0=0.0, **kwargs):
        self.amplitude = amplitude
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.amplitude
        )
        return pulse


#######--#############--#####################--########--########


class Lasers:
    def __init__(
        self,
        lasers,
        field_strength,
        omega,
        tprime,
        phase=None,
        start=None,
        sigma=None,
        **kwargs,
    ):
        n_pulses = len(lasers)
        if phase is None:
            phase = np.zeros_like(field_strength)
        if start is None:
            start = np.zeros_like(field_strength)
        self.lasers = [
            lasers[i](f, w, tp, phase=ph)
            for i, f, w, tp, ph in zip(
                np.arange(n_pulses), field_strength, omega, tprime, phase
            )
        ]
        for i in np.arange(n_pulses):
            self.lasers[i].set_t0(
                start[i] + 0.5 * float(self.lasers[i].t0_at_center) * tprime[i]
            )
            if sigma is not None:
                self.lasers[i].set_sigma(sigma[i])

    def A0(self, pulse_no):
        return self.lasers[pulse_no].A0

    def __call__(self, t):
        # pulse = np.zeros_like(t, dtype=np.complex128)
        pulse = 0
        for laser in self.lasers:
            pulse += laser(t)
        return pulse


class adiabatic_laser:
    def __init__(self, field_strength, omega, tau=3, half_time=25, **kwargs):
        self.field_strength = field_strength
        self.omega = omega
        self.tau = tau
        self.half_time = half_time

    def _Fermi(self, t):
        e = np.exp((t - self.half_time) / self.tau)
        return 1 - 1 / (1 + e)

    def __call__(self, t):
        return self.field_strength * np.cos(self.omega * t) * self._Fermi(t)


class gaussian_laser:
    def __init__(self, field_strength, omega, sigma, phase=0.0, center=0.0, **kwargs):
        self.field_strength = field_strength
        self.omega = omega
        self.sigma2 = sigma ** 2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = center

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-(dt ** 2) / (2 * self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = np.exp(-(dt ** 2) / (2 * self.sigma2)) * np.sin(
            self.omega * dt + self._phi(dt)
        )
        return self.field_strength * pulse


class gaussian_laser_cos:
    def __init__(self, field_strength, omega, center, sigma, N, **kwargs):
        self.F = field_strength
        self.omega = omega
        self.t0 = center
        self.sigma = sigma
        self.t_start = self.t0 - N * sigma
        self.t_end = self.t0 + N * sigma

    def temporal_range(self):
        return self.t_start, self.t_end

    def _cut(self, t):
        return np.heaviside(t - self.t_start, 1.0) * np.heaviside(self.t_end - t, 1.0)

    def _envelope(self, t):
        dt = t - self.t0
        s2 = self.sigma ** 2
        f = np.exp(-(dt ** 2) / (2 * s2)) * self._cut(t)
        return f

    def __call__(self, t):
        return self.F * np.cos(self.omega * (t - self.t0)) * self._envelope(t)


class gaussian_lasers_cos:
    def __init__(self, field_strength, omega, center, sigma, N, **kwargs):
        self.lasers = [
            gaussian_laser_cos(f, w, t0, s, n)
            for f, w, t0, s, n in zip(field_strength, omega, center, sigma, N)
        ]

    def __call__(self, t):
        pulse = np.zeros_like(t, dtype=np.float64)
        for laser in self.lasers:
            pulse += laser(t)
        return pulse


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.constants as const

    eV_to_Eh = const.e / const.physical_constants["atomic unit of energy"][0]

    F = [0.01, 0.1]
    omega = [3.55247 * eV_to_Eh, 57.6527 * eV_to_Eh]
    center = [90, 130]
    sigma = [20, 10]
    N = [4.5, 5]
    laser1 = gaussian_laser_cos(F[0], omega[0], center[0], sigma[0], N[0])
    laser2 = gaussian_laser_cos(F[1], omega[1], center[1], sigma[1], N[1])
    laser = gaussian_lasers_cos(F, omega, center, sigma, N)

    a0, b0 = laser1.temporal_range()
    delta = 1e-8
    fa = laser1(a0 + delta)
    fb = laser1(b0 - delta)
    print("Temporal range of pump laser: ", a0, b0, "Fields:", fa, fb)

    a1, b1 = laser2.temporal_range()
    delta = 1e-3
    fa = laser2(a1 + delta)
    fb = laser2(b1 - delta)
    print("Temporal range of probe laser:", a1, b1, "Fields:", fa, fb)

    t_start = min(a0, a1)
    t_end = max(b0, b1)
    dt = 0.1
    nsteps = int((t_end - t_start) / dt + 1)
    t = np.linspace(t_start, t_end, nsteps)

    fig1 = plt.figure()
    plt.plot(t, laser1(t), label="Pump")
    plt.legend()
    plt.grid()
    print("Pump: max neighbor diff =", np.diff(laser1(t)).max())

    fig2 = plt.figure()
    plt.plot(t, laser2(t), label="Probe")
    plt.legend()
    plt.grid()
    print("Probe: max neighbor diff =", np.diff(laser2(t)).max())

    fig3 = plt.figure()
    plt.plot(t, laser(t), label="Pump+Probe")
    plt.legend()
    plt.grid()
    print("Pump+probe: max neighbor diff =", np.diff(laser(t)).max())

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    wat

    F = 100.0
    omega = 2.87
    phase = -np.pi / 2
    center = 2.5
    fig = plt.figure()
    t = np.linspace(0, 5, 1000)
    TOL = 1e-6
    sigma = 0.4171830535594
    Gpulse = gaussian_laser(-F, omega, sigma, phase=phase, center=center)
    plt.plot(t, Gpulse(t), label=f"Gaussian, sigma={sigma}")
    Spulse = sine_square_laser(F, omega, 5.0, phase=phase, start=0.0)
    plt.plot(t, Spulse(t), label=f"Sin2")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

    t0_fs = const.physical_constants["atomic unit of time"][0] * 1e15
    field_strength = 1.0
    omega0 = 0.2
    phi0 = 0.0
    b = 0.001
    phase = time_dependent_phase(phi0=phi0, b=b)
    t_cycle = 2 * np.pi / omega0
    tprime = 2 * t_cycle
    t0 = 0.0
    t = np.linspace(0.0, t0 + tprime, 2000)
    chirped_laser = sine_square_laser(
        field_strength, omega0, tprime, phase=phase, start=t0
    )
    laser = sine_square_laser(field_strength, omega0, tprime, phase=phi0, start=t0)
    freq = omega0 + phase._frequency(t - t0)

    fig1 = plt.figure()
    plt.plot(t, chirped_laser(t), label="chirped sine-square laser")
    plt.plot(t, laser(t), label="sine-square laser")
    plt.legend()
    plt.grid()

    fig2 = plt.figure()
    plt.plot(t, freq, label="chirped sine-square laser frequency")
    plt.legend()
    plt.grid()

    plt.show()
    plt.close(fig1)
    plt.close(fig2)

    field_strength = [1.0, 1.0, 1.0]
    omega = [2.8, 0.9, 4.4]
    tprime = [5.0, 5.0, 5.0]
    phase = [0.0, 0.0, 0.0]
    start = [0.0, 5.0, 10.0]

    l = sine_square_lasers(field_strength, omega, tprime, phase=phase, start=start)

    l0 = sine_square_laser(
        field_strength[0], omega[0], tprime[0], phase=phase[0], start=start[0]
    )
    l1 = sine_square_laser(
        field_strength[1], omega[1], tprime[1], phase=phase[1], start=start[1]
    )
    l2 = sine_square_laser(
        field_strength[2], omega[2], tprime[2], phase=phase[2], start=start[2]
    )

    t = np.linspace(0.0, 15.0, 1500)
    assert np.allclose(l(t), l0(t) + l1(t) + l2(t)), "Laser mismatch"

    fig1 = plt.figure()
    plt.plot(t, l(t), label="sine_square superposition")
    plt.legend()
    plt.grid()

    fig2 = plt.figure()
    plt.plot(t, l0(t), label="sine_square laser 1")
    plt.legend()
    plt.grid()

    fig3 = plt.figure()
    plt.plot(t, l1(t), label="sine_square laser 2")
    plt.legend()
    plt.grid()

    fig4 = plt.figure()
    plt.plot(t, l2(t), label="sine_square laser 3")
    plt.legend()
    plt.grid()

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
