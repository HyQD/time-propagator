import numpy as np
from scipy.special import erf
import abc

class Pulse(metaclass=abc.ABCMeta):
    @property
    def t0_at_center(self):
        return False

    def set_t0(self,t0):
        self.t0 = t0

    def set_sigma(self,sigma):
        self.sigma = sigma
        self.sigma2 = sigma**2

    def set_tprime(self,tprime):
        self.tprime = tprime

    def A2E(self):
        pass

    def E2A(self):
        raise NotImplementedError

    def __call__(self,t):
        pass

class ProductPulse(Pulse):
    def __init__(F,G,**kwargs):
        self.F = F(**kwargs)
        self.G = G(**kwargs)

        self.return_function = lambda t: self.F(t)*self.G(t)

    def A2E(self):
        PP = self.copy()
        PP.return_function = lambda t: self.F(t)*self.G.derivative(t) + self.derivative.F(t)*self.G(t)
        return PP

    def __call__(self,t):
        dt = t - self.t0
        return self.return_function(dt)

class TimeDependentFunction:
    def __call__(self,t):
        return 0


class HarmonicPulse(Pulse):
    def __init__(self, E0, omega, tprime, phase=0., t0=0., field_type='E',return_derivative=False):
        if field_type == 'E':
            self.F0 = E0
        elif field_type == 'A':
            self.F0 = F_str/omega

        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0
        self.return_derivative = return_derivative

    def return_function(self,t):
        if not self.return_derivative:
            return self.G.envelope(t)*np.cos(self.omega*t+self.phase)
        else:
            return ( self.G.derivative*np.cos(self.omega*t+self.phase)
            + self.G.envelope*np.sin(self.omega*t+self.phase)

    def __call__(self,t):
        dt = t - self.t0
        return self.return_function(dt)

class DeltaPulse(Pulse):
    def __init__(self, E0, t0=0., field_type='E'):
    def __call__(self,t):
        pass







class Envelope(metaclass=abc.ABCMeta):
    def envelope(self,t):
        return envelope(t)
    def derivative(self,t):
        return derivative(t)

class Gaussian(Envelope):
    def __init__(self, sigma):
        self.sigma2 = sigma**2

    def envelope(self,t):
        return np.exp(-t**2/(2*self.sigma2))

    def derivative(self,t):
        return -t*np.exp(-t**2/(2*self.sigma2))/self.sigma2


class SineSquare(Envelope):
    def __init__(self, tprime):
        self.tprime = tprime

    def envelope(self,t):
        return ( (np.sin(np.pi * t / self.tprime) ** 2)
        * np.heaviside(t, 1.0)
        * np.heaviside(self.tprime - t, 1.0) )

    def derivative(self,t):
        return ( ( 2*np.pi*sin(np.pi*t/self.tprime)
        *np.cos(np.pi*t/self.tprime) )/self.tprime
        * np.heaviside(t, 1.0)
        * np.heaviside(self.tprime - t, 1.0) )




class time_dependent_polarization_vector:
    def __init__(self, p0, p1, t_cut):
        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)
        self.T = t_cut

    def __call__(self, t=0.):
        # NB! t must be a scalar here
        dt = t - self.T
        pol = ( self.p0 * np.heaviside(-dt, 1e0)
              + self.p1 * np.heaviside(dt, 1e0))
        return pol

class time_dependent_phase:
    def __init__(self, phi0=0., a=0., b=0., c=0.):
        self.phi0 = phi0
        self.a = a
        self.b = b
        self.c = c

    def _frequency(self, t):
        return self.a + self.b*t + self.c*t*t

    def __call__(self, t):
        t2 = t**2
        t3 = t2*t
        return self.phi0 + self.a*t + self.b*t2 + self.c*t3


### hard-coded delta pulse #########################################
class length_delta(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.sigma = sigma
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        start_t = 0.0025
        pulse = (
            self.F_str
            * np.heaviside(dt - start_t, 1.0)
            * np.heaviside((self.sigma + start_t) - dt, 1.0)
        )
        return pulse



class zero_pulse(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
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
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
        return pulse_cos,pulse_sin

class square_velocity_plane_cos(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def envelope(self,t):
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
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
            np.sin(np.pi*dt/self.tprime)*( self.omega*np.sin(np.pi*dt/self.tprime)*np.sin(self.omega*dt + self.phase)
            - (2*np.pi/self.tprime)*np.cos(np.pi*dt/self.tprime)*np.cos(self.omega*dt + self.phase) )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.A0
        )
        return pulse

#######--#############--#####################--########--########


### gaussian envelope on A ################################

class gaussian_A_plane_cos(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
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
            np.exp(-dt**2/(2*self.sigma2))
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse

class gaussian_A_plane_sin(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
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
            np.exp(-dt**2/(2*self.sigma2))
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse

class gaussian_A_velocity(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
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
            np.exp(-dt**2/(2*self.sigma2))
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse

class gaussian_A_length(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0
        self.sigma2 = sigma**2

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
            np.exp(-dt**2/(2*self.sigma2))
            * ( (dt/self.sigma2)*np.cos(self.omega * dt + self._phase(dt))
              + self.omega*np.sin(self.omega * dt + self._phase(dt))  )
            * self.A0
        )
        return pulse

#######--#############--#####################--########--########

### gaussian envelope on E ################################

class gaussian_E_length(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def set_t0(self,t0):
        self.t0 = t0

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-dt**2/(2*self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * np.sin(self.omega * dt + self._phi(dt))
        )
        return self.F_str*pulse

class gaussian_E_length_cos(Laser):
    def __init__(self, F_str, omega, sigma, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0

    @property
    def t0_at_center(self):
        return True

    def set_t0(self,t0):
        self.t0 = t0

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-dt**2/(2*self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * np.cos(self.omega * dt + self._phi(dt))
        )
        return self.F_str*pulse

class gaussian_E_velocity(Laser):
    """Dysfunctional..."""
    def __init__(self, F_str, omega, sigma, phase=0., t0=0., N=100.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.phi = phase
        self.call_phi = callable(self.phi)
        self.t0 = t0
        self.a = self.t0 - N*self.sigma
        self.b = self.t0 + N*self.sigma
        self.N = N

    @property
    def t0_at_center(self):
        return True

    def set_t0(self,t0):
        self.t0 = t0
        self.a = self.t0 - self.N*self.sigma
        self.b = self.t0 + self.N*self.sigma

    def _phi(self, t):
        if self.call_phi:
            return self.phi(t)
        else:
            return self.phi

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-dt**2/(2*self.sigma2))

    def __call__(self, t):
        def f(time,sign):
            return erf((self.sigma2*self.omega+sign*1j*time)/(np.sqrt(2*self.sigma2)))

        dt = t - self.t0
        pulse = (
            0.5*np.sqrt(self.sigma*np.pi/2)
            * np.exp(self.sigma2*self.omega**2/2-1j*self._phi(t))
            * (  np.exp(2*1j*self._phi(t)) * ( f(self.a,-1) - f(t,-1) )
            +  f(self.a,1) - f(t,1) )
            * np.heaviside(t - self.a, 1.0)
            * np.heaviside(self.b - t, 1.0)
        )
        return self.F_str*pulse




### sin lasers without envelope ################################


class noenv_velocity_plane_cos(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = ( np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse

class noenv_velocity_plane_sin(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = ( np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse



class noenv_velocity_dipole(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = ( np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class noenv_length_dipole(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = ( np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.omega*np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse







#######--#############--#####################--########--########

### sin2 envelope on E ##########################################

class sine_square_laser_length(Laser):
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
            * self.F_str
        )
        return pulse


class sine_square_laser_velocity(Laser):
    """sine square laser in velocity gauge"""
    def __init__(self, F_str, omega, tprime, phase=0., t0=0.):
        self.F_str = F_str
        self.A0 = F_str/omega
        self.omega = omega
        self.tprime = tprime
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
            f0 = self.tprime*np.cos(t_*(self.omega-2*np.pi/self.tprime)+self._phase(dt))/(self.omega*self.tprime-2*np.pi)
            f1 = self.tprime*np.cos(t_*(self.omega+2*np.pi/self.tprime)+self._phase(dt))/(self.omega*self.tprime+2*np.pi)
            f2 = 2*np.cos(self._phase(dt))*np.cos(self.omega*t_)/self.omega
            f3 = 2*np.sin(self._phase(dt))*np.sin(self.omega*t_)/self.omega
            return (1/4.)*(-f0-f1+f2-f3)

        pulse = (
            self.F_str*(f(dt)-f(0))
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse


#######--#############--#####################--########--########




class Lasers:
    def __init__(self, lasers, F_str, omega, tprime, phase=None, start=None, sigma=None):
        n_pulses = len(lasers)
        if phase is None: phase = np.zeros_like(F_str)
        if start is None: start = np.zeros_like(F_str)
        self.lasers = [lasers[i](f, w, tp, phase=ph) for i,f,w,tp,ph in zip(np.arange(n_pulses),F_str, omega, tprime, phase)]
        for i in np.arange(n_pulses):
            self.lasers[i].set_t0(start[i] + 0.5*float(self.lasers[i].t0_at_center)*tprime[i])
            if sigma is not None:
                self.lasers[i].set_sigma(sigma[i])

    def A0(self,pulse_no):
        return self.lasers[pulse_no].A0

    def __call__(self,t):
        #pulse = np.zeros_like(t, dtype=np.complex128)
        pulse = 0
        for laser in self.lasers:
            pulse += laser(t)
        return pulse


class adiabatic_laser:
    def __init__(self, F_str, omega, tau=3, half_time=25):
        self.F_str = F_str
        self.omega = omega
        self.tau = tau
        self.half_time = half_time

    def _Fermi(self, t):
        e = np.exp((t - self.half_time)/self.tau)
        return 1 - 1/(1 + e)

    def __call__(self, t):
        return self.F_str*np.cos(self.omega*t)*self._Fermi(t)

class gaussian_laser:
    def __init__(self, F_str, omega, sigma, phase=0., center=0.):
        self.F_str = F_str
        self.omega = omega
        self.sigma2 = sigma**2
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
        return np.exp(-dt**2/(2*self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * np.sin(self.omega * dt + self._phi(dt))
        )
        return self.F_str*pulse

class gaussian_laser_cos:
    def __init__(self, F_str, omega, center, sigma, N):
        self.F = F_str
        self.omega = omega
        self.t0 = center
        self.sigma = sigma
        self.t_start = self.t0 - N*sigma
        self.t_end = self.t0 + N*sigma

    def temporal_range(self):
        return self.t_start, self.t_end

    def _cut(self, t):
        return np.heaviside(t-self.t_start, 1.0)*np.heaviside(self.t_end-t, 1.0)

    def _envelope(self, t):
        dt = t - self.t0
        s2 = self.sigma**2
        f = np.exp(-dt**2/(2*s2))*self._cut(t)
        return f

    def __call__(self, t):
        return self.F*np.cos(self.omega*(t-self.t0))*self._envelope(t)

class gaussian_lasers_cos:
    def __init__(self,  F_str, omega, center, sigma, N):
        self.lasers = [gaussian_laser_cos(f, w, t0, s, n) for f,w,t0,s,n in zip(F_str, omega, center, sigma, N)]

    def __call__(self, t):
        pulse = np.zeros_like(t, dtype=np.float64)
        for laser in self.lasers:
            pulse += laser(t)
        return pulse



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scipy.constants as const

    eV_to_Eh = const.e/const.physical_constants['atomic unit of energy'][0]

    F = [0.01, 0.1]
    omega = [3.55247*eV_to_Eh, 57.6527*eV_to_Eh]
    center = [90, 130]
    sigma = [20, 10]
    N = [4.5, 5]
    laser1 = gaussian_laser_cos(F[0], omega[0], center[0], sigma[0], N[0])
    laser2 = gaussian_laser_cos(F[1], omega[1], center[1], sigma[1], N[1])
    laser = gaussian_lasers_cos(F, omega, center, sigma, N)

    a0, b0 = laser1.temporal_range()
    delta = 1e-8
    fa = laser1(a0+delta)
    fb = laser1(b0-delta)
    print("Temporal range of pump laser: ",a0,b0,"Fields:",fa,fb)

    a1, b1 = laser2.temporal_range()
    delta = 1e-3
    fa = laser2(a1+delta)
    fb = laser2(b1-delta)
    print("Temporal range of probe laser:",a1,b1,"Fields:",fa,fb)

    t_start = min(a0,a1)
    t_end = max(b0,b1)
    dt = 0.1
    nsteps = int((t_end-t_start)/dt + 1)
    t = np.linspace(t_start, t_end, nsteps)

    fig1 = plt.figure()
    plt.plot(t, laser1(t), label="Pump")
    plt.legend()
    plt.grid()
    print("Pump: max neighbor diff =",np.diff(laser1(t)).max())

    fig2 = plt.figure()
    plt.plot(t, laser2(t), label="Probe")
    plt.legend()
    plt.grid()
    print("Probe: max neighbor diff =",np.diff(laser2(t)).max())

    fig3 = plt.figure()
    plt.plot(t, laser(t), label="Pump+Probe")
    plt.legend()
    plt.grid()
    print("Pump+probe: max neighbor diff =",np.diff(laser(t)).max())

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    wat

    F = 100.
    omega = 2.87
    phase = -np.pi/2
    center = 2.5
    fig = plt.figure()
    t = np.linspace(0, 5, 1000)
    TOL = 1e-6
    sigma = 0.4171830535594
    Gpulse = gaussian_laser(-F, omega, sigma, phase=phase, center=center)
    plt.plot(t, Gpulse(t), label=f"Gaussian, sigma={sigma}")
    Spulse = sine_square_laser(F, omega, 5., phase=phase, start=0.)
    plt.plot(t, Spulse(t), label=f"Sin2")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

    t0_fs = const.physical_constants['atomic unit of time'][0]*1e15
    F_str = 1.
    omega0 = 0.2
    phi0 = 0.
    b = 0.001
    phase = time_dependent_phase(phi0=phi0, b=b)
    t_cycle = 2*np.pi/omega0
    tprime = 2*t_cycle
    t0 = 0.
    t = np.linspace(0.,t0+tprime, 2000)
    chirped_laser = sine_square_laser(F_str, omega0, tprime, phase=phase, start=t0)
    laser = sine_square_laser(F_str, omega0, tprime, phase=phi0, start=t0)
    freq = omega0 + phase._frequency(t-t0)

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

    F_str = [1.0, 1.0, 1.0]
    omega = [2.8, 0.9, 4.4]
    tprime = [5., 5., 5.]
    phase = [0., 0., 0.]
    start = [0., 5., 10.]

    l = sine_square_lasers(F_str, omega, tprime, phase=phase, start=start)

    l0 = sine_square_laser(F_str[0], omega[0], tprime[0], phase=phase[0], start=start[0])
    l1 = sine_square_laser(F_str[1], omega[1], tprime[1], phase=phase[1], start=start[1])
    l2 = sine_square_laser(F_str[2], omega[2], tprime[2], phase=phase[2], start=start[2])

    t = np.linspace(0.,15., 1500)
    assert np.allclose(l(t), l0(t)+l1(t)+l2(t)), "Laser mismatch"

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
