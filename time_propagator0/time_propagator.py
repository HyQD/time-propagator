import numpy as np
import sys
import importlib
import quantum_systems
from quantum_systems.time_evolution_operators import (
    DipoleFieldInteraction,
    CustomOneBodyOperator,
)
from coupled_cluster.rcc2 import RCC2, TDRCC2
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
import tqdm
import matplotlib.pyplot as plt
import lasers
from projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
    rccsd_overlap,
)
from helper_functions import compute_R0_ as compute_R0
import daltonproject as dp
import os

from utils import Inputs, OperatorClass

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)

from setup_daltonproject import (
    setup_system_da,
    setup_dp_dalton,
    setup_dp_molcas,
    get_amps,
    get_response_vectors,
)

import time


class TimePropagator:
    def __init__(self, method=None, molecule=None, n_electrons=None, inputs=None, **kwargs):
        """ inputs: str (input file name) or dict
        """
        #setup input parameters
        self.inputs = Inputs({})
        self.inputs.set_from_file('default_inputs')

        if inputs is not None:
            s = 'inputs must be either str or dict'
            assert (type(inputs) == str) or (type(inputs) == dict), s

            if type(inputs) == str:
                self.inputs.set_from_file(inputs)
            else:
                self.inputs.set_from_dict(inputs)

        self.inputs.set_from_dict(kwargs)

        if method is None:
            s = "'method' parameter must be defined at initiation"
            assert self.inputs.has_key('method'), s
        else:
            self.inputs.set('method',method)

        if molecule is None:
            s = "'molecule' parameter must be defined at initiation"
            assert self.inputs.has_key('molecule'), s
        else:
            self.inputs.set('molecule',molecule)

        if n_electrons is None:
            s = "'n_electrons' parameter must be defined at initiation"
            assert self.inputs.has_key('n_electrons'), s
        else:
            self.inputs.set('n_electrons',n_electrons)

        self.inputs.check_consistency()

        #setup molecule
        self.molecule = f'{inputs("molecule")}.xyz'

        #defining calculation type
        corr_methods = ["rcc2","rccsd","romp2","roaccd","ccs","cc2","ccsd","omp2","oaccd"]
        orb_adapt_methods = ["romp2", "roaccd", "omp2", "oaccd", "rhf"]
        restricted_methods = ["rcc2","rccsd","romp2","roaccd"]

        self.correlated = (
            True if self.inputs("method") in corr_methods else False
        )
        self.orbital_adaptive = (
            True if self.inputs("method") in orb_adapt_methods else False
        )
        self.restricted = (
            True if self.inputs("method") in restricted_methods else False
        )


    #def set_default_inputs(self):
    #    default_inputs = importlib.import_module("default_inputs").default_inputs
    #    for key in default_inputs:
    #        if not key in self.inputs:
    #            self.inputs.set(key,default_inputs[key])

    def setup_quantum_system(self, program):
        """program: 'pyscf' ,'dalton', """
        implemented = ['pyscf','dalton']
        if not program in implemented:
            raise NotImplementedError
        if program == 'pyscf':
            if (self.correlated) and (self.restricted):
                from quantum_systems import construct_pyscf_system_rhf
                QS = construct_pyscf_system_rhf(molecule=,basis=,charge=)
            elif (not self.correlated) and (self.restricted):
                from quantum_systems import construct_pyscf_system_ao
                QS = construct_pyscf_system_ao(molecule=,basis=,charge=)
        elif program == 'dalton':
            if (self.correlated) and (self.restricted):
                from setup_daltonproject import construct_dalton_system_rhf
                QS = construct_dalton_system_rhf(molecule=,basis=,charge=)
            elif (not self.correlated) and (self.restricted):
                from setup_daltonproject import construct_dalton_system_ao
                QS = construct_dalton_system_ao(molecule=,basis=,charge=)
        self.set_quantum_system(QS)

    def set_quantum_system(self,QS):
        """QS: QuantumSystem"""
        self.QS = QS


    def setup_response_vectors(self, program):
        """program: str"""
        implemented = ['dalton']
        if not program in implemented:
            raise NotImplementedError
        if program == 'dalton':
            da = setup_dp_dalton(
                self.input_file,
                self.inputs("basis"),
                self.inputs("n_excited_states"),
                self.inputs("n_electrons"),
                method=method,
                custom_basis=self.inputs("custom_basis"),
            )
        self.set_response_vectors(da)


    def set_response_vectors(self,da=None,L1=None,L2=None,R1=None,R2=None,M1=None,M2=None):
        """da : daltonproject.dalton.arrays.Arrays object"""
        if (    (L1 is not None)
            or  (L2 is not None)
            or  (R1 is not None)
            or  (R2 is not None)
            or  (M1 is not None)
            or  (M2 is not None)    ):
            raise NotImplementedError
        elif da not is None:
            self.da = da
            energies = self.da.state_energies
            excitation_energies = (self.energies - self.energies[0])[1:]
            self.nr_of_excited_states = len(self.excitation_energies)
        else:
            pass



    def setup_operator(self):
        inputs = self.inputs
        F_str = inputs("field_strength")
        omega = inputs("omega")
        k_direction = inputs("k_direction")
        polarization = inputs("polarization")
        time_after_pulse = inputs("time_after_pulse")
        ncycles = inputs("ncycles")
        sigma = inputs("sigma")
        phase = inputs("phase")

        dt = inputs("dt")
        init_time = inputs("initial_time")
        quadratic_terms = inputs("quadratic_terms")
        cross_terms = inputs("cross_terms")

        self.n_pulses = len(omega)
        tprime = []
        for i in np.arange(self.n_pulses):
            t_cycle = 2 * np.pi / omega[i]
            tprime.append(ncycles[i] * t_cycle)

        self.dt = dt
        self.polarization = polarization
        self.tprime = tprime

        Ru = polarization.real
        Iu = polarization.imag

        self.Ru = Ru
        self.Iu = Iu

        # generating start times for pulses (NOTE: if t0 is central time for pulse, this is adjusted in setup of the Lasers class):
        self.start_times = np.zeros(self.n_pulses)
        self.start_times[0] = init_time
        for i in np.arange(self.n_pulses - 1):
            self.start_times[i + 1] = (
                tprime[i] + time_after_pulse[i] + self.start_times[i]
            )

        self.tfinal = sum(tprime) + sum(time_after_pulse)
        self.num_steps = int(self.tfinal / self.dt) + 1

        laser_list = get_laser_list(self.inputs('pulse'))

        if self.inputs("laser_approx") == "dipole":
            operators = []
            Rpulses = []
            Ipulses = []
            for i in np.arange(self.n_pulses):
                if self.inputs("gauge") == "velocity":
                    Rpulse = lasers.Lasers(
                        [laser_list[i]],
                        [F_str[i]],
                        [omega[i]],
                        [tprime[i]],
                        phase=[phase[i]],
                        sigma=[sigma[i]],
                        start=[self.start_times[i]],
                    )
                    Ipulse = lasers.Lasers(
                        [laser_list[i]],
                        [F_str[i]],
                        [omega[i]],
                        [tprime[i]],
                        phase=[phase[i]-np.pi/2],
                        sigma=[sigma[i]],
                        start=[self.start_times[i]],
                    )
                    Rop = np.tensordot(
                        Ru[i],
                        self.QS.momentum,
                        axes=(0, 0),
                    )
                    Iop = np.tensordot(
                        Iu[i],
                        self.QS.momentum,
                        axes=(0, 0),
                    )

                elif self.inputs("gauge") == "length":
                    Rpulse = lasers.Lasers(
                        [laser_list[i]],
                        [F_str[i]],
                        [omega[i]],
                        [tprime[i]],
                        phase=[phase[i]],
                        sigma=[sigma[i]],
                        start=[self.start_times[i]],
                    )
                    Ipulse = lasers.Lasers(
                        [laser_list[i]],
                        [F_str[i]],
                        [omega[i]],
                        [tprime[i]],
                        phase=[phase[i]-np.pi/2],
                        sigma=[sigma[i]],
                        start=[self.start_times[i]],
                    )
                    Rop = np.tensordot(
                        Ru[i],
                        self.QS.position,
                        axes=(0, 0),
                    )
                    Iop = np.tensordot(
                        Iu[i],
                        self.QS.position,
                        axes=(0, 0),
                    )

                Ru_is_zero = True if np.max(np.abs(Ru[i])) < 1e-14 else False
                Iu_is_zero = True if np.max(np.abs(Iu[i])) < 1e-14 else False

                Rpulses.append(Rpulse)
                Ipulses.append(Ipulse)

                if not Ru_is_zero:
                    o0 = CustomOneBodyOperator(Rpulse, Rop)
                    operators.append(o0)
                if not Iu_is_zero:
                    o0 = CustomOneBodyOperator(Ipulse, Iop)
                    operators.append(o0)

                if self.inputs("gauge") == "velocity" and quadratic_terms:
                    op2 = 0.5 * np.eye(self.QS.l)
                    pulse2 = lambda x: Rpulse(x) ** 2
                    o1 = CustomOneBodyOperator(pulse2, op2)
                    operators.append(o1)

            if self.inputs("gauge") == "velocity" and cross_terms:
                pulse_nums = np.arange(self.n_pulses)
                for i in pulse_nums:
                    for j in pulse_nums[pulse_nums != i]:
                        RRop_ij = (
                            0.5
                            * np.eye(self.QS.l)
                            * np.dot(self.Ru[i], self.Ru[j])
                        )
                        RRpulse_ij = lambda x: Rpulses[i](x) * Rpulses[j](x)
                        RRo_ij = CustomOneBodyOperator(RRpulse_ij, RRop_ij)

                        IIop_ij = (
                            0.5
                            * np.eye(self.QS.l)
                            * np.dot(self.Iu[i], self.Iu[j])
                        )
                        IIpulse_ij = lambda x: Ipulses[i](x) * Ipulses[j](x)
                        IIo_ij = CustomOneBodyOperator(IIpulse_ij, IIop_ij)

                        RIop_ij = (
                            0.5
                            * np.eye(self.QS.l)
                            * np.dot(self.Ru[i], self.Iu[j])
                        )
                        RIpulse_ij = lambda x: Rpulses[i](x) * Ipulses[j](x)
                        RIo_ij = CustomOneBodyOperator(RIpulse_ij, RIop_ij)

                        IRop_ij = (
                            0.5
                            * np.eye(self.QS.l)
                            * np.dot(self.Iu[i], self.Ru[j])
                        )
                        IRpulse_ij = lambda x: Ipulses[i](x) * Rpulses[j](x)
                        IRo_ij = CustomOneBodyOperator(IRpulse_ij, IRop_ij)

                        RuRu_is_zero = True if np.max(np.abs(np.dot(self.Ru[i], self.Ru[j]))) < 1e-14 else False
                        IuIu_is_zero = True if np.max(np.abs(np.dot(self.Iu[i], self.Iu[j]))) < 1e-14 else False
                        RuIu_is_zero = True if np.max(np.abs(np.dot(self.Ru[i], self.Iu[j]))) < 1e-14 else False
                        IuRu_is_zero = True if np.max(np.abs(np.dot(self.Iu[i], self.Ru[j]))) < 1e-14 else False

                        if not RuRu_is_zero:
                            operators.append(RRo_ij)
                        if not IuIu_is_zero:
                            operators.append(IIo_ij)
                        if not RuIu_is_zero:
                            operators.append(RIo_ij)
                        if not IuRu_is_zero:
                            operators.append(IRo_ij)

            self.pulse = lasers.Lasers(
                laser_list,
                F_str,
                omega,
                tprime,
                phase=phase,
                sigma=sigma,
                start=self.start_times,
            )
            self.Rpulses = Rpulses
            self.Ipulses = Ipulses

            self.QS.set_time_evolution_operator(operators)

        if self.inputs("laser_approx") == "plane_wave" or self.inputs(
            "general_response"
        ):
            compute_A = True if (inputs("kinetic_momentum") == True) else False
            if self.correlated:
                C = self.da.c.T
            else:
                C = self.C
            oc = OperatorClass(
                C,
                self.input_file,
                self.inputs("basis"),
                self.n_pulses,
                self.QS.l,
                custom_basis=self.inputs("custom_basis"),
                change_basis=True
            )
            oc.construct_operators(
                inputs,
                laser_list,
                self.start_times,
                tprime,
                compute_A=compute_A,
            )
            self.oc = oc

        if self.inputs("laser_approx") == "plane_wave":
            operators = []
            for i in np.arange(oc.n_pulses):
                if not oc.linear_operator_is_zero(i, "cos", "real"):
                    o0 = CustomOneBodyOperator(
                        oc.linear_pulse(i, "cos"), oc.linear_operator(i, "cos", "real")
                    )
                    operators.append(o0)
                if not oc.linear_operator_is_zero(i, "sin", "real"):
                    o0 = CustomOneBodyOperator(
                        oc.linear_pulse(i, "sin"), oc.linear_operator(i, "sin", "real")
                    )
                    operators.append(o0)
                if not oc.linear_operator_is_zero(i, "cos", "imag"):
                    o0 = CustomOneBodyOperator(
                        oc.linear_pulse(i, "sin"), oc.linear_operator(i, "cos", "imag")
                    )
                    operators.append(o0)
                if not oc.linear_operator_is_zero(i, "sin", "imag"):
                    o0 = CustomOneBodyOperator(
                        oc.linear_pulse(i, "cos"), -oc.linear_operator(i, "sin", "imag")
                    )
                    operators.append(o0)

                #if quadratic_terms:
                #    o2 = CustomOneBodyOperator(
                #        oc.quadratic_pulse(i, i, "cos+"),
                #        oc.quadratic_operator(i, i, "cos+"),
                #    )
                #    o3 = CustomOneBodyOperator(
                #        oc.quadratic_pulse(i, i, "sin+"),
                #        oc.quadratic_operator(i, i, "sin+"),
                #    )
                #    o4 = CustomOneBodyOperator(
                #        oc.quadratic_pulse(i, i, "cos-"),
                #        oc.quadratic_operator(i, i, "cos-"),
                #    )
                #    operators.append(o2)
                #    operators.append(o3)
                #    operators.append(o4)

            if quadratic_terms:
                pulse_nums = np.arange(oc.n_pulses)
                for i in pulse_nums:
                    for j in pulse_nums:#[pulse_nums != i]:
                        g1i_g1j = oc.quadratic_pulse(i, j, "cos", "cos")
                        g1i_g2j = oc.quadratic_pulse(i, j, "cos", "sin")
                        g2i_g1j = oc.quadratic_pulse(i, j, "sin", "cos")
                        g2i_g2j = oc.quadratic_pulse(i, j, "sin", "sin")

                        uRR = np.dot(self.Ru[i],self.Ru[j])
                        uRI = np.dot(self.Ru[i],self.Iu[j])
                        uII = np.dot(self.Iu[i],self.Iu[j])
                        uIR = uRI

                        p_cos_p = lambda x: 0.5*( uRR*(g1i_g1j(x) - g2i_g2j(x))
                                                + uRI*(g1i_g2j(x) + g2i_g1j(x))
                                                + uIR*(g2i_g1j(x) + g1i_g2j(x))
                                                + uII*(g2i_g2j(x) - g1i_g1j(x)) )

                        p_cos_m = lambda x: 0.5*( uRR*(g1i_g1j(x) + g2i_g2j(x))
                                                + uRI*(g1i_g2j(x) - g2i_g1j(x))
                                                + uIR*(g2i_g1j(x) - g1i_g2j(x))
                                                + uII*(g2i_g2j(x) + g1i_g1j(x)) )

                        p_sin_p = lambda x: 0.5*( uRR*(g1i_g2j(x) + g2i_g1j(x))
                                                + uRI*(g2i_g2j(x) - g1i_g1j(x))
                                                + uIR*(g2i_g2j(x) - g1i_g1j(x))
                                                + uII*(g2i_g1j(x) + g1i_g2j(x)) )

                        p_sin_m = lambda x: 0.5*( uRR*(g2i_g1j(x) - g1i_g2j(x))
                                                + uRI*(g2i_g2j(x) + g1i_g1j(x))
                                                + uIR*(g2i_g2j(x) + g1i_g1j(x))
                                                + uII*(g2i_g1j(x) - g1i_g2j(x)) )


                        o0 = CustomOneBodyOperator(
                            p_cos_p,
                            oc.quadratic_operator(i, j, "cos+"),
                        )
                        o1 = CustomOneBodyOperator(
                            p_sin_p,
                            oc.quadratic_operator(i, j, "sin+"),
                        )
                        o2 = CustomOneBodyOperator(
                            p_cos_m,
                            oc.quadratic_operator(i, j, "cos-"),
                        )
                        o3 = CustomOneBodyOperator(
                            p_sin_m,
                            oc.quadratic_operator(i, j, "sin-"),
                        )
                        operators.append(o0)
                        operators.append(o1)
                        operators.append(o2)
                        operators.append(o3)

            self.QS.set_time_evolution_operator(operators)

    def setup_method(self):
        """set ground state and TD methods"""

        if method == "rcc2":
            from coupled_cluster.rcc2 import RCC2 as CC
            from coupled_cluster.rcc2 import TDRCC2 as TDCC
        elif method == "rccsd":
            from coupled_cluster.rccsd import RCCSD as CC
            from coupled_cluster.rccsd import TDRCCSD as TDCC
        elif method == "romp2":
            from optimized_mp2.romp2 import ROMP2 as CC
            from optimized_mp2.romp2 import TDROMP2 as TDCC
        elif method == "roaccd":
            from coupled_cluster.rccd import ROACCD as CC
            from coupled_cluster.rccd import ROATDCCD as TDCC
        elif method == "rhf":
            from hartree_fock.rhf import RHF as CC
            from hartree_fock.rhf import TDRHF as TDCC
        elif method == "ccs":
            from coupled_cluster.ccs import CCS as CC
            from coupled_cluster.ccs import TDCCS as TDCC
        elif method == "cc2":
            from coupled_cluster.cc2 import CC2 as CC
            from coupled_cluster.cc2 import TDCC2 as TDCC
        elif method == "ccsd":
            from coupled_cluster.ccsd import CCSD as CC
            from coupled_cluster.ccsd import TDCCSD as TDCC
        elif method == "omp2":
            from optimized_mp2.omp2 import OMP2 as CC
            from optimized_mp2.omp2 import TDOMP2 as TDCC
        elif method == "oaccd":
            from coupled_cluster.oaccd import OACCD as CC
            from coupled_cluster.oaccd import TDOACCD as TDCC

        cc_kwargs = dict(verbose=False)
        self.cc = CC(self.QS, **cc_kwargs)

        ground_state_tolerance = 1e-10

        if self.correlated:
            if self.orbital_adaptive:
                self.cc.compute_ground_state(
                    tol=ground_state_tolerance,
                    termination_tol=ground_state_tolerance,
                )
            else:
                self.cc.compute_ground_state(
                    t_kwargs=dict(tol=ground_state_tolerance),
                    l_kwargs=dict(tol=ground_state_tolerance),
                )
            self.amps0 = self.cc.get_amplitudes(get_t_0=True)
            self.y0 = self.amps0.asarray()
        else:
            self.cc.compute_ground_state(
                tol=ground_state_tolerance,
                change_system_basis=False
            )
            self.C = self.cc.C.copy()
            self.cc.change_basis()
            self.y0 = self.cc.C.ravel()
        self.tdcc = TDCC(self.QS)


    def setup_integrator(
        self, integrator=None, initial_state=None, **integrator_params
    ):

        #init_time = self.inputs('initial_time') if init_time is None else init_time
        if integrator is not None:
            self.inputs('integrator') = integrator
        if initial_state is not None:
            self.inputs('initial_state') = initial_state

        integrator = self.inputs('integrator')
        init_state = self.inputs('initial_state')
        init_time = self.inputs('initial_time')

        if len(integrator_params) == 0:
            if self.inputs.has_key('integrator_params'):
                integrator_params = self.inputs('integrator_params')

        self.r = complex_ode(self.tdcc).set_integrator(integrator, **integrator_params)
        if init_state is not None:
            if type(init_state) is str:
                init_state = np.load(init_state.replace(".npz", "") + ".npz")["state"]
            self.r.set_initial_value(init_state, init_time)
        else:
            self.r.set_initial_value(self.y0, init_time)


    def time_transformed_operator(self, a):
        """get time-transformed operator"""
        if not self.orbital_adaptive:
            return a
        else:
            if self.correlated:
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(self.r.y)
                return C_tilde @ a @ C
            else:
                C = self.r.y.reshape(self.QS.l, self.QS.l)
                C_tilde = C.conj()
                return C_tilde @ a @ C


    def compute_F_vpi(self, t, rho_qp):
        """computes the D_{pq}Z_{pq,j,m}"""
        F = np.zeros((2, 2, self.n_pulses), dtype=np.complex128)

        if self.orbital_adaptive:
            if self.correlated:
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(self.r.y)
            else:
                C = self.r.y.reshape(self.QS.l, self.QS.l)
                C_tilde = C.conj()
        else:
            C = C_tilde = None

        #F00,m and F01,m
        for m in np.arange(self.n_pulses):
            # cos (i=1)
            # NOTE: the linear_operator is already contracted with the polarization vector
            Z00_l = -1j * (self.oc.linear_operator(m, "cos", "real", C=C, C_tilde=C_tilde))

            Z00_q = 0
            for n in np.arange(self.n_pulses):
                Z00_q00 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        + self.oc.quadratic_operator(n, m, "cos+", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "cos")(t)
                )
                Z00_q11 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        + self.oc.quadratic_operator(n, m, "sin-", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "sin")(t)
                )
                Z00_q10 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        + self.oc.quadratic_operator(n, m, "sin-", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "cos")(t)
                )
                Z00_q01 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        + self.oc.quadratic_operator(n, m, "cos+", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "sin")(t)
                )

                Z00_q += (
                    -1j
                    *(np.dot(self.Ru[m], self.Ru[n])* Z00_q00
                    + np.dot(self.Ru[m], self.Ru[n])* Z00_q11
                    + np.dot(self.Ru[m], self.Iu[n])* Z00_q10
                    + np.dot(self.Ru[m], self.Iu[n])* Z00_q01
                    )
                )

            Z00 = Z00_l + Z00_q
            F[0, 0, m] = np.einsum("qp,pq ->", rho_qp, Z00)

            Z01_l = -1j * (self.oc.linear_operator(m, "cos", "imag", C=C, C_tilde=C_tilde))
            Z01_q = 0

            Z01_q += (
                -1j
                *(np.dot(self.Iu[m], self.Ru[n])* Z00_q00
                + np.dot(self.Iu[m], self.Ru[n])* Z00_q11
                + np.dot(self.Iu[m], self.Iu[n])* Z00_q10
                + np.dot(self.Iu[m], self.Iu[n])* Z00_q01
                )
            )

            Z01 = Z01_l + Z01_q
            F[0, 1, m] = np.einsum("qp,pq ->", rho_qp, Z01)

        #F11,m and F10,m
        for m in np.arange(self.n_pulses):
            # sin (i=2)
            Z11_l = -1j * (self.oc.linear_operator(m, "sin", "real", C=C, C_tilde=C_tilde))

            Z11_q = 0
            for n in np.arange(self.n_pulses):
                Z11_q00 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        - self.oc.quadratic_operator(n, m, "sin-", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "cos")(t)
                )
                Z11_q11 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        - self.oc.quadratic_operator(n, m, "cos+", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "sin")(t)
                )
                Z11_q10 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        - self.oc.quadratic_operator(n, m, "cos+", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "cos")(t)
                )
                Z11_q01 = (
                    0.5
                    * (
                        self.oc.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        - self.oc.quadratic_operator(n, m, "sin-", C=C, C_tilde=C_tilde)
                    )
                    * self.oc.linear_pulse(n, "sin")(t)
                )


                Z11_q += (
                    -1j
                    *(np.dot(self.Ru[m], self.Ru[n])* Z00_q00
                    + np.dot(self.Ru[m], self.Ru[n])* Z00_q11
                    + np.dot(self.Ru[m], self.Iu[n])* Z00_q10
                    + np.dot(self.Ru[m], self.Iu[n])* Z00_q01
                    )
                )

            Z11 = Z11_l + Z11_q
            F[1, 1, m] = np.einsum("qp,pq ->", rho_qp, Z11)

            Z10_l = -1j * (self.oc.linear_operator(m, "sin", "imag", C=C, C_tilde=C_tilde))
            Z10_q = 0

            Z10_q += (
                -1j
                *(np.dot(self.Iu[m], self.Ru[n])* Z11_q00
                + np.dot(self.Iu[m], self.Ru[n])* Z11_q11
                + np.dot(self.Iu[m], self.Iu[n])* Z11_q10
                + np.dot(self.Iu[m], self.Iu[n])* Z11_q01
                )
            )

            Z10 = Z10_l + Z10_q
            F[1, 0, m] = np.einsum("qp,pq ->", rho_qp, Z10)

        return F

    def compute_F(self, t, rho_qp):
        p = self.time_transformed_operator(self.QS.momentum)

        F = np.zeros((1, self.n_pulses), dtype=np.complex128)
        for m in np.arange(self.n_pulses):
            Z = -1j * np.tensordot(p, self.polarization[m], axes=(0, 0))
            F[:, m] = np.einsum("qp,pq ->", rho_qp, Z)
            if self.inputs("gauge") == "velocity":
                F[:, m] += -1j * self.inputs("n_electrons") * self.Rpulses[m](t)

        return F



    def setup_samples(self, samples=None):
        if samples is None:
            samples = {}
            # samples['time_points'] = np.linspace(0, self.tfinal, self.num_steps)
            samples["time_points"] = np.zeros(self.num_steps)
            samples["laser_pulse"] = (
                np.zeros(self.num_steps) if self.inputs("laser_pulse") else None
            )
            samples["dipole_moment"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("dipole_moment")
                else None
            )
            samples["quadrupole_moment"] = (
                np.zeros((self.num_steps, 6), dtype=np.complex128)
                if self.inputs("quadrupole_moment")
                else None
            )
            samples["momentum"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("momentum")
                else None
            )
            samples["kinetic_momentum"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("kinetic_momentum")
                else None
            )
            samples["EOM_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("EOM_projectors")
                else None
            )
            samples["EOM2_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("EOM2_projectors")
                else None
            )
            samples["LR_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("LR_projectors")
                else None
            )
            samples["auto_correlation"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("auto_correlation")
                else None
            )
            samples["energy"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("energy")
                else None
            )
            samples["dipole_response"] = (
                np.zeros((self.num_steps, 2, 2, self.n_pulses), dtype=np.complex128)
                if self.inputs("dipole_response")
                else None
            )
            samples["general_response"] = (
                np.zeros((self.num_steps, 2, 2, self.n_pulses), dtype=np.complex128)
                if self.inputs("general_response")
                else None
            )

        self.samples = samples

    def propagate(self):
        if not hasattr(self, "samples"):
            self.setup_samples()

        final_step = int(self.num_steps * self.inputs("stop_run_at")) - 1

        time_points = self.samples["time_points"]
        it_n = np.arange(len(time_points))

        compute_projectors = bool(
            self.inputs("EOM_projectors")
            + self.inputs("EOM2_projectors")
            + self.inputs("LR_projectors")
        )

        load_all_vectors = True if self.inputs("load_all_response_vectors") else False

        if compute_projectors:
            if load_all_vectors:
                L1, L2, R1, R2 = get_response_vectors(
                    self.da, self.nr_of_excited_states, M1=False, M2=False
                )
                if self.inputs("LR_projectors"):
                    M1, M2 = get_response_vectors(
                        self.da,
                        self.nr_of_excited_states,
                        L1=False,
                        L2=False,
                        R1=False,
                        R2=False,
                    )
            t, l = self.amps0
            t0, t1, t2, l1, l2 = t[0][0], t[1], t[2], l[0], l[1]
            t0, t1, t2, l1, l2 = t[0][0], t[1], t[2], l[0], l[1]
            t, l = self.amps0.from_array(self.r.y)
            t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]
            R0 = np.empty(self.nr_of_excited_states)
            for n in range(self.nr_of_excited_states):
                if not load_all_vectors:
                    R1n, R2n = get_response_vectors(
                        self.da,
                        self.nr_of_excited_states,
                        excitation_levels=[n + 1],
                        M1=False,
                        M2=False,
                        L1=False,
                        L2=False,
                    )
                    R0[n] = compute_R0(l1, l2, R1n[0], R2n[0])
                else:
                    R0[n] = compute_R0(l1, l2, R1[n], R2[n])

        i = 0
        i0 = 0

        self.samples["time_points"][i0] = self.r.t

        if final_step is None:
            f0 = self.num_steps - 1
        else:
            f0 = final_step

        if self.inputs('verbose'):
            print("Total number of simulation iterations: ", len(time_points))
            print("Initial step: ", i0)
            print("Final step: ", f0)
            print("Total number of run iterations: ", f0 - i0)


        for i_, _t in tqdm.tqdm(enumerate(time_points[i0:f0])):
            self.r.integrate(self.r.t + self.dt)
            if not self.r.successful():
                break

            # if i%10 == 0:
            #    print (f'{i} / {self.num_steps}')

            self.samples["time_points"][i + 1] = self.r.t


            # ENERGY
            if self.inputs("energy"):
                self.samples["energy"][i + 1] = self.tdcc.compute_energy(
                    self.r.t, self.r.y
                )

            # DIPOLE MOMENT
            if self.inputs("dipole_moment"):
                for j in range(3):
                    x = self.QS.dipole_moment[j].copy()
                    self.samples["dipole_moment"][
                        i + 1, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, x
                    )

            # CANONICAL MOMENTUM
            if self.inputs("momentum"):
                for j in range(3):
                    p = self.QS.momentum[j].copy()
                    self.samples["momentum"][
                        i + 1, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, p
                    )

            # KINETIC MOMENTUM
            if self.inputs("kinetic_momentum"):
                for j in range(3):
                    pi = self.QS.momentum[j].copy()
                    if self.inputs("gauge") == "velocity":
                        for k in np.arange(self.n_pulses):
                            if self.inputs("laser_approx") == "dipole":
                                pi += (
                                    np.eye(self.QS.l)
                                    * self.Rpulses[k](time_points[i + 1])
                                    * self.Ru[k][j]
                                )
                                pi += (
                                    np.eye(self.QS.l)
                                    * self.Ipulses[k](time_points[i + 1])
                                    * self.Iu[k][j]
                                )
                            else:
                                pi += (
                                    self.oc.linear_pulse(k, "cos")(time_points[i + 1])
                                    * self.oc.A_operator(k, "cos")
                                    + self.oc.linear_pulse(k, "sin")(time_points[i + 1])
                                    * self.oc.A_operator(k, "sin")
                                ) * self.polarization[k][j]
                    self.samples["kinetic_momentum"][
                        i + 1, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, pi
                    )

            # QUADRUPOLE MOMENTS
            if self.inputs("quadrupole_moment"):
                for j in range(6):
                    r2 = self.r2[j].copy()
                    self.samples["quadrupole_moment"][
                        i + 1, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, r2
                    )

            # SPECTRAL RESPONSE
            if self.inputs("dipole_response") or self.inputs("general_response"):
                rho_qp = self.tdcc.compute_one_body_density_matrix(self.r.t, self.r.y)
                if (
                    self.inputs("dipole_response")
                    and self.inputs("laser_approx") == "dipole"
                ):
                    self.samples["dipole_response"][i + 1, 0, :] = self.compute_F(
                        self.r.t, rho_qp
                    )
                if self.inputs("general_response"):
                    self.samples["general_response"][i + 1, :, :, :] = self.compute_F_vpi(
                        self.r.t, rho_qp
                    )

            if compute_projectors or self.inputs("auto_correlation"):
                t, l = self.amps0.from_array(self.r.y)
                t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

            # AUTO CORRELATION
            if self.inputs("auto_correlation"):
                self.samples["auto_correlation"][i + 1] = self.tdcc.compute_overlap(
                    self.r.t, self.y0, self.r.y
                )

            # EOM PROJECTORS
            if compute_projectors:
                for n in range(self.nr_of_excited_states):
                    if not load_all_vectors:
                        L1n, L2n, R1n, R2n = get_response_vectors(
                            self.da,
                            self.nr_of_excited_states,
                            excitation_levels=[n + 1],
                            M1=False,
                            M2=False,
                        )
                        L1n, L2n, R1n, R2n = L1n[0], L2n[0], R1n[0], R2n[0]
                    else:
                        L1n, L2n, R1n, R2n = L1[n], L2[n], R1[n], R2[n]

                    if self.inputs("EOM_projectors"):
                        self.samples["EOM_projectors"][
                            i + 1, n
                        ] = conventional_EOM_projector(
                            L1n,
                            L2n,
                            l1_t,
                            l2_t,
                            t0_t,
                            t1_t,
                            t2_t,
                            t1,
                            t2,
                            R0[n],
                            R1n,
                            R2n,
                        )

                    if self.inputs("EOM2_projectors"):
                        self.samples["EOM2_projectors"][
                            i + 1, n
                        ] = two_component_EOM_projector(
                            L1n,
                            L2n,
                            l1_t,
                            l2_t,
                            t0_t,
                            t1_t,
                            t2_t,
                            t1,
                            t2,
                            R0[n],
                            R1n,
                            R2n,
                        )

                    if self.inputs("LR_projectors"):
                        if not self.load_all_vectors:
                            M1n, M2n = get_response_vectors(
                                self.da,
                                self.nr_of_excited_states,
                                excitation_levels=[n + 1],
                                L1=False,
                                L2=False,
                                R1=False,
                                R2=False,
                            )
                            M1n, M2n = M1n[0], M2n[0]
                        else:
                            M1n, M2n = M1[n], M2[n]
                        self.samples["LR_projectors"][i + 1, n] = LR_projector(
                            M1n,
                            M2n,
                            L1n,
                            L2n,
                            l1,
                            l2,
                            l1_t,
                            l2_t,
                            t1,
                            t2,
                            t0_t,
                            t1_t,
                            t2_t,
                            R0[n],
                            R1n,
                            R2n,
                        )
            i += 1

        if self.inputs("laser_pulse"):
            if self.inputs("laser_approx") == "dipole":
                self.samples["laser_pulse"] = self.pulse(self.samples["time_points"])
            else:
                print(
                    'WARNING: Laser pulse can only be stored if laser_approx is "dipole"'
                )

        if self.inputs("final_state") or (f0 < self.num_steps - 1):
            self.samples["state"] = self.r.y
            self.samples["old_i"] = i
            self.samples["old_t"] = self.r.t

        return self.samples




def get_laser_list(pulses):
    n_pulses = len(pulses)
    laser_list = []
    for i in np.arange(n_pulses):
        laser_list.append(vars(lasers)[pulses[i]])
    return laser_list

