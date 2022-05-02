import numpy as np
from quantum_systems.time_evolution_operators import (
    CustomOneBodyOperator,
)
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
import tqdm

from time_propagator0 import lasers

from time_propagator0.projectors import (
    two_component_EOM_projector,
    conventional_EOM_projector,
    LR_projector,
    rccsd_overlap,
)
from time_propagator0.helper_functions import compute_R0_ as compute_R0

from time_propagator0.utils import Inputs, PlaneWaveOperators

from time_propagator0.setup_daltonproject import (
    setup_system_da,
    setup_dp_dalton,
    get_response_vectors,
)

import time


class TimePropagator:
    def __init__(self, method=None, inputs=None, **kwargs):
        """inputs: str (input file name) or dict"""
        # setup input parameters
        self.inputs = Inputs({})
        self.inputs.set_from_file("time_propagator0.default_inputs")

        if inputs is not None:
            s = "inputs must be either str or dict"
            assert (type(inputs) == str) or (type(inputs) == dict), s

            if type(inputs) == str:
                self.inputs.set_from_file(inputs)
            else:
                self.inputs.set_from_dict(inputs)

        self.inputs.set_from_dict(kwargs)

        if method is None:
            s = "'method' parameter must be defined at initiation"
            assert self.inputs.has_key("method"), s
        else:
            self.inputs.set("method", method)

        self.inputs.check_consistency()

        # defining calculation type
        corr_methods = [
            "rcc2",
            "rccsd",
            "romp2",
            "roaccd",
            "rcis",
            "ccs",
            "cc2",
            "ccsd",
            "omp2",
            "oaccd",
        ]
        orb_adapt_methods = ["romp2", "roaccd", "omp2", "oaccd", "rhf"]
        restricted_methods = ["rcc2", "rccsd", "romp2", "roaccd", "rhf", "rcis"]

        self.correlated = True if self.inputs("method") in corr_methods else False
        self.orbital_adaptive = (
            True if self.inputs("method") in orb_adapt_methods else False
        )
        self.restricted = True if self.inputs("method") in restricted_methods else False

        implemented_methods = [
            "rcc2",
            "rccsd",
            "romp2",
            "roaccd",
            "rhf",
            "rcis",
            "ccs",
            "cc2",
            "ccsd",
            "omp2",
            "oaccd",
        ]

        s = r"Computational method is not supported"
        assert self.inputs("method") in implemented_methods, s

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
        elif method == "rcis":
            from ci_singles import CIS as CC
            from ci_singles import TDCIS as TDCC
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

        self.CC = CC
        self.TDCC = TDCC

    def setup_quantum_system(self, program, molecule=None, charge=None, basis=None):
        """program: 'pyscf' ,'dalton',"""
        implemented = ["pyscf", "dalton"]
        if not program in implemented:
            raise NotImplementedError

        if molecule is not None:
            self.inputs.set("molecule", molecule)
        if basis is not None:
            self.inputs.set("basis", basis)
        if charge is not None:
            self.inputs.set("charge", charge)

        molecule = self.inputs("molecule")
        basis = self.inputs("basis")
        charge = self.inputs("charge")

        if program == "pyscf":
            if (self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_pyscf_system_rhf,
                )

                QS, self.C = construct_pyscf_system_rhf(
                    molecule=molecule,
                    basis=basis,
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                )
            elif (not self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import construct_pyscf_system_ao

                QS = construct_pyscf_system_ao(
                    molecule=molecule,
                    basis=basis,
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                )
            elif (self.correlated) and (not self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_pyscf_system_rhf,
                )

                QS, self.C = construct_pyscf_system_rhf(
                    molecule=molecule, basis=basis, charge=charge
                )
        elif program == "dalton":
            if (self.correlated) and (self.restricted):
                from time_propagator0.setup_daltonproject import construct_dalton_system

                QS, self.C = construct_dalton_system(
                    molecule=molecule, basis=basis, charge=charge, change_basis=True
                )
            elif (not self.correlated) and (self.restricted):
                from time_propagator0.setup_daltonproject import construct_dalton_system

                QS, C = construct_dalton_system(
                    molecule=molecule, basis=basis, charge=charge, change_basis=False
                )
        self.set_quantum_system(QS)

    def set_quantum_system(self, QS):
        """QS: QuantumSystem"""
        self.QS = QS

    def setup_response_vectors(self, program):
        """program: str"""
        implemented = ["dalton"]
        if not program in implemented:
            raise NotImplementedError
        if program == "dalton":
            da = setup_dp_dalton(
                self.input_file,
                self.inputs("basis"),
                self.inputs("n_excited_states"),
                self.inputs("n_electrons"),
                method=method,
                custom_basis=self.inputs("custom_basis"),
            )
        self.set_response_vectors(da)

    def set_response_vectors(
        self, da=None, L1=None, L2=None, R1=None, R2=None, M1=None, M2=None
    ):
        """da : daltonproject.dalton.arrays.Arrays object"""
        if (
            (L1 is not None)
            or (L2 is not None)
            or (R1 is not None)
            or (R2 is not None)
            or (M1 is not None)
            or (M2 is not None)
        ):
            raise NotImplementedError
        elif da is not None:
            self.da = da
            energies = self.da.state_energies
            excitation_energies = (self.energies - self.energies[0])[1:]
            self.nr_of_excited_states = len(self.excitation_energies)
        else:
            pass

    def setup_ground_state(self):
        """set ground state and TD methods"""

        cc_kwargs = dict(verbose=False)
        self.cc = self.CC(self.QS, **cc_kwargs)

        ground_state_tolerance = 1e-10

        if self.inputs("method") == "rcis":
            y0 = np.zeros(1 + self.QS.m * self.QS.n, dtype=np.complex128)
            y0[0] = 1.0
        elif self.correlated:
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
            y0 = self.cc.get_amplitudes(get_t_0=True).asarray()
        else:
            self.cc.compute_ground_state(
                tol=ground_state_tolerance, change_system_basis=False
            )
            y0 = self.cc.C.ravel()
        self.set_initial_state(y0)

    def set_initial_state(self, initial_state):
        if self.correlated:
            self.y0 = initial_state
        else:
            self.C = initial_state.reshape(self.QS.l, self.QS.l).copy()
            self.QS.change_basis(self.C)
            self.y0 = np.identity(len(self.C)).ravel()

    def construct_pulses(self):
        dipole_approx = True if self.inputs("laser_approx") == "dipole" else False

        Rpulses = []
        Ipulses = []

        Ru = np.atleast_2d(np.zeros((self.n_pulses, 3), dtype=float))
        Iu = np.atleast_2d(np.zeros((self.n_pulses, 3), dtype=float))

        c = 0
        for pulse in self.inputs("pulses"):
            inputs = self.inputs(pulse)

            Ru_ = (inputs["polarization"]).real
            Iu_ = (inputs["polarization"]).imag

            Ru_is_zero = True if np.max(np.abs(Ru_)) < 1e-14 else False
            Iu_is_zero = True if np.max(np.abs(Iu_)) < 1e-14 else False

            pulse_class = inputs["pulse_class"]

            Laser = vars(lasers)[pulse_class]

            if (not Ru_is_zero) or (not dipole_approx):
                Rpulse = Laser(**inputs)
            else:
                Rpulse = lambda t: 0

            if (not Iu_is_zero) or (not dipole_approx):
                imag_inputs = inputs.copy()
                if "phase" in imag_inputs:
                    imag_inputs["phase"] -= np.pi / 2
                else:
                    imag_inputs["phase"] = -np.pi / 2
                Ipulse = Laser(**imag_inputs)
            else:
                Ipulse = lambda t: 0

            Rpulses.append(Rpulse)
            Ipulses.append(Ipulse)

            Ru[c, :] = Ru_
            Iu[c, :] = Iu_

            c += 1

        return Rpulses, Ipulses, Ru, Iu

    def construct_linear_dipole_operators(self):
        dipole_operator = (
            self.QS.position if self.inputs("gauge") == "length" else self.QS.momentum
        )

        operators = []

        for i in np.arange(self.n_pulses):
            pulse_name = self.inputs("pulses")[i]

            inputs = self.inputs(pulse_name)

            Ru = (inputs["polarization"]).real
            Iu = (inputs["polarization"]).imag

            Ru_is_zero = True if np.max(np.abs(Ru)) < 1e-14 else False
            Iu_is_zero = True if np.max(np.abs(Iu)) < 1e-14 else False

            if not Ru_is_zero:
                Rpulse = self.Rpulses[i]
                Rop = np.tensordot(
                    Ru,
                    dipole_operator,
                    axes=(0, 0),
                )
                o0 = CustomOneBodyOperator(Rpulse, Rop)
                operators.append(o0)

            if not Iu_is_zero:
                imag_inputs = inputs.copy()
                if "phase" in imag_inputs:
                    imag_inputs["phase"] -= np.pi / 2
                else:
                    imag_inputs["phase"] = -np.pi / 2
                Ipulse = self.Ipulses[i]
                Iop = np.tensordot(
                    Iu[i],
                    dipole_operator,
                    axes=(0, 0),
                )
                o0 = CustomOneBodyOperator(Ipulse, Iop)
                operators.append(o0)

        return operators

    def construct_quadratic_dipole_operators(self):
        quadratic_dipole_operator = 0.5 * np.eye(self.QS.l)

        operators = []

        for i in np.arange(self.n_pulses):
            pulse_name = self.inputs("pulses")[i]

            inputs = self.inputs(pulse_name)

            Ru = (inputs["polarization"]).real
            Iu = (inputs["polarization"]).imag

            RuRu = np.dot(Ru, Ru)
            RuIu = np.dot(Ru, Iu)
            IuIu = np.dot(Iu, Iu)

            RuRu_is_zero = True if np.max(np.abs(RuRu)) < 1e-14 else False
            RuIu_is_zero = True if np.max(np.abs(RuIu)) < 1e-14 else False
            IuIu_is_zero = True if np.max(np.abs(IuIu)) < 1e-14 else False

            Rpulse = self.Rpulses[i]
            Ipulse = self.Ipulses[i]

            if not RuRu_is_zero:
                pulse = lambda x: RuRu * Rpulse(x) ** 2
                o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
                operators.append(o0)
            if not RuIu_is_zero:
                pulse = lambda x: 2 * RuIu * Rpulse(x) * Ipulse(x)
                o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
                operators.append(o0)
            if not IuIu_is_zero:
                pulse = lambda x: IuIu * Ipulse(x) ** 2
                o0 = CustomOneBodyOperator(pulse, quadratic_dipole_operator)
                operators.append(o0)

        return operators

    def construct_cross_dipole_operators(self):
        cross_dipole_operator = np.eye(self.QS.l)

        operators = []

        pulse_nums = np.arange(self.n_pulses)
        for i in pulse_nums:
            for j in pulse_nums[pulse_nums > i]:
                pulse_name_i = self.inputs("pulses")[i]
                pulse_name_j = self.inputs("pulses")[j]

                inputs_i = self.inputs(pulse_name_i)
                inputs_j = self.inputs(pulse_name_j)

                Ru_i = (inputs_i["polarization"]).real
                Iu_i = (inputs_i["polarization"]).imag
                Ru_j = (inputs_j["polarization"]).real
                Iu_j = (inputs_j["polarization"]).imag

                RuRu = np.dot(Ru_i, Ru_j)
                RuIu = np.dot(Ru_i, Iu_j)
                IuRu = np.dot(Iu_i, Ru_j)
                IuIu = np.dot(Iu_i, Iu_j)

                RuRu_is_zero = True if np.max(np.abs(RuRu)) < 1e-14 else False
                RuIu_is_zero = True if np.max(np.abs(RuIu)) < 1e-14 else False
                IuRu_is_zero = True if np.max(np.abs(IuRu)) < 1e-14 else False
                IuIu_is_zero = True if np.max(np.abs(IuIu)) < 1e-14 else False

                Rpulse_i = self.Rpulses[i]
                Ipulse_i = self.Ipulses[i]
                Rpulse_j = self.Rpulses[j]
                Ipulse_j = self.Ipulses[j]

                if not RuRu_is_zero:
                    pulse = lambda x: RuRu * Rpulse_i(x) * Rpulse_j(x)
                    o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                    operators.append(o0)
                if not RuIu_is_zero:
                    pulse = lambda x: RuIu * Rpulse_i(x) * Ipulse_j(x)
                    o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                    operators.append(o0)
                if not IuRu_is_zero:
                    pulse = lambda x: IuRu * Ipulse_i(x) * Rpulse_j(x)
                    o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                    operators.append(o0)
                if not IuIu_is_zero:
                    pulse = lambda x: IuIu * Ipulse(x) ** 2
                    o0 = CustomOneBodyOperator(pulse, cross_dipole_operator)
                    operators.append(o0)

        return operators

    def construct_linear_plane_wave_operators(self):
        operators = []

        pwo = self.pwo
        for i in np.arange(self.n_pulses):
            if not pwo.linear_operator_is_zero(i, "cos", "real"):
                o0 = CustomOneBodyOperator(
                    pwo.linear_pulse(i, "cos"), pwo.linear_operator(i, "cos", "real")
                )
                operators.append(o0)
            if not pwo.linear_operator_is_zero(i, "sin", "real"):
                o0 = CustomOneBodyOperator(
                    pwo.linear_pulse(i, "sin"), pwo.linear_operator(i, "sin", "real")
                )
                operators.append(o0)
            if not pwo.linear_operator_is_zero(i, "cos", "imag"):
                o0 = CustomOneBodyOperator(
                    pwo.linear_pulse(i, "sin"), pwo.linear_operator(i, "cos", "imag")
                )
                operators.append(o0)
            if not pwo.linear_operator_is_zero(i, "sin", "imag"):
                o0 = CustomOneBodyOperator(
                    pwo.linear_pulse(i, "cos"), -pwo.linear_operator(i, "sin", "imag")
                )
                operators.append(o0)

        return operators

    def construct_quadratic_plane_wave_operators(self):
        operators = []
        pwo = self.pwo

        pulse_nums = np.arange(self.n_pulses)

        for i in pulse_nums:
            g1i_g1j = pwo.quadratic_pulse(i, i, "cos", "cos")
            g1i_g2j = pwo.quadratic_pulse(i, i, "cos", "sin")
            g2i_g1j = pwo.quadratic_pulse(i, i, "sin", "cos")
            g2i_g2j = pwo.quadratic_pulse(i, i, "sin", "sin")

            uRR = np.dot(self.Ru[i], self.Ru[i])
            uRI = np.dot(self.Ru[i], self.Iu[i])
            uII = np.dot(self.Iu[i], self.Iu[i])
            uIR = uRI

            p_cos_p = lambda x: 0.5 * (
                uRR * (g1i_g1j(x) - g2i_g2j(x))
                + uRI * (g1i_g2j(x) + g2i_g1j(x))
                + uIR * (g2i_g1j(x) + g1i_g2j(x))
                + uII * (g2i_g2j(x) - g1i_g1j(x))
            )

            p_cos_m = lambda x: 0.5 * (
                uRR * (g1i_g1j(x) + g2i_g2j(x))
                + uRI * (g1i_g2j(x) - g2i_g1j(x))
                + uIR * (g2i_g1j(x) - g1i_g2j(x))
                + uII * (g2i_g2j(x) + g1i_g1j(x))
            )

            p_sin_p = lambda x: 0.5 * (
                uRR * (g1i_g2j(x) + g2i_g1j(x))
                + uRI * (g2i_g2j(x) - g1i_g1j(x))
                + uIR * (g2i_g2j(x) - g1i_g1j(x))
                + uII * (g2i_g1j(x) + g1i_g2j(x))
            )

            o0 = CustomOneBodyOperator(
                p_cos_p,
                pwo.quadratic_operator(i, i, "cos+"),
            )
            o1 = CustomOneBodyOperator(
                p_sin_p,
                pwo.quadratic_operator(i, i, "sin+"),
            )
            o2 = CustomOneBodyOperator(
                p_cos_m,
                pwo.quadratic_operator(i, i, "cos-"),
            )
            operators.append(o0)
            operators.append(o1)
            operators.append(o2)

        return operators

    def construct_cross_plane_wave_operators(self):
        operators = []
        pwo = self.pwo

        pulse_nums = np.arange(self.n_pulses)

        for i in pulse_nums:
            for j in pulse_nums[pulse_nums > i]:
                g1i_g1j = pwo.quadratic_pulse(i, j, "cos", "cos")
                g1i_g2j = pwo.quadratic_pulse(i, j, "cos", "sin")
                g2i_g1j = pwo.quadratic_pulse(i, j, "sin", "cos")
                g2i_g2j = pwo.quadratic_pulse(i, j, "sin", "sin")

                uRR = np.dot(self.Ru[i], self.Ru[j])
                uRI = np.dot(self.Ru[i], self.Iu[j])
                uII = np.dot(self.Iu[i], self.Iu[j])
                uIR = uRI

                p_cos_p = lambda x: 0.5 * (
                    uRR * (g1i_g1j(x) - g2i_g2j(x))
                    + uRI * (g1i_g2j(x) + g2i_g1j(x))
                    + uIR * (g2i_g1j(x) + g1i_g2j(x))
                    + uII * (g2i_g2j(x) - g1i_g1j(x))
                )

                p_cos_m = lambda x: 0.5 * (
                    uRR * (g1i_g1j(x) + g2i_g2j(x))
                    + uRI * (g1i_g2j(x) - g2i_g1j(x))
                    + uIR * (g2i_g1j(x) - g1i_g2j(x))
                    + uII * (g2i_g2j(x) + g1i_g1j(x))
                )

                p_sin_p = lambda x: 0.5 * (
                    uRR * (g1i_g2j(x) + g2i_g1j(x))
                    + uRI * (g2i_g2j(x) - g1i_g1j(x))
                    + uIR * (g2i_g2j(x) - g1i_g1j(x))
                    + uII * (g2i_g1j(x) + g1i_g2j(x))
                )

                p_sin_m = lambda x: 0.5 * (
                    uRR * (g2i_g1j(x) - g1i_g2j(x))
                    + uRI * (g2i_g2j(x) + g1i_g1j(x))
                    + uIR * (g2i_g2j(x) + g1i_g1j(x))
                    + uII * (g2i_g1j(x) - g1i_g2j(x))
                )

                o0 = CustomOneBodyOperator(
                    p_cos_p,
                    pwo.quadratic_operator(i, j, "cos+"),
                )
                o1 = CustomOneBodyOperator(
                    p_sin_p,
                    pwo.quadratic_operator(i, j, "sin+"),
                )
                o2 = CustomOneBodyOperator(
                    p_cos_m,
                    pwo.quadratic_operator(i, j, "cos-"),
                )
                o3 = CustomOneBodyOperator(
                    p_sin_m,
                    pwo.quadratic_operator(i, j, "sin-"),
                )
                operators.append(o0)
                operators.append(o1)
                operators.append(o2)
                operators.append(o3)

        return operators

    def build(self, integrator=None, **integrator_params):
        self.build_hamiltonian()
        self.build_integrator(integrator, **integrator_params)

    def build_hamiltonian(self):
        inputs = self.inputs

        dt = inputs("dt")
        init_time = inputs("initial_time")

        self.n_pulses = len(self.inputs("pulses"))

        total_time = self.inputs("final_time") - self.inputs("initial_time")
        self.num_steps = int(total_time / self.inputs("dt")) + 1

        self.Rpulses, self.Ipulses, self.Ru, self.Iu = self.construct_pulses()

        if self.inputs("laser_approx") == "dipole":
            operators = self.construct_linear_dipole_operators()

            if self.inputs("gauge") == "velocity":
                if self.inputs("quadratic_terms"):
                    operators += self.construct_quadratic_dipole_operators()
                if self.inputs("cross_terms"):
                    operators += self.construct_cross_dipole_operators()

        if self.inputs("laser_approx") == "plane_wave" or self.inputs(
            "sample_general_response"
        ):
            compute_A = True if (inputs("sample_kinetic_momentum") == True) else False

            pwo = PlaneWaveOperators(
                self.C,
                self.inputs("molecule"),
                self.inputs("basis"),
                self.Rpulses,
                self.Ipulses,
                self.QS.l,
                custom_basis=self.inputs("custom_basis"),
                change_basis=True,
            )
            pwo.construct_operators(
                inputs,
                compute_A=compute_A,
            )
            self.pwo = pwo

        if self.inputs("laser_approx") == "plane_wave":
            operators = self.construct_linear_plane_wave_operators()

            if self.inputs("quadratic_terms"):
                operators += self.construct_quadratic_plane_wave_operators()
            if self.inputs("cross_terms"):
                operators += self.construct_cross_plane_wave_operators()

        self.QS.set_time_evolution_operator(operators)

    def build_integrator(self, integrator=None, **integrator_params):

        self.tdcc = self.TDCC(self.QS)

        if integrator is not None:
            self.inputs.set("integrator", integrator)
        integrator = self.inputs("integrator")

        if (len(integrator_params) > 0) or (
            not self.inputs.has_key("integrator_params")
        ):
            self.inputs.set("integrator_params", integrator_params)
        integrator_params = self.inputs("integrator_params")

        self.r = complex_ode(self.tdcc).set_integrator(integrator, **integrator_params)
        self.r.set_initial_value(self.y0, self.inputs("initial_time"))

    def time_transformed_operator(self, a):
        """get time-transformed operator"""
        if not self.orbital_adaptive:
            return a
        else:
            if self.correlated:
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(
                    self.r.y
                )
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
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(
                    self.r.y
                )
            else:
                C = self.r.y.reshape(self.QS.l, self.QS.l)
                C_tilde = C.conj()
        else:
            C = C_tilde = None

        # F00,m and F01,m
        for m in np.arange(self.n_pulses):
            # cos (i=1)
            # NOTE: the linear_operator is already contracted with the polarization vector
            Z00_l = -1j * (
                self.pwo.linear_operator(m, "cos", "real", C=C, C_tilde=C_tilde)
            )

            Z00_q = 0
            for n in np.arange(self.n_pulses):
                Z00_q00 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        + self.pwo.quadratic_operator(
                            n, m, "cos+", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "cos")(t)
                )
                Z00_q11 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        + self.pwo.quadratic_operator(
                            n, m, "sin-", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "sin")(t)
                )
                Z00_q10 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        + self.pwo.quadratic_operator(
                            n, m, "sin-", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "cos")(t)
                )
                Z00_q01 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        + self.pwo.quadratic_operator(
                            n, m, "cos+", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "sin")(t)
                )

                Z00_q += -1j * (
                    np.dot(self.Ru[m], self.Ru[n]) * Z00_q00
                    + np.dot(self.Ru[m], self.Ru[n]) * Z00_q11
                    + np.dot(self.Ru[m], self.Iu[n]) * Z00_q10
                    + np.dot(self.Ru[m], self.Iu[n]) * Z00_q01
                )

            Z00 = Z00_l + Z00_q
            F[0, 0, m] = np.einsum("qp,pq ->", rho_qp, Z00)

            Z01_l = -1j * (
                self.pwo.linear_operator(m, "cos", "imag", C=C, C_tilde=C_tilde)
            )
            Z01_q = 0

            Z01_q += -1j * (
                np.dot(self.Iu[m], self.Ru[n]) * Z00_q00
                + np.dot(self.Iu[m], self.Ru[n]) * Z00_q11
                + np.dot(self.Iu[m], self.Iu[n]) * Z00_q10
                + np.dot(self.Iu[m], self.Iu[n]) * Z00_q01
            )

            Z01 = Z01_l + Z01_q
            F[0, 1, m] = np.einsum("qp,pq ->", rho_qp, Z01)

            # F11,m and F10,m
            # sin (i=2)
            Z11_l = -1j * (
                self.pwo.linear_operator(m, "sin", "real", C=C, C_tilde=C_tilde)
            )

            Z11_q = 0
            for n in np.arange(self.n_pulses):
                Z11_q00 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        - self.pwo.quadratic_operator(
                            n, m, "sin-", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "cos")(t)
                )
                Z11_q11 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        - self.pwo.quadratic_operator(
                            n, m, "cos+", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "sin")(t)
                )
                Z11_q10 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "cos-", C=C, C_tilde=C_tilde)
                        - self.pwo.quadratic_operator(
                            n, m, "cos+", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "cos")(t)
                )
                Z11_q01 = (
                    0.5
                    * (
                        self.pwo.quadratic_operator(n, m, "sin+", C=C, C_tilde=C_tilde)
                        - self.pwo.quadratic_operator(
                            n, m, "sin-", C=C, C_tilde=C_tilde
                        )
                    )
                    * self.pwo.linear_pulse(n, "sin")(t)
                )

                Z11_q += -1j * (
                    np.dot(self.Ru[m], self.Ru[n]) * Z00_q00
                    + np.dot(self.Ru[m], self.Ru[n]) * Z00_q11
                    + np.dot(self.Ru[m], self.Iu[n]) * Z00_q10
                    + np.dot(self.Ru[m], self.Iu[n]) * Z00_q01
                )

            Z11 = Z11_l + Z11_q
            F[1, 1, m] = np.einsum("qp,pq ->", rho_qp, Z11)

            Z10_l = -1j * (
                self.pwo.linear_operator(m, "sin", "imag", C=C, C_tilde=C_tilde)
            )
            Z10_q = 0

            Z10_q += -1j * (
                np.dot(self.Iu[m], self.Ru[n]) * Z11_q00
                + np.dot(self.Iu[m], self.Ru[n]) * Z11_q11
                + np.dot(self.Iu[m], self.Iu[n]) * Z11_q10
                + np.dot(self.Iu[m], self.Iu[n]) * Z11_q01
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

    def compute_dipole_vector_potential(self):
        A = np.zeros((3, self.QS.l, self.QS.l), dtype=np.complex128)

        for i in np.arange(3):
            A[i, :, :] = np.eye(self.QS.l)

        pulse = np.zeros(3)
        for i in np.arange(self.n_pulses):
            pulse += self.Ru[i, :] * self.Rpulses[i](self.r.t) + self.Iu[
                i, :
            ] * self.Ipulses[i](self.r.t)

        for i in np.arange(3):
            A[i, :, :] *= pulse[i]

        return A

    def compute_plane_wave_vector_potential(self):
        A = np.zeros((3, self.QS.l, self.QS.l), dtype=np.complex128)
        for i in np.arange(self.n_pulses):
            for j in np.arange(3):
                A[j] += self.Ru[i, j] * (
                    self.pwo.A_operator(i, "cos")
                    * self.pwo.linear_pulse(i, "cos")(self.r.t)
                    + self.pwo.A_operator(i, "sin")
                    * self.pwo.linear_pulse(i, "sin")(self.r.t)
                )
                A[j] += self.Iu[i, j] * (
                    self.pwo.A_operator(i, "cos")
                    * self.pwo.linear_pulse(i, "sin")(self.r.t)
                    - self.pwo.A_operator(i, "sin")
                    * self.pwo.linear_pulse(i, "cos")(self.r.t)
                )
        return A

    def setup_samples(self, samples=None):
        if samples is None:
            samples = {}
            # samples['time_points'] = np.linspace(0, self.tfinal, self.num_steps)
            samples["time_points"] = np.zeros(self.num_steps)
            samples["laser_pulse"] = (
                np.zeros((self.num_steps, 3))
                if self.inputs("sample_laser_pulse")
                else None
            )
            samples["dipole_moment"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("sample_dipole_moment")
                else None
            )
            samples["quadrupole_moment"] = (
                np.zeros((self.num_steps, 6), dtype=np.complex128)
                if self.inputs("sample_quadrupole_moment")
                else None
            )
            samples["momentum"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("sample_momentum")
                else None
            )
            samples["kinetic_momentum"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("sample_kinetic_momentum")
                else None
            )
            samples["EOM_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("sample_EOM_projectors")
                else None
            )
            samples["EOM2_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("sample_EOM2_projectors")
                else None
            )
            samples["LR_projectors"] = (
                np.zeros((self.num_steps, self.nr_of_excited_states))
                if self.inputs("sample_LR_projectors")
                else None
            )
            samples["auto_correlation"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("sample_auto_correlation")
                else None
            )
            samples["energy"] = (
                np.zeros((self.num_steps, 3), dtype=np.complex128)
                if self.inputs("sample_energy")
                else None
            )
            samples["dipole_response"] = (
                np.zeros((self.num_steps, 2, 2, self.n_pulses), dtype=np.complex128)
                if self.inputs("sample_dipole_response")
                else None
            )
            samples["general_response"] = (
                np.zeros((self.num_steps, 2, 2, self.n_pulses), dtype=np.complex128)
                if self.inputs("sample_general_response")
                else None
            )

        self.samples = samples

    def propagate(self):
        if not hasattr(self, "samples"):
            self.setup_samples()

        final_step = int(self.num_steps) - 1

        time_points = self.samples["time_points"]
        it_n = np.arange(len(time_points))

        compute_projectors = bool(
            self.inputs("sample_EOM_projectors")
            + self.inputs("sample_EOM2_projectors")
            + self.inputs("sample_LR_projectors")
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
            t, l = self.cc.get_amplitudes(get_t_0=True)
            t0, t1, t2, l1, l2 = t[0][0], t[1], t[2], l[0], l[1]
            t0, t1, t2, l1, l2 = t[0][0], t[1], t[2], l[0], l[1]
            t, l = self.cc.get_amplitudes(get_t_0=True).from_array(self.r.y)
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

        f0 = final_step

        if self.inputs("verbose"):
            print("Total number of simulation iterations: ", len(time_points))
            print("Initial step: ", i0)
            print("Final step: ", f0)
            print("Total number of run iterations: ", f0 - i0)

        for i_, _t in tqdm.tqdm(enumerate(time_points[i0 : f0 + 1])):
            # if i%10 == 0:
            #    print (f'{i} / {self.num_steps}')

            self.samples["time_points"][i] = self.r.t

            # ENERGY
            if self.inputs("sample_energy"):
                self.samples["energy"][i] = self.tdcc.compute_energy(self.r.t, self.r.y)

            # DIPOLE MOMENT
            if self.inputs("sample_dipole_moment"):
                for j in range(3):
                    x = self.QS.dipole_moment[j].copy()
                    self.samples["dipole_moment"][
                        i, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, x
                    )

            # CANONICAL MOMENTUM
            if self.inputs("sample_momentum"):
                for j in range(3):
                    p = self.QS.momentum[j].copy()
                    self.samples["momentum"][
                        i, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, p
                    )

            # KINETIC MOMENTUM
            if self.inputs("sample_kinetic_momentum"):
                pi = self.QS.momentum.copy()
                if self.inputs("gauge") == "velocity":
                    if self.inputs("laser_approx") == "dipole":
                        A = self.compute_dipole_vector_potential()
                    else:
                        A = self.compute_plane_wave_vector_potential()
                    pi += A
                for j in range(3):
                    self.samples["kinetic_momentum"][
                        i, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, pi[j]
                    )

            # QUADRUPOLE MOMENTS
            if self.inputs("sample_quadrupole_moment"):
                for j in range(6):
                    r2 = self.r2[j].copy()
                    self.samples["quadrupole_moment"][
                        i, j
                    ] = self.tdcc.compute_one_body_expectation_value(
                        self.r.t, self.r.y, r2
                    )

            # SPECTRAL RESPONSE
            if self.inputs("sample_dipole_response") or self.inputs(
                "sample_general_response"
            ):
                rho_qp = self.tdcc.compute_one_body_density_matrix(self.r.t, self.r.y)
                if (
                    self.inputs("sample_dipole_response")
                    and self.inputs("laser_approx") == "dipole"
                ):
                    self.samples["dipole_response"][i, 0, :] = self.compute_F(
                        self.r.t, rho_qp
                    )
                if self.inputs("sample_general_response"):
                    self.samples["general_response"][i, :, :, :] = self.compute_F_vpi(
                        self.r.t, rho_qp
                    )

            if compute_projectors or self.inputs("sample_auto_correlation"):
                t, l = self.cc.get_amplitudes(get_t_0=True).from_array(self.r.y)
                t0_t, t1_t, t2_t, l1_t, l2_t = t[0][0], t[1], t[2], l[0], l[1]

            # AUTO CORRELATION
            if self.inputs("sample_auto_correlation"):
                self.samples["auto_correlation"][i] = self.tdcc.compute_overlap(
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

                    if self.inputs("sample_EOM_projectors"):
                        self.samples["EOM_projectors"][
                            i, n
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

                    if self.inputs("sample_EOM2_projectors"):
                        self.samples["EOM2_projectors"][
                            i, n
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

                    if self.inputs("sample_LR_projectors"):
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
                        self.samples["LR_projectors"][i, n] = LR_projector(
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

            self.r.integrate(self.r.t + self.inputs("dt"))
            if not self.r.successful():
                break

            i += 1

        if self.inputs("sample_laser_pulse"):
            Ru = self.Ru
            Iu = self.Iu
            Rpulses = self.Rpulses
            Ipulses = self.Ipulses
            for i in np.arange(self.n_pulses):
                self.samples["laser_pulse"][:, 0] += Ru[i, 0] * Rpulses[i](
                    self.samples["time_points"]
                )
                self.samples["laser_pulse"][:, 0] += Iu[i, 0] * Ipulses[i](
                    self.samples["time_points"]
                )
                self.samples["laser_pulse"][:, 1] += Ru[i, 1] * Rpulses[i](
                    self.samples["time_points"]
                )
                self.samples["laser_pulse"][:, 1] += Iu[i, 1] * Ipulses[i](
                    self.samples["time_points"]
                )
                self.samples["laser_pulse"][:, 2] += Ru[i, 2] * Rpulses[i](
                    self.samples["time_points"]
                )
                self.samples["laser_pulse"][:, 2] += Iu[i, 2] * Ipulses[i](
                    self.samples["time_points"]
                )

        if self.inputs("return_final_state") or (f0 < self.num_steps - 1):
            self.samples["state"] = self.r.y
            self.samples["old_i"] = i
            self.samples["old_t"] = self.r.t

        return self.samples
