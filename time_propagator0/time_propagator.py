import numpy as np
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
import tqdm
import time
import copy

from time_propagator0.stationary_states.compute_projectors import (
    compute_R0,
)

from time_propagator0.stationary_states.stationary_states_containers import (
    CIStatesContainer,
    CCStatesContainerMemorySaver,
    setup_CCStatesContainer_from_dalton,
    setup_CCStatesContainerMemorySaver_from_dalton,
)

from time_propagator0.field_interaction.construct_operators import (
    construct_linear_dipole_operators,
    construct_quadratic_dipole_operators,
    construct_cross_dipole_operators,
    construct_linear_plane_wave_operators,
    construct_quadratic_plane_wave_operators,
    construct_cross_plane_wave_operators,
)

from time_propagator0.inputs import Inputs

from time_propagator0.utils import get_basis

from time_propagator0.compute_properties import (
    compute_expectation_value,
    compute_conventional_EOM_projectors,
    compute_two_component_EOM_projectors,
    compute_LR_projectors,
    compute_F,
)

from time_propagator0.setup_daltonproject import (
    compute_response_vectors_from_dalton,
)

from time_propagator0.field_interaction.pulses import (
    setup_Pulses,
)

from time_propagator0.field_interaction.plane_wave_integrals_containers import (
    IntegralContainerFixedOrbitals,
    IntegralContainerOrbitalAdaptive,
    setup_plane_wave_integrals_from_molcas,
)


class TimePropagator:
    def __init__(self, method=None, inputs=None, **kwargs):
        """inputs: str (input file name) or dict"""
        # setup input parameters
        self.inputs = Inputs({})
        self.inputs.set_from_file("time_propagator0.default_inputs")

        init_from_output = False

        if inputs is not None:
            s = "inputs must be either str or dict"
            assert (type(inputs) == str) or (type(inputs) == dict), s

            if type(inputs) == str and inputs[-3:] == ".py":
                self.inputs.set_from_file(inputs)

            elif type(inputs) == str and inputs[-4:] == ".npz":
                inputs = np.load(inputs, allow_pickle=True)

            if (type(inputs) == np.lib.npyio.NpzFile) or (
                type(inputs) == dict and "inputs" in inputs.keys()
            ):
                input_dict = (
                    copy.deepcopy(inputs["inputs"])
                    if type(inputs["inputs"]) == dict
                    else inputs["inputs"].item()
                )
                self.inputs.set_from_dict(input_dict)
                init_from_output = True

            elif type(inputs) == dict:
                self.inputs.set_from_dict(copy.deepcopy(inputs))

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
            "cis",
            "cid",
            "cisd",
            "cidt",
            "cisdt",
            "cidtq",
            "cisdtq",
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
            "cis",
            "cid",
            "cisd",
            "cidt",
            "cisdt",
            "cidtq",
            "cisdtq",
        ]

        if init_from_output:
            self.init_from_output(inputs)

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
        elif method == "cis":
            from configuration_interaction import CIS as CC
            from configuration_interaction import TDCIS as TDCC
        elif method == "cid":
            from configuration_interaction import CID as CC
            from configuration_interaction import TDCID as TDCC
        elif method == "cisd":
            from configuration_interaction import CISD as CC
            from configuration_interaction import TDCISD as TDCC
        elif method == "cidt":
            from configuration_interaction import CIDT as CC
            from configuration_interaction import TDCIDT as TDCC
        elif method == "cisdt":
            from configuration_interaction import CISDT as CC
            from configuration_interaction import TDCISDT as TDCC
        elif method == "cidtq":
            from configuration_interaction import CIDTQ as CC
            from configuration_interaction import TDCIDTQ as TDCC
        elif method == "cisdtq":
            from configuration_interaction import CISDTQ as CC
            from configuration_interaction import TDCISDTQ as TDCC

        self.CC = CC
        self.TDCC = TDCC

        if not hasattr(self, "_log"):
            self._log = ""

        s = f" - TimePropagator instance initiated with {self.inputs('method')} method."
        self.log(s)

    def log(self, s, printout=False):
        self._log += s
        if printout:
            print(s)

    def init_from_output(self, output):
        samples = (
            copy.deepcopy(output["samples"])
            if type(output["samples"]) == dict
            else output["samples"].item()
        )

        arrays = (
            copy.deepcopy(output["arrays"])
            if type(output["arrays"]) == dict
            else output["arrays"].item()
        )

        misc = (
            copy.deepcopy(output["misc"])
            if type(output["misc"]) == dict
            else output["misc"].item()
        )

        log = (
            copy.deepcopy(output["log"])
            if type(output["log"]) == str
            else output["log"].item()
        )

        self._log = log
        s = "\n - Initiated from a previous run."
        self.log(s)

        self.set_samples(samples)

        s = "state vector must have been stored to initiale from previous run"
        assert "state" in arrays, s

        self.set_initial_state(arrays["state"])

        self.iter = misc["iter"]

    def set_input(self, key, value):
        self.inputs.set(key, value)

        s = "\nAdded input {key}:{value}"
        self.log(s)

    def add_pulse(self, name, pulse_inputs):
        (self.inputs.inputs["pulses"]).append(name)
        self.inputs.set(name, pulse_inputs)

        s = "\nAdded pulse {name}"
        self.log(s)

    def setup_quantum_system(
        self, reference_program=None, molecule=None, charge=None, basis=None
    ):
        """reference_program: 'pyscf' ,'dalton',"""
        if molecule is not None:
            self.inputs.set("molecule", molecule)
        if basis is not None:
            self.inputs.set("basis", basis)
        if charge is not None:
            self.inputs.set("charge", charge)
        if reference_program is not None:
            self.inputs.set("reference_program", reference_program)

        implemented = ["pyscf", "dalton"]
        if not self.inputs("reference_program") in implemented:
            raise NotImplementedError

        molecule = self.inputs("molecule")
        basis = self.inputs("basis")
        charge = self.inputs("charge")

        program = self.inputs("reference_program")

        if program == "pyscf":
            if (self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_pyscf_system_rhf,
                )

                QS, C = construct_pyscf_system_rhf(
                    molecule=molecule,
                    basis=get_basis(basis, "pyscf"),
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                )

            elif (not self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_pyscf_system_rhf,
                )

                QS, C = construct_pyscf_system_rhf(
                    molecule=molecule,
                    basis=get_basis(basis, "pyscf"),
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                )

            elif (self.correlated) and (not self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_pyscf_system_rhf,
                )

                QS, C = construct_pyscf_system_rhf(
                    molecule=molecule, basis=get_basis(basis, "pyscf"), charge=charge
                )
        elif program == "dalton":
            if (self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_dalton_system_rhf,
                )

                QS, C = construct_dalton_system_rhf(
                    molecule=molecule,
                    basis=get_basis(basis, "dalton"),
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                    custom_basis=self.inputs("custom_basis"),
                )

            elif (not self.correlated) and (self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_dalton_system_rhf,
                )

                QS, C = construct_dalton_system_rhf(
                    molecule=molecule,
                    basis=get_basis(basis, "dalton"),
                    charge=charge,
                    add_spin=False,
                    anti_symmetrize=False,
                    custom_basis=self.inputs("custom_basis"),
                )

            elif (self.correlated) and (not self.restricted):
                from time_propagator0.custom_system_mod import (
                    construct_dalton_system_rhf,
                )

                QS, C = construct_dalton_system_rhf(
                    molecule=molecule,
                    basis=get_basis(basis, "dalton"),
                    charge=charge,
                    custom_basis=self.inputs("custom_basis"),
                )

        self.set_quantum_system(QS, C)

        s = f'\n - Setup quantum system using {self.inputs("reference_program")}'
        self.log(s)

    def set_quantum_system(self, QS, C=None):
        """QS: QuantumSystem
        C: HF coefficient matrix"""
        self.QS = QS
        if C is not None:
            self.C = C

        s = "\n - Set quantum system"
        self.log(s)

    def setup_projectors(self, EOMCC_program=None):
        """EOMCC_program: str"""
        implemented_methods = [
            "rcc2",
            "rccsd",
            "cis",
            "cid",
            "cisd",
            "cidt",
            "cisdt",
            "cidtq",
            "cisdtq",
        ]

        s = r'projectors setup is not implemented for {self.inputs("method")}'
        assert self.inputs("method") in implemented_methods

        if self.inputs("method")[:2] == "ci":
            cc_kwargs = dict(verbose=False)
            cc = self.CC(self.QS, **cc_kwargs)

            n = self.inputs("n_excited_states") + 1
            cc.compute_ground_state(k=n)

            self.set_projectors(C=cc.C)

            s = f'\n - Setup {self.inputs("method")} projectors'
            self.log(s)

        else:
            if EOMCC_program is not None:
                self.inputs.set("EOMCC_program", EOMCC_program)

            implemented_programs = ["dalton"]

            program = self.inputs("EOMCC_program")

            s = f"CC projectors setup is not implemented with {self.inputs('EOMCC_program')}"
            assert program in implemented_programs

            if program == "dalton":
                da = compute_response_vectors_from_dalton(
                    self.inputs("molecule"),
                    get_basis(self.inputs("basis"), "dalton"),
                    self.inputs("n_excited_states"),
                    self.inputs("charge"),
                    method=self.inputs("method"),
                    custom_basis=self.inputs("custom_basis"),
                )

                self.set_projectors(da=da)

                GS_energy = da.state_energies[0]
                ES_energies = (da.state_energies - da.state_energies[0])[1:]
                s = f"\n - Setup {self.inputs('method')} projectors using {self.inputs('EOMCC_program')}"
                s += f"\n - Ground state energy: {GS_energy}"
                s += f"\n - Excited state energies: {ES_energies}\n"
                self.log(s, self.inputs("verbose"))

    def set_projectors(
        self, da=None, L1=None, L2=None, R1=None, R2=None, M1=None, M2=None, C=None
    ):
        """da : daltonproject.dalton.arrays.Arrays object
        L1: left EOMCC eigenvectors, singles
        L2: left EOMCC eigenvectors, doubles
        R1: right EOMCC eigenvectors, singles
        R2: right EOMCC eigenvectors, doubles
        M1: M vector, singles
        M2: M vector, doubles
        C: CI coefficient matrix"""
        if (
            (L1 is not None)
            or (L2 is not None)
            or (R1 is not None)
            or (R2 is not None)
            or (M1 is not None)
            or (M2 is not None)
        ):
            self.ssc = CCStatesContainer(L1, L2, R1, R2, M1, M2)

        elif C is not None:
            self.ssc = CIStatesContainer(C)

        elif da is not None:
            if self.inputs("load_all_response_vectors"):
                self.ssc = setup_CCStatesContainer_from_dalton(
                    da, self.inputs("sample_LR_projectors")
                )
            else:
                self.ssc = setup_CCStatesContainerMemorySaver_from_dalton(da)

        s = f"\n - Set projectors"
        self.log(s)

    def setup_ground_state(self):
        """set ground state and TD methods"""

        cc_kwargs = dict(verbose=False)
        self.cc = self.CC(self.QS, **cc_kwargs)

        ground_state_tolerance = self.inputs("ground_state_tolerance")

        if self.inputs("method") == "rcis":
            y0 = np.zeros(1 + self.QS.m * self.QS.n, dtype=np.complex128)
            y0[0] = 1.0
        elif self.inputs("method")[:2] == "ci":
            self.cc.compute_ground_state(k=1)
            y0 = self.cc.C[:, 0]
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
            # self.cc.compute_ground_state(
            #    tol=ground_state_tolerance, change_system_basis=False
            # )
            # y0 = self.cc.C.ravel()
            y0 = np.identity(len(self.C)).ravel()
        self.set_initial_state(y0)

        s = f'\n - Setup {self.inputs("method")} ground state'
        self.log(s)

    def setup_initial_state(self, state_number=0):
        """state_number : int"""
        if hasattr(self, "ssc") and self.inputs("method")[:2] == "ci":
            self.set_initial_state(self.ssc.C[:, state_number])
        elif state_number == 0:
            self.setup_ground_state()
        else:
            raise NotImplementedError

        s = f'\n - Setup {self.inputs("method")} ground state'
        self.log(s)

    def set_initial_state(self, initial_state):
        self.y0 = initial_state

        s = f"\n - Set initial state"
        self.log(s)

    def setup_plane_wave_integrals(self, PWI_program=None):
        if PWI_program is not None:
            self.inputs.set("PWI_program", PWI_program)

        implemented_programs = ["molcas"]

        s = f"plane wave integral setup is not implemented with {self.inputs('PWI_program')}"
        assert self.inputs("PWI_program") in implemented_programs

        program = self.inputs("PWI_program")

        if program == "molcas":
            pulse_inputs = [self.inputs(el) for el in self.inputs("pulses")]
            integrals, index_mapping = setup_plane_wave_integrals_from_molcas(
                pulse_inputs,
                self.inputs("molecule"),
                get_basis(self.inputs("basis"), "molcas"),
                quadratic_terms=self.inputs("quadratic_terms"),
                cross_terms=self.inputs("cross_terms"),
                compute_A=self.inputs("sample_kinetic_momentum"),
                custom_basis=self.inputs("custom_basis"),
            )

        self.set_plane_wave_integrals(integrals, index_mapping)

        s = f"\n - Setup plane wave integrals using {program}"
        self.log(s)

    def set_plane_wave_integrals(self, integrals, index_mapping):
        """
        integrals : list containing integral arrays
        index_mapping : dict containing a mapping from
        a string defining the array to the list index.

        Convention for the index_mapping (m is pulse number
        and i is list index):
        cos(k_m.r): {'cos,m':i}
        sin(k_m.r): {'sin,m':i}
        cos(k_m.r).p: {'cosp,m':i}
        sin(k_m.r).p: {'sinp,m':i}
        cos(2[k_m+k_n].r).p: {'cos+,mn':i}
        sin(2[k_m+k_n].r).p: {'sin+,mn':i}
        cos(2[k_m-k_n].r).p: {'cos-,mn':i}
        sin(2[k_m-k_n].r).p: {'sin-,mn':i}

        cos(k_m.r) and sin(k_m.r) are only required if
        'sampling_kinetic_momentum'.

        Two-index terms: m=n terms are required if
        'quadratic_terms', while n>m terms are required
        if 'cross_terms'.
        """
        if self.orbital_adaptive:
            self.pwi = IntegralContainerOrbitalAdaptive(
                integrals, index_mapping, C=self.C, C_tilde=self.C.conj().T
            )
        else:
            self.pwi = IntegralContainerFixedOrbitals(
                integrals, index_mapping, C=self.C, C_tilde=self.C.conj().T
            )
        self.pwi.change_basis()

        s = f"\n - Set plane wave integrals"
        self.log(s)

    def setup_pulses(self):
        pulse_inputs = []
        for el in self.inputs("pulses"):
            pulse_inputs.append(self.inputs(el))
        pulses = setup_Pulses(pulse_inputs)
        self.set_pulses(pulses)

    def set_pulses(self, pulses):
        """Pulses class"""
        self.pulses = pulses

    def build(self, integrator=None, **integrator_params):
        compute_projectors = bool(
            self.inputs("sample_EOM_projectors")
            + self.inputs("sample_EOM2_projectors")
            + self.inputs("sample_LR_projectors")
            + self.inputs("sample_CI_projectors")
        )

        if not hasattr(self, "QS"):
            self.setup_quantum_system()
        if not hasattr(self, "y0"):
            self.setup_ground_state()
        if (compute_projectors) and (not hasattr(self, "ssc")):
            self.setup_projectors()

        self.build_hamiltonian()
        self.build_integrator(integrator, **integrator_params)
        self.build_samples()

        s = f"\n - Building TimePropagator ..."
        self.log(s)

    def build_hamiltonian(self):
        s = f"\n - Building Hamiltonian"
        self.log(s)

        if not hasattr(self, "pulses"):
            self.setup_pulses()

        if self.inputs("laser_approx") == "dipole":

            operators = construct_linear_dipole_operators(
                self.QS, self.pulses, self.inputs("gauge")
            )

            if self.inputs("gauge") == "velocity":
                if self.inputs("quadratic_terms"):
                    operators += construct_quadratic_dipole_operators(
                        self.QS, self.pulses
                    )
                if self.inputs("cross_terms"):
                    operators += construct_cross_dipole_operators(self.QS, self.pulses)

        if self.inputs("laser_approx") == "plane_wave" or self.inputs(
            "sample_general_response"
        ):
            self.setup_plane_wave_integrals()

        if self.inputs("laser_approx") == "plane_wave":
            operators = construct_linear_plane_wave_operators(self.pwi, self.pulses)

            if self.inputs("quadratic_terms"):
                operators += construct_quadratic_plane_wave_operators(
                    self.pwi, self.pulses
                )
            if self.inputs("cross_terms"):
                operators += construct_cross_plane_wave_operators(self.pwi, self.pulses)

        self.QS.set_time_evolution_operator(operators)

    def build_integrator(self, integrator=None, **integrator_params):
        s = f"\n - Building integrator"
        self.log(s)

        self.tdcc = self.TDCC(self.QS)

        if integrator is not None:
            self.inputs.set("integrator", integrator)
        integrator = self.inputs("integrator")

        if hasattr(self, "samples"):
            t0 = np.max(self.samples["time_points"]) + self.inputs("time_step")
        else:
            t0 = self.inputs("initial_time")

        if (len(integrator_params) > 0) or (
            not self.inputs.has_key("integrator_params")
        ):
            self.inputs.set("integrator_params", integrator_params)
        integrator_params = self.inputs("integrator_params")

        self.r = complex_ode(self.tdcc).set_integrator(integrator, **integrator_params)
        self.r.set_initial_value(self.y0, t0)

    def build_samples(self, samples=None):
        s = f"\n - Building samples"
        self.log(s)

        total_time = self.inputs("final_time") - self.inputs("initial_time")
        self.num_steps = int(total_time / self.inputs("time_step")) + 1
        if not hasattr(self, "samples"):
            self.setup_samples()
        else:
            self.update_samples()

    def compute_dipole_vector_potential(self):
        A = np.zeros((3, self.QS.l, self.QS.l), dtype=np.complex128)

        for i in np.arange(3):
            A[i, :, :] = np.eye(self.QS.l)

        pulse = np.zeros(3)
        for m in np.arange(self.pulses.n_pulses):
            pulse += self.pulses.Ru[m, :] * self.pulses.Rg[m](
                self.r.t
            ) + self.pulses.Iu[m, :] * self.pulses.Ig[m](self.r.t)

        for i in np.arange(3):
            A[i, :, :] *= pulse[i]

        return A

    def compute_plane_wave_vector_potential(self):
        A = np.zeros((3, self.QS.l, self.QS.l), dtype=np.complex128)
        for m in np.arange(self.pulses.n_pulses):
            for i in np.arange(3):
                A[i] += self.pulses.Ru[m, i] * (
                    self.pwi[f"cos,{m}"] * self.pulses.Rg[m](self.r.t)
                    + self.pwi[f"sin,{m}"] * self.pulses.Ig[m](self.r.t)
                )
                A[i] += self.pulses.Iu[m, i] * (
                    self.pwi[f"cos,{m}"] * self.pulses.Ig[m](self.r.t)
                    - self.pwi[f"sin,{m}"] * self.pulses.Rg[m](self.r.t)
                )
        return A

    def get_C(self):
        if self.orbital_adaptive:
            if self.correlated:
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(
                    self.r.y
                )
            else:
                C = self.r.y.reshape(self.QS.l, self.QS.l)
                C_tilde = C.conj().T
        else:
            C = C_tilde = None
        return C, C_tilde

    def setup_samples(self):
        samples = {}
        # samples['time_points'] = np.linspace(0, self.tfinal, self.num_steps)
        samples["time_points"] = np.zeros(self.num_steps)
        samples["laser_pulse"] = (
            np.zeros((self.num_steps, 3)) if self.inputs("sample_laser_pulse") else None
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
        samples["CI_projectors"] = (
            np.zeros((self.num_steps, self.ssc.n_states))
            if self.inputs("sample_CI_projectors")
            else None
        )
        samples["EOM_projectors"] = (
            np.zeros((self.num_steps, self.ssc.n_states))
            if self.inputs("sample_EOM_projectors")
            else None
        )
        samples["EOM2_projectors"] = (
            np.zeros((self.num_steps, self.ssc.n_states))
            if self.inputs("sample_EOM2_projectors")
            else None
        )
        samples["LR_projectors"] = (
            np.zeros((self.num_steps, self.ssc.n_states))
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
            np.zeros((self.num_steps, 2, 2, self.pulses.n_pulses), dtype=np.complex128)
            if self.inputs("sample_dipole_response")
            else None
        )
        samples["general_response"] = (
            np.zeros((self.num_steps, 2, 2, self.pulses.n_pulses), dtype=np.complex128)
            if self.inputs("sample_general_response")
            else None
        )

        self.set_samples(samples)

    def set_samples(self, samples):
        self.samples = samples

    def update_samples(self):
        n_time_points = len(self.samples["time_points"])
        if self.num_steps > n_time_points:
            add_n_points = self.num_steps - n_time_points

            for el in self.samples.keys():
                if not self.samples[el] is None:
                    s = list((self.samples[el]).shape)
                    s[0] = add_n_points
                    ext = np.zeros(s)
                    self.samples[el] = np.concatenate((self.samples[el], ext))

    def setup_R0(self):
        if not hasattr(self, "cc"):
            cc_kwargs = dict(verbose=False)
            self.cc = self.CC(self.QS, **cc_kwargs)
            self.cc.compute_ground_state(
                t_kwargs=dict(tol=self.inputs("ground_state_tolerance")),
                l_kwargs=dict(tol=self.inputs("ground_state_tolerance")),
            )
            print("recomputed_GS")
        t, l = self.cc.get_amplitudes(get_t_0=True)
        self.ssc.t = t
        self.ssc.l = l

        self.ssc.R0 = compute_R0(self.ssc)

    def checkpoint(self):
        if self.inputs("checkpoint_unit") == "iterations":
            if (not self.iter % self.inputs("checkpoint")) and (self.iter > 0):
                np.savez(f"tp_ckpt_{self.iter}", **self.get_output())

        if self.inputs("checkpoint_unit") == "hours":
            t = time.time() / 3600
            if t - self.ckpt_sys_time > self.inpits("checkpoint"):
                np.savez(f"tp_ckpt_{self.iter}", **self.get_output())
                self.ckpt_sys_time = time.time() / 3600

    def get_output(self):
        output = {
            "samples": self.samples,
            "inputs": self.inputs.inputs,
            "log": self._log,
            "arrays": {},
            "misc": {},
        }

        if self.inputs("return_state"):
            output["arrays"]["state"] = self.r.y
        if self.inputs("return_C"):
            output["arrays"]["C"] = self.C
        if self.inputs("return_stationary_states") and hasattr(self, "ssc"):
            # if self.inputs()
            # output["arrays_stationary_states"]
            pass
        if self.inputs("return_plane_wave_integrals"):
            pass

        output["misc"]["iter"] = self.iter

        return output

    def propagate(self):
        compute_CC_projectors = bool(
            self.inputs("sample_EOM_projectors")
            + self.inputs("sample_EOM2_projectors")
            + self.inputs("sample_LR_projectors")
        )

        if compute_CC_projectors and self.ssc.R0 is None:
            self.setup_R0()

        if not hasattr(self, "iter"):
            self.iter = 0

        i0 = self.iter
        f0 = int(self.num_steps)
        time_points = self.samples["time_points"]

        s = f"\n - Starting time propagation ..."
        s += f"\nTotal number of simulation iterations: {len(time_points)}"
        s += f"\nInitial time step: {i0}"
        s += f"\nFinal time step: {i0}"
        s += f"\nTotal number of run iterations: {f0 - i0}"
        self.log(s, self.inputs("verbose"))

        self.ckpt_sys_time = time.time() / 3600

        sys_time0 = time.time()

        for i_, _t in tqdm.tqdm(enumerate(time_points[i0:f0])):
            # if not i%10:
            #    print (f'{i} / {self.num_steps}')
            i = self.iter

            self.samples["time_points"][i] = self.r.t

            # ENERGY
            if self.inputs("sample_energy"):
                self.samples["energy"][i] = self.tdcc.compute_energy(self.r.t, self.r.y)

            # DIPOLE MOMENT
            if self.inputs("sample_dipole_moment"):
                M = self.QS.dipole_moment.copy()
                self.samples["dipole_moment"][i, :] = compute_expectation_value(
                    self.tdcc, self.r.t, self.r.y, M
                )

            # CANONICAL MOMENTUM
            if self.inputs("sample_momentum"):
                M = self.QS.momentum.copy()
                self.samples["momentum"][i, :] = compute_expectation_value(
                    self.tdcc, self.r.t, self.r.y, M
                )

            # KINETIC MOMENTUM
            if self.inputs("sample_kinetic_momentum"):
                M = self.QS.momentum.copy()
                if self.inputs("gauge") == "velocity":
                    if self.inputs("laser_approx") == "dipole":
                        M += self.compute_dipole_vector_potential()
                    else:
                        M += self.compute_plane_wave_vector_potential()
                self.samples["kinetic_momentum"][i, :] = compute_expectation_value(
                    self.tdcc, self.r.t, self.r.y, M
                )

            # QUADRUPOLE MOMENTS
            if self.inputs("sample_quadrupole_moment"):
                M = self.r2.copy()
                self.samples["quadrupole_moment"][i, :] = compute_expectation_value(
                    self.tdcc, self.r.t, self.r.y, M
                )

            # CI projectors
            if self.inputs("sample_CI_projectors"):
                self.samples["CI_projectors"][i, :] = compute_CI_projectors(
                    self.ssc, self.r.t, self.r.y
                )

            # AUTO CORRELATION
            if self.inputs("sample_auto_correlation"):
                self.samples["auto_correlation"][i] = self.tdcc.compute_overlap(
                    self.r.t, self.y0, self.r.y
                )

            if compute_CC_projectors:
                t, l = self.cc.get_amplitudes(get_t_0=True).from_array(self.r.y)

                # EOM PROJECTORS
                if self.inputs("sample_EOM_projectors"):
                    self.samples["EOM_projectors"][
                        i, :
                    ] = compute_conventional_EOM_projectors(self.ssc, t, l)

                # EOM2 PROJECTORS
                if self.inputs("sample_EOM2_projectors"):
                    self.samples["EOM2_projectors"][
                        i, :
                    ] = compute_two_component_EOM_projectors(self.ssc, t, l)

                # LR PROJECTORS
                if self.inputs("sample_LR_projectors"):
                    self.samples["LR_projectors"][i, :] = compute_LR_projectors(
                        self.ssc, t, l
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
                    self.samples["dipole_response"][i, 0, :] = self.compute_F_dipole(
                        self.r.t, rho_qp
                    )
                if self.inputs("sample_general_response"):
                    self.pwi.C, self.pwi.C_tilde = self.get_C()
                    self.samples["general_response"][i, :, :, :] = compute_F(
                        self.r.t, rho_qp, self.pulses, self.pwi
                    )

            self.r.integrate(self.r.t + self.inputs("time_step"))
            if not self.r.successful():
                break

            self.iter += 1

            # if self.inputs("checkpoint"):
            #    if (not i % self.inputs("checkpoint")) and (i > 0):
            #        np.savez(f"time_propagator_checkpoint_{i}", **self.get_output())

            if self.inputs("checkpoint"):
                self.checkpoint()

        if self.inputs("sample_laser_pulse"):
            self.samples["laser_pulse"][:, :] = self.pulses.pulses(
                self.samples["time_points"]
            )

        sys_time1 = time.time()

        s = f"\n - Time propagation finished"
        s += f"\n - Run time: {sys_time1-sys_time0} seconds"
        self.log(s, self.inputs("verbose"))

        return self.get_output()
