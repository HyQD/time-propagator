import numpy as np
from scipy.integrate import complex_ode
import tqdm
import time
import copy
import importlib
import operator


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

from time_propagator0.custom_system_mod import (
    QuantumSystemValues,
    construct_quantum_system,
)

from time_propagator0.inputs import Inputs, load_inputs, inspect_inputs

from time_propagator0.utils import get_basis

from time_propagator0.compute_properties import (
    compute_expectation_value,
    compute_conventional_EOM_projectors,
    compute_two_component_EOM_projectors,
    compute_LR_projectors,
    compute_CI_projectors,
    compute_R0,
    compute_dipole_vector_potential,
    compute_plane_wave_vector_potential,
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

from time_propagator0.logging import Logger, log_messages, style

from time_propagator0.lookup_tables import LookupTables


class TimePropagator:
    def __init__(self, method=None, inputs=None, **kwargs):
        """method: str, inputs: str (input file name), dict or list"""
        self.system = None
        self.system_values = None

        self.cc = None
        self.tdcc = None

        self.pulses = None

        self.states_container = None
        self.pwi_container = None

        self.iter = None
        self.num_steps = None

        self.y0 = None
        self.C = None

        self.one_body_sampling_operators = {}
        self.two_body_sampling_operators = {}

        self.samples = None

        inputs = copy.deepcopy(inputs)

        self.logger = Logger(log_messages, style)
        self.lookup = LookupTables()
        self.inputs = Inputs(
            input_requirements=self.lookup.input_requirements
        ).set_from_file("time_propagator0.default_inputs")

        inputs = load_inputs(inputs)
        valid_type, init_from_output, set_from_dict = inspect_inputs(inputs)

        if not valid_type:
            raise TypeError("Could not convert inputs to internal dict format.")

        if init_from_output:
            self._init_from_output(inputs)
        elif set_from_dict:
            self.inputs.set_from_dict(inputs)

        self.inputs.set_from_dict(kwargs)

        if method is not None:
            self.inputs.set("method", method)
        elif not self.inputs.has_key("method"):
            raise TypeError("Parameter 'method' must be defined at initiation.")

        method = self.inputs("method")

        if not method in self.lookup.methods:
            raise ValueError(f"Computational method {method} is not supported.")

        self.restricted = self.lookup.methods[method]["restricted"]
        self.correlated = self.lookup.methods[method]["correlated"]
        self.orbital_adaptive = self.lookup.methods[method]["orbital_adaptive"]

        module = self.lookup.methods[method]["module"]
        cc = self.lookup.methods[method]["cc"]
        tdcc = self.lookup.methods[method]["tdcc"]

        self.CC = getattr(importlib.import_module(module), cc)
        self.TDCC = getattr(importlib.import_module(module), tdcc)

        if not init_from_output:
            self.logger.log(
                self.inputs("print_level"), values=[self.inputs("method")], new_lines=0
            )

    def _init_from_output(self, inputs_):
        samples = copy.deepcopy(inputs_["samples"])
        inputs = copy.deepcopy(inputs_["inputs"])
        arrays = copy.deepcopy(inputs_["arrays"])
        misc = copy.deepcopy(inputs_["misc"])
        log = copy.deepcopy(inputs_["log"])

        self.set_samples(samples)

        self.inputs.set_from_dict(inputs)

        if not "state" in arrays:
            raise TypeError(
                "state vector must have been stored in order to initiate from a previous run."
            )

        self.set_initial_state(arrays["state"])

        self.iter = misc["iter"]

        self.logger.set_log(log)
        self.logger.log(self.inputs("print_level"))

    def set_input(self, key, value):
        self.inputs.set(key, value)

        self.logger.log(self.inputs("print_level"))

    def add_pulse(self, name, pulse_inputs):
        (self.inputs.inputs["pulses"]).append(name)
        self.inputs.set(name, pulse_inputs)

        self.logger.log(self.inputs("print_level"), values=[name])

    def setup_quantum_system(self, reference_program=None):
        """reference_program: 'pyscf' ,'dalton',"""
        if reference_program is not None:
            self.inputs.set("reference_program", reference_program)

        program = self.inputs("reference_program")

        if not program in self.lookup.reference_programs:
            raise TypeError(f"Quantum system setup is not implemented with {program}.")

        molecule = self.inputs("molecule")
        basis = self.inputs("basis")
        charge = self.inputs("charge")

        if program == "pyscf":
            from time_propagator0 import run_pyscf_rhf

            self.system_values = QuantumSystemValues().set_pyscf_values_rhf(
                run_pyscf_rhf(molecule, basis=basis, charge=charge)
            )
        elif program == "dalton":
            from time_propagator0 import run_dalton_rhf

            self.system_values = QuantumSystemValues().set_dalton_values_rhf(
                run_dalton_rhf(molecule, basis=basis, charge=charge)
            )

        if (self.correlated) and (not self.restricted):
            QS = construct_quantum_system(
                self.system_values, add_spin=True, anti_symmetrize=True
            )
        else:
            QS = construct_quantum_system(self.system_values)

        self.logger.log(self.inputs("print_level"), values=[program])
        self.logger.log(
            self.inputs("print_level"),
            name_ext="hf_energy",
            values=[self.system_values.hf_energy],
        )

        self.set_quantum_system(QS, self.system_values.C)

    def set_quantum_system(self, QS, C=None):
        """QS: QuantumSystem
        C: HF coefficient matrix"""
        self.system = QS
        if C is not None:
            self.C = C

        self.logger.log(self.inputs("print_level"))

    def setup_projectors(self, EOMCC_program=None):
        """EOMCC_program: str"""
        self.logger.log(self.inputs("print_level"), values=[self.inputs("method")])

        if self.inputs("method")[:2] == "ci":
            cc_kwargs = dict(verbose=False)
            cc = self.CC(self.system, **cc_kwargs)

            n = self.inputs("n_excited_states") + 1
            cc.compute_ground_state(k=n)

            self.set_projectors(C=cc.C)
            self.state_energies = cc.energies

        else:
            if EOMCC_program is not None:
                self.inputs.set("EOMCC_program", EOMCC_program)

            program = self.inputs("EOMCC_program")

            if not program in self.lookup.EOMCC_programs:
                raise TypeError(
                    f"CC projectors setup is not implemented with {program}."
                )

            if program == "dalton":
                da = compute_response_vectors_from_dalton(
                    self.inputs("molecule"),
                    get_basis(self.inputs("basis"), "dalton"),
                    self.inputs("n_excited_states"),
                    self.inputs("charge"),
                    method=self.inputs("method"),
                    custom_basis=self.inputs("custom_basis"),
                )

                self.state_energies = da.state_energies

                self.logger.log(
                    self.inputs("print_level"),
                    name_ext="EOMCC_program",
                    values=[self.inputs("EOMCC_program")],
                )

                self.set_projectors(da=da)

        self.logger.log(
            self.inputs("print_level"),
            name_ext="energies",
            values=[self.state_energies],
        )

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
            self.states_container = CCStatesContainer(L1, L2, R1, R2, M1, M2)

        elif C is not None:
            self.states_container = CIStatesContainer(C)

        elif da is not None:
            if self.inputs("load_all_response_vectors"):
                self.states_container = setup_CCStatesContainer_from_dalton(
                    da, self.inputs("sample_LR_projectors")
                )
            else:
                self.states_container = setup_CCStatesContainerMemorySaver_from_dalton(
                    da
                )

        self.logger.log(self.inputs("print_level"))

    def setup_ground_state(self):
        """set ground state and TD methods"""

        cc_kwargs = dict(verbose=False)
        self.cc = self.CC(self.system, **cc_kwargs)

        ground_state_tolerance = self.inputs("ground_state_tolerance")

        if self.inputs("method") == "rcis":
            y0 = np.zeros(1 + self.system.m * self.system.n, dtype=np.complex128)
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
            y0 = np.identity(len(self.C)).ravel()

        self.logger.log(self.inputs("print_level"))

        self.set_initial_state(y0)

    def setup_initial_state(self, state_number=0):
        """state_number : int"""
        self.logger.log(self.inputs("print_level"), values=[state_number])

        if self.states_container is not None and self.inputs("method")[:2] == "ci":
            self.set_initial_state(self.states_container.C[:, state_number])
        elif state_number == 0:
            self.setup_ground_state()
        else:
            raise NotImplementedError(
                "Setup of excited initial state is only \
            implemented with CI methods, and only if stationary states \
            have been calulated."
            )

    def set_initial_state(self, initial_state):
        self.y0 = initial_state

        self.logger.log(self.inputs("print_level"))

    def setup_plane_wave_integrals(self, PWI_program=None):
        if PWI_program is not None:
            self.inputs.set("PWI_program", PWI_program)

        program = self.inputs("PWI_program")

        if not program in self.lookup.PWI_programs:
            raise TypeError(
                f"Plane wave integral setup is not implemented with {program}."
            )

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

        self.logger.log(self.inputs("print_level"))

        self.set_plane_wave_integrals(integrals, index_mapping)

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
        cos(2[k_m+k_n].r): {'cos+,mn':i}
        sin(2[k_m+k_n].r): {'sin+,mn':i}
        cos(2[k_m-k_n].r): {'cos-,mn':i}
        sin(2[k_m-k_n].r): {'sin-,mn':i}

        cos(k_m.r) and sin(k_m.r) are only required if
        'sampling_kinetic_momentum'.

        Two-index terms: m=n terms are required if
        'quadratic_terms', while n>m terms are required
        if 'cross_terms'.
        """
        if self.orbital_adaptive:
            self.pwi_container = IntegralContainerOrbitalAdaptive(
                integrals, index_mapping, C=self.C, C_tilde=self.C.conj().T
            )
        else:
            self.pwi_container = IntegralContainerFixedOrbitals(
                integrals, index_mapping, C=self.C, C_tilde=self.C.conj().T
            )
        self.pwi_container.change_basis()

        self.logger.log(self.inputs("print_level"))

    def setup_pulses(self):
        pulse_inputs = []
        for el in self.inputs("pulses"):
            pulse_inputs.append(self.inputs(el))
        pulses = setup_Pulses(pulse_inputs)
        self.set_pulses(pulses)

        self.logger.log(self.inputs("print_level"))

    def set_pulses(self, pulses):
        """Pulses class"""
        self.pulses = pulses

        self.logger.log(self.inputs("print_level"))

    def build(self, integrator=None, **integrator_params):
        self.logger.log(self.inputs("print_level"))

        compute_projectors = bool(
            self.inputs("sample_EOM_projectors")
            + self.inputs("sample_EOM2_projectors")
            + self.inputs("sample_LR_projectors")
            + self.inputs("sample_CI_projectors")
        )

        if self.system is None:
            self.setup_quantum_system()
        if self.y0 is None:
            self.setup_ground_state()
        if (compute_projectors) and (self.states_container is None):
            self.setup_projectors()
        if (
            self.inputs("laser_approx") == "plane_wave"
            or self.inputs("sample_general_response")
        ) and (self.pwi_container is None):
            self.setup_plane_wave_integrals()

        self.build_hamiltonian()
        self.build_integrator(integrator, **integrator_params)
        self.build_samples()
        self.build_default_one_body_sampling_operators()

    def set_custom_one_body_sampling_operator(self, name, operator, change_basis=False):
        if not (isinstance(operator, np.ndarray) or isinstance(operator, np.ndarray)):
            raise TypeError(f"Operator must be of type numpy.ndarray or Operator")
        if self.samples is None:
            raise TypeError(
                f"Custom sampling operators can only be set after calling build()."
            )
        self.inputs.set_custom_input(f"sample_{name}", True)
        if isinstance(operator, np.ndarray):
            self.one_body_sampling_operators[name] = operator

    def build_default_one_body_sampling_operators(self):
        one_body_ops = self.lookup.find("sample_properties", "is_one_body_operator")
        for name, props in zip(one_body_ops, one_body_ops.values()):
            if self.inputs(props["sample_keyword"]):
                op = operator.attrgetter(props["operator_attr"])(self)
                self.one_body_sampling_operators[name] = copy.copy(
                    op if callable(op) else lambda t, op=op: op
                )

    def build_hamiltonian(self):
        self.logger.log(self.inputs("print_level"))

        if self.pulses is None:
            self.setup_pulses()

        if self.inputs("laser_approx") == "dipole":

            operators = construct_linear_dipole_operators(
                self.system, self.pulses, self.inputs("gauge")
            )

            if self.inputs("gauge") == "velocity":
                if self.inputs("quadratic_terms"):
                    operators += construct_quadratic_dipole_operators(
                        self.system, self.pulses
                    )
                if self.inputs("cross_terms"):
                    operators += construct_cross_dipole_operators(
                        self.system, self.pulses
                    )

        if self.inputs("laser_approx") == "plane_wave":
            operators = construct_linear_plane_wave_operators(
                self.pwi_container, self.pulses
            )

            if self.inputs("quadratic_terms"):
                operators += construct_quadratic_plane_wave_operators(
                    self.pwi_container, self.pulses
                )
            if self.inputs("cross_terms"):
                operators += construct_cross_plane_wave_operators(
                    self.pwi_container, self.pulses
                )

        self.system.set_time_evolution_operator(operators)

    def build_integrator(
        self, integrator=None, integrator_module=None, **integrator_params
    ):
        self.logger.log(self.inputs("print_level"))

        if integrator is not None:
            self.inputs.set("integrator", integrator)
        if integrator is not None:
            self.inputs.set("integrator_module", integrator_module)
        if (len(integrator_params) > 0) or (
            not self.inputs.has_key("integrator_params")
        ):
            self.inputs.set("integrator_params", integrator_params)

        integrator = self.inputs("integrator")
        integrator_module = self.inputs("integrator_module")
        integrator_params = self.inputs("integrator_params")

        if self.samples is not None:
            t0 = np.max(self.samples["time_points"]) + self.inputs("time_step")
        else:
            t0 = self.inputs("initial_time")

        if integrator_module is not None:
            name = integrator
            module = integrator_module
        else:
            name = self.lookup.integrators[integrator]["name"]
            module = self.lookup.integrators[integrator]["module"]

        if module is not None:
            getattr(importlib.import_module(module), name)

        self.tdcc = self.TDCC(self.system)

        self.r = complex_ode(self.tdcc).set_integrator(name, **integrator_params)
        self.r.set_initial_value(self.y0, t0)

    def build_samples(self, samples=None):
        self.logger.log(self.inputs("print_level"))

        total_time = self.inputs("final_time") - self.inputs("initial_time")
        self.num_steps = int(total_time / self.inputs("time_step")) + 1
        if self.samples is None:
            self.setup_samples()
        else:
            self.update_samples()

    def setup_samples(self):
        samples = {}
        samples["time_points"] = np.zeros(self.num_steps)

        sample_prop = self.lookup.sample_properties
        for key, prop in zip(sample_prop, sample_prop.values()):
            if not self.inputs(prop["sample_keyword"]):
                continue
            dim = self._get_dynamic_sample_dim(prop["dim"])
            samples[key] = np.zeros(dim, dtype=prop["dtype"])

        self.set_samples(samples)

    def set_samples(self, samples):
        self.samples = samples

    def update_samples(self):
        n_time_points = len(self.samples["time_points"])
        if self.num_steps > n_time_points:
            add_n_points = self.num_steps - n_time_points

            for el in self.samples.keys():
                s = list((self.samples[el]).shape)
                s[0] = add_n_points
                ext = np.zeros(s)
                self.samples[el] = np.concatenate((self.samples[el], ext))

    def checkpoint(self):
        if self.inputs("checkpoint_unit") == "iterations":
            if (not self.iter % self.inputs("checkpoint")) and (self.iter > 0):
                np.savez(
                    f"{self.inputs('checkpoint_name')}_{self.iter}", **self.get_output()
                )

        if self.inputs("checkpoint_unit") == "hours":
            t = time.time() / 3600
            if t - self.ckpt_sys_time > self.inpits("checkpoint"):
                np.savez(
                    f"{self.inputs('checkpoint_name')}_{self.iter}", **self.get_output()
                )
                self.ckpt_sys_time = time.time() / 3600

        self.logger.log(self.inputs("print_level"), values=[self.iter])

    def get_output(self):
        output = {
            "samples": self.samples,
            "inputs": self.inputs.inputs,
            "log": self.logger._log,
            "arrays": {},
            "misc": {},
        }

        if self.inputs("return_state"):
            output["arrays"]["state"] = self.r.y
        if self.inputs("return_C"):
            output["arrays"]["C"] = self.C
        if (
            self.inputs("return_stationary_states")
            and self.states_container is not None
        ):
            # if self.inputs()
            # output["arrays_stationary_states"]
            pass
        if self.inputs("return_plane_wave_integrals"):
            pass

        output["misc"]["iter"] = self.iter

        return output

    def _get_dynamic_sample_dim(self, dim):
        dim_ = list(dim)
        n = len(dim)
        for i in [j for j in range(n) if isinstance(dim_[j], str)]:
            dim_[i] = operator.attrgetter(dim[i])(self)

        return [self.num_steps] + dim_

    def _get_C(self):
        if self.orbital_adaptive:
            if self.correlated:
                t_amps, l_amps, C, C_tilde = self.tdcc._amp_template.from_array(
                    self.r.y
                )
            else:
                C = self.r.y.reshape(self.system.l, self.system.l)
                C_tilde = C.conj().T
        else:
            C = C_tilde = None
        return C, C_tilde

    def _compute_R0(self):
        if self.cc is None:
            cc_kwargs = dict(verbose=False)
            self.cc = self.CC(self.system, **cc_kwargs)
            self.cc.compute_ground_state(
                t_kwargs=dict(tol=self.inputs("ground_state_tolerance")),
                l_kwargs=dict(tol=self.inputs("ground_state_tolerance")),
            )
        t, l = self.cc.get_amplitudes(get_t_0=True)
        self.states_container.t = t
        self.states_container.l = l

        return compute_R0(self.states_container)

    def _compute_dipole_vector_potential(self, t):
        return compute_dipole_vector_potential(self.system, self.pulses, t)

    def _compute_plane_wave_vector_potential(self, t):
        return compute_plane_wave_vector_potential(
            self.system, self.pulses, self.pwi_container, t
        )

    def _compute_vector_potential(self, t):
        if self.inputs("gauge") == "length":
            return 0
        if self.inputs("laser_approx") == "dipole":
            return self._compute_dipole_vector_potential(t)
        else:
            return self._compute_plane_wave_vector_potential(t)

    def _kinetic_momentum_operator(self, t):
        return self.system.momentum + self._compute_vector_potential(t)

    def _compute_expectation_value(self, t, y, M):
        return compute_expectation_value(self.tdcc, t, y, M)

    def _compute_CI_projectors(self, t, y):
        return compute_CI_projectors(self.tdcc, self.states_container, t, y)

    def _compute_conventional_EOM_projectors(self, t_amp, l_amp):
        return compute_conventional_EOM_projectors(self.states_container, t_amp, l_amp)

    def _compute_two_component_EOM_projectors(self, t_amp, l_amp):
        return compute_two_component_EOM_projectors(self.states_container, t_amp, l_amp)

    def _compute_LR_projectors(self, t_amp, l_amp):
        return compute_LR_projectors(self.states_container, t_amp, l_amp)

    def _compute_F_dipole(self, t, rho_qp):
        return compute_F_dipole(t, rho_qp)

    def _compute_F(self, t, rho_qp):
        return compute_F(t, rho_qp, self.pulses, self.pwi_container)

    def propagate(self):
        compute_CC_projectors = bool(
            self.inputs("sample_EOM_projectors")
            + self.inputs("sample_EOM2_projectors")
            + self.inputs("sample_LR_projectors")
        )

        if compute_CC_projectors and self.states_container.R0 is None:
            self.states_container.R0 = self._compute_R0()

        if self.iter is None:
            self.iter = 0

        i0 = self.iter
        f0 = int(self.num_steps)
        time_points = self.samples["time_points"]

        self.logger.log(self.inputs("print_level"))
        self.logger.log(self.inputs("print_level"), name_ext="init_step", values=[i0])
        self.logger.log(self.inputs("print_level"), name_ext="final_step", values=[f0])
        self.logger.log(
            self.inputs("print_level"), name_ext="iterations", values=[f0 - i0]
        )

        self.ckpt_sys_time = time.time() / 3600

        sys_time0 = time.time()

        disable_tqdm = True if self.inputs("print_level") == 0 else False

        for _ in tqdm.tqdm(time_points[i0:f0], disable=disable_tqdm):
            # if not i%10:
            #    print (f'{i} / {self.num_steps}')
            i = self.iter

            self.samples["time_points"][i] = self.r.t

            # Sample one body operators
            for el in self.one_body_sampling_operators:
                M = self.one_body_sampling_operators[el](self.r.t)
                self.samples[el][i] = self._compute_expectation_value(
                    self.r.t, self.r.y, M
                )

            # ENERGY
            if self.inputs("sample_energy"):
                self.samples["energy"][i] = self.tdcc.compute_energy(self.r.t, self.r.y)

            # CI projectors
            if self.inputs("sample_CI_projectors"):
                self.samples["CI_projectors"][i, :] = self._compute_CI_projectors(
                    self.r.t, self.r.y
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
                    ] = self._compute_conventional_EOM_projectors(t, l)

                # EOM2 PROJECTORS
                if self.inputs("sample_EOM2_projectors"):
                    self.samples["EOM2_projectors"][
                        i, :
                    ] = self._compute_two_component_EOM_projectors(t, l)

                # LR PROJECTORS
                if self.inputs("sample_LR_projectors"):
                    self.samples["LR_projectors"][i, :] = self._compute_LR_projectors(
                        t, l
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
                    self.pwi_container.C, self.pwi_container.C_tilde = self._get_C()
                    self.samples["general_response"][i, :, :, :] = compute_F(
                        self.r.t, rho_qp, self.pulses, self.pwi_container
                    )

            if self.inputs("sample_laser_pulse"):
                self.samples["laser_pulse"][i, :] = self.pulses.pulses(self.r.t)

            self.r.integrate(self.r.t + self.inputs("time_step"))
            if not self.r.successful():
                break

            self.iter += 1

            if self.inputs("checkpoint"):
                self.checkpoint()

        sys_time1 = time.time()

        self.logger.log(
            self.inputs("print_level"),
            name_ext="finished",
            values=[self.iter, self.r.successful()],
        )

        self.logger.log(
            self.inputs("print_level"),
            name_ext="run_time",
            values=[sys_time1 - sys_time0],
        )

        return self.get_output()
