import time
import copy
import operator
import importlib
import warnings
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
from rk4_integrator import Rk4Integrator
import tqdm
import numpy as np

from scipy.integrate import complex_ode

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
from time_propagator0.compute_properties import compute_R0
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


logger = Logger(log_messages, style)
lookup = LookupTables()


class TimePropagator:
    """Class for setting up molecular quantum systems and propagate in time

    Variables:
        system : QuantumSystem object
            QuantumSystem object
        system_values : QuantumSystemValues object
            contains results from AO inegral calculation and/or
            HF calculation such as r, p, s, C matrices and
            HF energy
        CC : time-independent method class
            class for computing time-independent properties
        TDCC : time-dependent method class
            class for propagating a quantum system in time
        cc : instance of CC
        tdcc : instance of TDCC
        num_steps : int
            number of simulation time steps
        ckpt_sys_time : int
            system time in hours, used for checkpointing
        pulses : Pulses object
            object containing laser objects and polarization vectors
        pwi_container : PWIContainer objectr
            object for storing and accessing plane wave integrals
        states_container : StationaryStatesContainer object
            object for storing and accessing stationary states vectors
        state_energies : numpy.ndarray
            stationary state energies
        y0 : numpy.ndarray
            initial state vector
        r : scipy.integrate._ode.complex_ode object
            interface used for time integration
        iter : int
            current time iteration number
        sampling_operators : dict
            collection of callables that returns sampled values
        samples : dict
            dict containing arrays with samples values
        inputs : Inputs object
            handling inputs
        init_from_output : bool
            whether or not input is an output from previous run
    """

    def __init__(self, method=None, inputs=None, **kwargs):
        """
        Keyword arguments:
        method : str
            computational method, f.ex. 'rhf', 'ccsd', 'rccsd', ...
        inputs : dict, str, list
            either dict with input arguments or str with input file name
            (.py, .npz, .pickle). Accepts list of multiple dict/str.
        kwargs:
            see input_template.py for a comprehensive overview of
            allowed keyword arguments
        """

        self.system = None
        self.system_values = QuantumSystemValues()

        self.CC = None
        self.TDCC = None
        self.cc = None
        self.tdcc = None

        self.pulses = None
        self.pwi_container = None
        self.states_container = None
        self.state_energies = None

        self.y0 = None
        self.r = None

        self.iter = None
        self.num_steps = None
        self.ckpt_sys_time = None

        self.samples = {}
        self.sampling_operators = {}

        inputs = copy.deepcopy(inputs)

        self.inputs = Inputs().set_from_file("time_propagator0.default_inputs")

        inputs = load_inputs(inputs)
        valid_type, self.init_from_output, set_from_dict = inspect_inputs(inputs)

        if not valid_type:
            raise TypeError("Could not convert inputs to internal dict format.")

        if self.init_from_output:
            self._init_from_output(inputs)
        elif set_from_dict:
            self.inputs.set_from_dict(inputs)

        self.inputs.set_from_dict(kwargs)

        if method is not None:
            self.inputs.set("method", method)
        elif not self.inputs.has_key("method"):
            raise TypeError("Parameter 'method' must be defined at initiation.")

        self._init_method()

        if not self.init_from_output:
            logger.log(
                self.inputs("print_level"), values=[self.inputs("method")], new_lines=0
            )

    def _init_method(self):
        method = self.inputs("method")

        if not method in lookup.methods:
            raise ValueError(f"Computational method {method} is not supported.")

        self.restricted = lookup.methods[method]["restricted"]
        self.correlated = lookup.methods[method]["correlated"]
        self.orbital_adaptive = lookup.methods[method]["orbital_adaptive"]

        module = lookup.methods[method]["module"]
        cc = lookup.methods[method]["cc"]
        tdcc = lookup.methods[method]["tdcc"]

        self.CC = getattr(importlib.import_module(module), cc)
        self.TDCC = getattr(importlib.import_module(module), tdcc)

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

        if "hf_coeff_matrix" in arrays:
            self.set_hf_coeff_matrix(arrays["hf_coeff_matrix"])

        self.iter = misc["iter"]

        logger.set_log(log)
        logger.log(self.inputs("print_level"))

    def set_input(self, key, value):
        self.inputs.set(key, value)

        logger.log(self.inputs("print_level"))

    def add_pulse(self, name, pulse_inputs):
        (self.inputs.inputs["pulses"]).append(name)
        self.inputs.set(name, pulse_inputs)

        logger.log(self.inputs("print_level"), values=[name])

    def setup_quantum_system(
        self, reference_program=None, force_hf_recalculation=False
    ):
        """reference_program: 'pyscf' ,'dalton',"""
        if reference_program is not None:
            self.inputs.set("reference_program", reference_program)

        program = self.inputs("reference_program")

        if not program in lookup.reference_programs:
            raise TypeError(f"Quantum system setup is not implemented with {program}.")

        molecule = self.inputs("molecule")
        basis = self.inputs("basis")
        charge = self.inputs("charge")

        skip_hf = (self.system_values.C is not None) and (not force_hf_recalculation)

        if program == "pyscf":
            if skip_hf:
                from time_propagator0 import run_pyscf_ao

                self.system_values.set_pyscf_values_ao(
                    run_pyscf_ao(molecule, basis=basis, charge=charge)
                )
            else:
                from time_propagator0 import run_pyscf_rhf

                self.system_values.set_pyscf_values_rhf(
                    run_pyscf_rhf(molecule, basis=basis, charge=charge)
                )
        elif program == "dalton":
            if skip_hf:
                warnings.warn(
                    "Reusing hf_coeff_matrix does not currently work with Dalton."
                )

            from time_propagator0 import run_dalton_rhf

            self.system_values.set_dalton_values_rhf(
                run_dalton_rhf(molecule, basis=basis, charge=charge)
            )

        if self.restricted:
            system = construct_quantum_system(self.system_values)
        else:
            system = construct_quantum_system(
                self.system_values, add_spin=True, anti_symmetrize=True
            )

        self.set_quantum_system(system, self.system_values.C)

        logger.log(self.inputs("print_level"), values=[program])
        logger.log(
            self.inputs("print_level"),
            tag="hf_energy",
            values=[self.system_values.hf_energy],
        )

    def set_quantum_system(self, system, C=None):
        """system: QuantumSystem
        C: HF coefficient matrix"""
        self.system = system
        if C is not None:
            self.set_hf_coeff_matrix(C)

        logger.log(self.inputs("print_level"))

    def set_hf_coeff_matrix(self, C):
        self.system_values.C = C

    def setup_projectors(self, EOMCC_program=None):
        """EOMCC_program: str"""
        logger.log(self.inputs("print_level"), values=[self.inputs("method")])

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

            if not program in lookup.EOMCC_programs:
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

                logger.log(
                    self.inputs("print_level"),
                    tag="EOMCC_program",
                    values=[self.inputs("EOMCC_program")],
                )

                self.set_projectors(da=da)

        logger.log(
            self.inputs("print_level"),
            tag="energies",
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
        if L1 is not None:
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

        logger.log(self.inputs("print_level"))

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
            y0 = np.identity(len(self.system_values.C)).ravel()

        logger.log(self.inputs("print_level"))

        self.set_initial_state(y0)

    def setup_initial_state(self, state_number=0):
        """state_number : int, list, numpy.ndarray"""
        logger.log(self.inputs("print_level"), values=[state_number])

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

        logger.log(self.inputs("print_level"))

    def setup_plane_wave_integrals(self, PWI_program=None):
        if PWI_program is not None:
            self.inputs.set("PWI_program", PWI_program)

        program = self.inputs("PWI_program")

        if not program in lookup.PWI_programs:
            raise TypeError(
                f"Plane wave integral setup is not implemented with {program}."
            )

        if program == "molcas":
            pulse_inputs = [self.inputs(el) for el in self.inputs("pulses")]
            integrals = setup_plane_wave_integrals_from_molcas(
                pulse_inputs,
                self.inputs("molecule"),
                get_basis(self.inputs("basis"), "molcas"),
                quadratic_terms=self.inputs("quadratic_terms"),
                cross_terms=self.inputs("cross_terms"),
                compute_A=self.inputs("sample_kinetic_momentum"),
                custom_basis=self.inputs("custom_basis"),
            )
        self.set_plane_wave_integrals(integrals)

        logger.log(self.inputs("print_level"))

    def set_plane_wave_integrals(self, integrals, change_basis=True):
        """
        integrals : dict containing integral arrays

        Convention for the keys (m,n ∈ 0,1,2... are pulse numbers, according
        to the order given in the input 'pulses' list):
        cos(k_m.r): 'cos,m'
        sin(k_m.r): 'sin,m'
        cos(k_m.r).p: 'cosp,m'
        sin(k_m.r).p: 'sinp,m'
        cos([k_m+k_n].r): 'cos+,mn'
        sin([k_m+k_n].r): 'sin+,mn'
        cos([k_m-k_n].r): 'cos-,mn'
        sin([k_m-k_n].r): 'sin-,mn'

        These integrals are only required if 'laser_approx':'plane_wave'.

        cos(k_m.r) and sin(k_m.r) are only required if
        'sampling_kinetic_momentum':True.

        Two-index terms:
            m=n terms are required if 'quadratic_terms':True
            n!=m terms are required if 'cross_terms':True
        """
        if self.orbital_adaptive:
            self.pwi_container = IntegralContainerOrbitalAdaptive(
                integrals, C=self.system_values.C, C_tilde=self.system_values.C.conj().T
            )
        else:
            self.pwi_container = IntegralContainerFixedOrbitals(
                integrals, C=self.system_values.C, C_tilde=self.system_values.C.conj().T
            )
        if change_basis:
            self.pwi_container.change_basis()

        logger.log(self.inputs("print_level"))

    def setup_pulses(self):
        pulse_inputs = []
        for el in self.inputs("pulses"):
            pulse_inputs.append(self.inputs(el))
        pulses = setup_Pulses(pulse_inputs)
        self.set_pulses(pulses)

        logger.log(self.inputs("print_level"))

    def set_pulses(self, pulses):
        """Pulses class"""
        self.pulses = pulses

        logger.log(self.inputs("print_level"))

    def build(self, integrator=None, **integrator_params):
        logger.log(self.inputs("print_level"))

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

        self._build_hamiltonian()
        self._build_sampling_arrays()
        self._build_default_sampling_operators()
        self._build_integrator(integrator, **integrator_params)

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

    def add_custom_sampling_operator(self, name, op, dim, dtype="complex128"):
        """op is a callable that returns sample value"""
        self.inputs.set_custom_input(f"sample_{name}", True)
        self.sampling_operators[name] = op
        dim_ = [self.num_steps] + list(dim)
        self.samples[name] = np.zeros(dim_, dtype=dtype)

    def _build_default_sampling_operators(self):
        sample_props = lookup.sample_properties
        for name, props in zip(sample_props, sample_props.values()):
            if self.inputs(props["sample_keyword"]):
                op = getattr(
                    importlib.import_module("time_propagator0.sampling_operators"),
                    props["sampling_operator"],
                )
                self.sampling_operators[name] = op

    def _build_hamiltonian(self):
        logger.log(self.inputs("print_level"))

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

    def _build_integrator(
        self, integrator=None, integrator_module=None, **integrator_params
    ):
        logger.log(self.inputs("print_level"))

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
            t0 = (
                np.max(self.samples["time_points"])
                + self.inputs("initial_time")
                + self.inputs("time_step")
            )
        else:
            t0 = self.inputs("initial_time")

        if integrator_module is not None:
            name = integrator
            module = integrator_module
        else:
            name = lookup.integrators[integrator]["name"]
            module = lookup.integrators[integrator]["module"]

        if module is not None:
            getattr(importlib.import_module(module), name)

        self.tdcc = self.TDCC(self.system)

        self.r = complex_ode(self.tdcc).set_integrator(name, **integrator_params)
        self.r.set_initial_value(self.y0, t0)

    def _build_sampling_arrays(self, samples=None):
        logger.log(self.inputs("print_level"))

        total_time = self.inputs("final_time") - self.inputs("initial_time")
        self.num_steps = int(total_time / self.inputs("time_step")) + 1
        if not self.init_from_output:
            self.setup_samples()
        else:
            self.update_samples()

    def setup_samples(self):
        samples = {}
        samples["time_points"] = np.zeros(self.num_steps)

        sample_prop = lookup.sample_properties
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

        logger.log(self.inputs("print_level"), values=[self.iter])

    def get_output(self):
        output = {
            "samples": self.samples,
            "inputs": self.inputs.inputs,
            "log": logger._log,
            "arrays": {},
            "misc": {},
        }

        if self.inputs("return_state"):
            output["arrays"]["state"] = self.r.y
        if self.inputs("return_hf_coeff_matrix"):
            output["arrays"]["hf_coeff_matrix"] = self.system_values.C
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
        """returns sampling array dimensionality in run time"""
        dim_ = list(dim)
        for i in [j for j in range(len(dim)) if isinstance(dim_[j], str)]:
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

        logger.log(self.inputs("print_level"))
        logger.log(self.inputs("print_level"), tag="init_step", values=[i0])
        logger.log(self.inputs("print_level"), tag="final_step", values=[f0])
        logger.log(self.inputs("print_level"), tag="iterations", values=[f0 - i0])

        self.ckpt_sys_time = time.time() / 3600

        sys_time0 = time.time()

        disable_tqdm = True if self.inputs("print_level") == 0 else False

        for _ in tqdm.tqdm(time_points[i0:f0], disable=disable_tqdm):
            i = self.iter

            # if not i%10:
            #    print (f'{i} / {self.num_steps}')

            self.samples["time_points"][i] = self.r.t

            # Sample
            for el in self.sampling_operators:
                self.samples[el][i] = self.sampling_operators[el](self)

            self.r.integrate(self.r.t + self.inputs("time_step"))
            if not self.r.successful():
                warnings.warn("Time integration step did not converge.")
                break

            self.iter += 1

            if self.inputs("checkpoint"):
                self.checkpoint()

        sys_time1 = time.time()

        logger.log(
            self.inputs("print_level"),
            tag="finished",
            values=[self.iter, self.r.successful()],
        )

        logger.log(
            self.inputs("print_level"),
            tag="run_time",
            values=[sys_time1 - sys_time0],
        )

        return self.get_output()
