# Contains dictionaries and tuples defining implemented methods,
# integrators, property sampling input types and implemented
# interfaces for 3rd party programs

Methods = {
    "rcc2": {
        "module": "coupled_cluster.rcc2",
        "cc": "RCC2",
        "tdcc": "TDRCC2",
        "restricted": True,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "rccsd": {
        "module": "coupled_cluster.rccsd",
        "cc": "RCCSD",
        "tdcc": "TDRCCSD",
        "restricted": True,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "romp2": {
        "module": "optimized_mp2.romp2",
        "cc": "ROMP2",
        "tdcc": "TDROMP2",
        "restricted": True,
        "correlated": True,
        "orbital_adaptive": True,
    },
    "roaccd": {
        "module": "coupled_cluster.rccd",
        "cc": "ROACCD",
        "tdcc": "ROATDCCD",
        "restricted": True,
        "correlated": True,
        "orbital_adaptive": True,
    },
    "rhf": {
        "module": "hartree_fock.rhf",
        "cc": "RHF",
        "tdcc": "TDRHF",
        "restricted": True,
        "correlated": False,
        "orbital_adaptive": True,
    },
    "rcis": {
        "module": "ci_singles",
        "cc": "CIS",
        "tdcc": "TDCIS",
        "restricted": True,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "ccs": {
        "module": "coupled_cluster.ccs",
        "cc": "CCS",
        "tdcc": "TDCCS",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cc2": {
        "module": "coupled_cluster.cc2",
        "cc": "CC2",
        "tdcc": "TDCC2",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "ccsd": {
        "module": "coupled_cluster.ccsd",
        "cc": "CCSD",
        "tdcc": "TDCCSD",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "omp2": {
        "module": "optimized_mp2.omp2",
        "cc": "OMP2",
        "tdcc": "TDOMP2",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": True,
    },
    "oaccd": {
        "module": "coupled_cluster.oaccd",
        "cc": "OACCD",
        "tdcc": "OATDCCD",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": True,
    },
    "cis": {
        "module": "configuration_interaction",
        "cc": "CIS",
        "tdcc": "TDCIS",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cid": {
        "module": "configuration_interaction",
        "cc": "CID",
        "tdcc": "TDCID",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cisd": {
        "module": "configuration_interaction",
        "cc": "CISD",
        "tdcc": "TDCISD",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cidt": {
        "module": "configuration_interaction",
        "cc": "CIDT",
        "tdcc": "TDCIDT",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cisdt": {
        "module": "configuration_interaction",
        "cc": "CISDT",
        "tdcc": "TDCISDT",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cidtq": {
        "module": "configuration_interaction",
        "cc": "CIDTQ",
        "tdcc": "TDCIDTQ",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
    "cisdtq": {
        "module": "configuration_interaction",
        "cc": "CISDTQ",
        "tdcc": "TDCISDTQ",
        "restricted": False,
        "correlated": True,
        "orbital_adaptive": False,
    },
}

Integrators = {
    "vode": {
        "name": "vode",
        "module": None,
    },
    "GaussIntegrator": {
        "name": "GaussIntegrator",
        "module": "gauss_integrator",
    },
    "Rk4Integrator": {
        "name": "Rk4Integrator",
        "module": "rk4-integrator",
    },
}

SampleProperties = {
    "laser_pulse": {
        "dim": (3,),
        "dtype": "float",
        "sample_keyword": "sample_laser_pulse",
        "operator_attr": "_pulse_value",
        "sampling_operator": "laser_pulse",
    },
    "energy": {
        "dim": (1,),
        "dtype": "complex128",
        "sample_keyword": "sample_energy",
        "operator_attr": "_hamiltonian_expectation_value",
        "sampling_operator": "expectation_value_hamiltonian",
    },
    "dipole_moment": {
        "dim": (3,),
        "dtype": "complex128",
        "sample_keyword": "sample_dipole_moment",
        "operator_attr": "_dipole_moment_expectation_value",
        "sampling_operator": "expectation_value_dipole_moment",
    },
    "quadrupole_moment": {
        "dim": (3,),
        "dtype": "complex128",
        "sample_keyword": "sample_quadrupole_moment",
        "operator_attr": "_quadrupole_moment_expectation_value",
        "sampling_operator": "expectation_value_quadrupole_moment",
    },
    "momentum": {
        "dim": (3,),
        "dtype": "complex128",
        "sample_keyword": "sample_momentum",
        "operator_attr": "_momentum_expectation_value",
        "sampling_operator": "expectation_value_momentum",
    },
    "kinetic_momentum": {
        "dim": (3,),
        "dtype": "complex128",
        "sample_keyword": "sample_kinetic_momentum",
        "operator_attr": "_kinetic_momentum_expectation_value",
        "sampling_operator": "expectation_value_kinetic_momentum",
    },
    "CI_projectors": {
        "dim": ("states_container.n_states",),
        "dtype": "complex128",
        "sample_keyword": "sample_CI_projectors",
        "operator_attr": "_compute_CI_projectors",
        "sampling_operator": "CI_projectors",
    },
    "auto_correlation": {
        "dim": (1,),
        "dtype": "complex128",
        "sample_keyword": "sample_auto_correlation",
        "operator_attr": "_compute_auto_correlation",
        "sampling_operator": "auto_correlation",
    },
    "EOM_projectors": {
        "dim": ("states_container.n_states",),
        "dtype": "complex128",
        "sample_keyword": "sample_EOM_projectors",
        "operator_attr": "_compute_conventional_EOM_projectors",
        "sampling_operator": "conventional_EOM_projectors",
    },
    "EOM2_projectors": {
        "dim": ("states_container.n_states",),
        "dtype": "complex128",
        "sample_keyword": "sample_EOM2_projectors",
        "operator_attr": "_compute_two_component_EOM_projectors",
        "sampling_operator": "two_component_EOM_projectors",
    },
    "LR_projectors": {
        "dim": ("states_container.n_states",),
        "dtype": "complex128",
        "sample_keyword": "sample_LR_projectors",
        "operator_attr": "_compute_LR_projectors",
        "sampling_operator": "LR_projectors",
    },
    "dipole_response": {
        "dim": (
            2,
            2,
            "pulses.n_pulses",
        ),
        "dtype": "complex128",
        "sample_keyword": "sample_dipole_response",
        "operator_attr": "_compute_F_dipole",
        "sampling_operator": "compute_F_dipole",
    },
    "general_response": {
        "dim": (
            2,
            2,
            "pulses.n_pulses",
        ),
        "dtype": "complex128",
        "sample_keyword": "sample_general_response",
        "operator_attr": "_compute_F",
        "sampling_operator": "compute_F",
    },
}

InputRequirements = {
    "method": {
        "dtypes": ("str",),
    },
    "basis": {
        "dtypes": (
            "str",
            "dict",
        )
    },
    "molecule": {
        "dtypes": ("str",),
    },
    "gauge": {
        "dtypes": ("str",),
    },
    "laser_approx": {
        "dtypes": ("str",),
    },
    "custom_basis": {
        "dtypes": ("bool",),
    },
    "time_step": {
        "dtypes": (
            "int",
            "float",
        )
    },
    "initial_time": {
        "dtypes": (
            "int",
            "float",
        )
    },
    "final_time": {
        "dtypes": (
            "int",
            "float",
        )
    },
    "integrator": {
        "dtypes": ("str",),
    },
    "integrator_params": {
        "dtypes": ("dict",),
    },
    "integrator_module": {
        "dtypes": (
            "NoneType",
            "str",
        )
    },
    "ground_state_tolerance": {
        "dtypes": ("float",),
    },
    "quadratic_terms": {
        "dtypes": ("bool",),
    },
    "cross_terms": {
        "dtypes": ("bool",),
    },
    "charge": {
        "dtypes": ("int",),
    },
    "n_excited_states": {
        "dtypes": ("int",),
    },
    "print_level": {
        "dtypes": ("int",),
    },
    "checkpoint": {
        "dtypes": (
            "int",
            "float",
        )
    },
    "checkpoint_name": {
        "dtypes": (
            "str",
            "NoneType",
        )
    },
    "checkpoint_unit": {
        "dtypes": ("str",),
    },
    "sample_laser_pulse": {
        "dtypes": ("bool",),
    },
    "sample_energy": {
        "dtypes": ("bool",),
    },
    "sample_dipole_moment": {
        "dtypes": ("bool",),
    },
    "sample_quadrupole_moment": {
        "dtypes": ("bool",),
    },
    "sample_momentum": {
        "dtypes": ("bool",),
    },
    "sample_kinetic_momentum": {
        "dtypes": ("bool",),
    },
    "sample_CI_projectors": {
        "dtypes": ("bool",),
    },
    "sample_auto_correlation": {
        "dtypes": ("bool",),
    },
    "sample_EOM_projectors": {
        "dtypes": ("bool",),
    },
    "sample_EOM2_projectors": {
        "dtypes": ("bool",),
    },
    "sample_LR_projectors": {
        "dtypes": ("bool",),
    },
    "sample_dipole_response": {
        "dtypes": ("bool",),
    },
    "sample_general_response": {
        "dtypes": ("bool",),
    },
    "load_all_response_vectors": {
        "dtypes": ("bool",),
    },
    "return_state": {
        "dtypes": ("bool",),
    },
    "return_C": {
        "dtypes": ("bool",),
    },
    "return_stationary_states": {
        "dtypes": ("bool",),
    },
    "return_plane_wave_integrals": {
        "dtypes": ("bool",),
    },
    "reference_program": {
        "dtypes": ("str",),
    },
    "EOMCC_program": {
        "dtypes": ("str",),
    },
    "PWI_program": {
        "dtypes": ("str",),
    },
    "pulses": {
        "dtypes": ("list",),
    },
}


ReferencePrograms = (
    "pyscf",
    "dalton",
)

EOMCCPrograms = ("dalton",)

PWIPrograms = ("molcas",)


class LookupTables:
    def __init__(self):
        self.methods = Methods
        self.integrators = Integrators
        self.sample_properties = SampleProperties
        self.input_requirements = InputRequirements
        self.reference_programs = ReferencePrograms
        self.EOMCC_programs = EOMCCPrograms
        self.PWI_programs = PWIPrograms

    def find(self, attr, key, value=True):
        table = getattr(self, attr)

        if not isinstance(table, dict):
            return None

        ret_table = {}
        for el, val in zip(table, table.values()):
            if val[key] == value:
                ret_table[el] = val
        return ret_table
