default_inputs = {
    "basis": "cc-pvdz",
    "gauge": "length",
    "laser_approx": "dipole",
    "custom_basis": False,
    "time_step": 1e-1,
    "initial_time": 0.0,
    "final_time": 10.0,
    "initial_state": None,
    "integrator": "GaussIntegrator",
    "ground_state_tolerance": 1e-10,
    "quadratic_terms": True,
    "cross_terms": True,
    "charge": 0,
    "n_excited_states": 10,
    "print_level": 1,
    "return_inputs": True,
    "checkpoint": 0,
    "checkpoint_unit": "iterations",
    "sample_laser_pulse": False,
    "sample_energy": False,
    "sample_dipole_moment": True,
    "sample_quadrupole_moment": False,
    "sample_momentum": False,
    "sample_kinetic_momentum": False,
    "sample_CI_projectors": False,
    "sample_auto_correlation": False,
    "sample_EOM_projectors": False,
    "sample_EOM2_projectors": False,
    "sample_LR_projectors": False,
    "load_all_response_vectors": True,
    "return_state": True,
    "return_C": False,
    "return_stationary_states": False,
    "return_plane_wave_integrals": False,
    "sample_dipole_response": False,
    "sample_general_response": False,
    "reference_program": "pyscf",
    "EOMCC_program": "dalton",
    "PWI_program": "molcas",
}
