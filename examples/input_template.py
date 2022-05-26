# example input with all possible input parameters (some exceptions since some inputs
# depends on external modules). Inputs from file may be given in one or more dicts of
# arbitrary name. All dicts are read at initiation.

pulses = {
    "pulses": [
        "pulse1",
        "pulse2",
    ],  # list with pulse names. Each name must be introduced
    # as a input parameter. Name cannot be another
    # valid input parameter name.
    # examples on how to introduce pulses. The details depends on the particular pulse class,
    # but all pulses must have a polarization vector (may be complex), and all pulses must
    # have a k_direction if 'laser_approx':'plane_wave'
    "pulse1": {
        "pulse_class": "square_velocity_dipole",
        "field_strength": 0.01,
        "omega": 1.0,
        "polarization": [0.0, 0.0, 1.0],
        "k_direction": [1.0, 0.0, 0.0],
        "ncycles": 2,
    },
    "pulse2": {
        "pulse_class": "square_velocity_dipole",
        "field_strength": 0.01,
        "omega": 2.0,
        "polarization": [0.0, 0.0, 1.0],
        "k_direction": [1.0, 0.0, 0.0],
        "ncycles": 2,
        "t0": 10,
    },
}


inputs = {
    "molecule": "he.xyz",  # molecule geometry. str or .xyz-file. Units are always bohr
    # str format example: 'Li 0.0 0.0 0.0; H 0.0 0.0 -3.0139491027559635'
    "basis": "cc-pvdz",  # name of basis set
    # Either str or dict
    # If "custom_basis":False : BSE module is used )
    # Else: Format is {'pyscf':{'aug-cc-pvdz'}, 'dalton':{h:'aug-cc-pvdz'},li:'cc-pvdz'}} etc.
    "custom_basis": False,  # use custom basis set files (not yet functional)
    "charge": 0,  # total charge of the molecule
    "gauge": "length",  #'length' or 'velocity'
    "laser_approx": "dipole",  #'dipole' or 'plane_wave'
    "time_step": 1e-1,  # time step
    "initial_time": 0.0,  # initial time of simulation
    "final_time": 10.0,  # final time of simulation (last time step)
    "integrator": "GaussIntegrator",  # 'GaussIntegrator' or 'vode'
    "integrator_params": {
        "s": 3,  # parameters for the integrator. Details depends
        "eps": 1e-10,
    },  # dict, keys depends on the choice of integrator
    "quadratic_terms": True,  # include Am.Am terms in the Hamiltonian. Only
    # relevant for velocity gauge
    "cross_terms": True,  # include Am.An (m!=n) terms in Hamiltonian. Only
    # relevant for simulations in velocity gauge with
    # more than one pulse
    "n_excited_states": 10,  # number of excited CC or EOM-CC states. Only
    # relevant if setup_projectors() is called
    "print_level": 1,  # printouts
    "checkpoint": 0,
    "checkpoint_unit": 'iterations',   #'iterations' or 'hours'
    "reference_program":'pyscf',    #   'pyscf' or 'dalton'
    "EOMCC_program":'dalton',   # 'dalton'
    "PWI_program":'molcas', # 'molcas'
    "sample_laser_pulse": False,  # sample the laser pulse (will sample the
    # pulse at origin if 'laser_approx':'plane_wave')
    "sample_energy": False,  # sample the expectation value of the
    # Hamiltonian (may be misleading name?)
    "sample_dipole_moment": True,  # sample dipole moment
    "sample_quadrupole_moment": False,  # sample quadrupole moment (not yet functional)
    "sample_momentum": False,  # sample canonical momentum
    "sample_kinetic_momentum": False,  # sample kinetic momentum
    "sample_CI_projectors": False,  # sample CI projectors (only for unrestricted CI)
    "sample_auto_correlation": False,  # sample correlation function (only relevant
    # for (r)cc2 and (r)ccsd)
    "sample_EOM_projectors": False,  # sample conventional EOM-CC projectors (only for
    # rcc2 and rccsd)
    "sample_EOM2_projectors": False,  # sample two-component EOM-CC projectors (only for
    # rcc2 and rccsd)
    "sample_LR_projectors": False,  # sample linear response projectors (only for
    # rcc2 and rccsd)
    "load_all_response_vectors": True,  # if False: only response vectors for one excitation
    # level is read to memory at a time to save memory. Only works if response vectors are
    # provided from a daltonproject.dalton.arrays.Arrays object.
    # Only relevant for rcc2 and rccsd
    "return_final_state": True,  # include the final state vector in the return dict
    "sample_dipole_response": False,  # sample F dipole approximation (not functional yet)
    "sample_general_response": False,  # sample F assuming enveloped plane wave interaction
}
