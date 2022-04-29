"""
### inputs ##:
'method':               coupled cluster calculation method. Available methods:
                        "rcc2","rccsd","romp2","roaccd","ccs","cc2","ccsd","omp2","oaccd"
'gauge':                gauge choice. Available gauge choices: "length","velocity"
'laser_approx':         laser approximation. "dipole","plane_wave"
'molecule':             name of molecule input file (.xyz file).
'n_electrons':          number of electrons
'custom_basis':         use custom basis input files (uses BSE if False)
'basis':                basis set name
'save_name':            name of files to store outputs (.npz file)
'save_dir':             directory to store output file
'n_excited_states':     number of excited states
'dt':                   time step (au)
'initial_time':         time at start of simulation (au)
'initial_state':        file to load initial state from (will be calculated if None)
'quadratic_terms':      include A^2 term (only relevant to velocity gauge)
'cross_terms':          include Am \cdot An terms (only relevant to velocity gauge)
'stop_run_at':          fraction of total simulation time to be run
'load_run':             file to laod unfinished simulation from (will start from beginning is None)
'integrator':           name of integrator
'integrator_params':    integrator parameters

### laser_inputs ##:
'field_strength':       electric field strength (amplitude)
'omega':                carrier wave frequency
'k_direction':          propagation direction (not relevant to dipole approximation)
'polarization':         electric field polarization vector
'time_after_pulse':     simulation time (au) after each pulse
'ncycles':              number of carrier wave cycles (sine square-type pulses)
'phase':                plane wave phase
'sigma':                standard deviation of gaussian-type pulses

### laser_classes ##:
'dipole_length':        electric field, dipole approximation
'dipole_velocity':      vector potential, dipole approximation
'plane_wave':           time-dependent function, of an enveloped
                        plane wave at the origin, Am(0,t)

### sample_settings ##:
'load_all_vectors':     bool,   if True:  all L1,L2,R1,R2,M1,M2 arrays are loaded into memory at the same time
                                if False: only arrays for one excitation level is loaded at a time to save memory
'final_state':          bool,   save state vector at final time step
'laser_pulse':          bool,   save laser pulse (only dipole approximation)
'energy':               bool,   sample energy
'dipole_moment':        bool,   sample dipole moment
'quadrupole_moment':    bool,   sample quadrupole moments
'momentum':             bool,   sample canonical momentum
'kinetic_momentum':     bool,   sample kinetic momentum
'auto_correlation':     bool,   sample auto correlation function (TDCCSD)
'EOM_projectors':       bool,   sample EOM projectors (TDCCSD)
'EOM2_projectors':      bool,   sample two-component EOM projectors (TDCCSD)
'LR_projectors':        bool,   sample linear response projectors (TDCCSD)
'dipole_response':      bool,   sample dipole response function
'general_response':     bool,   sample general response function
"""

inputs = {
    'method': 'rhf',
    'gauge': 'velocity',
    'laser_approx': 'plane_wave',
    'molecule': 'he',
    'n_electrons': 2,
    'custom_basis': False,
    'basis': 'aug-cc-pvdz',
    'save_name': "tdrhf_rhf2_vpi_newop",
    'save_dir': 'test/',
    'n_excited_states': 3,
    'dt': 5e-2,
    'initial_time': 0.0,
    'initial_state': None,
    'quadratic_terms': True,
    'cross_terms': True,
    'stop_run_at':1.0,
    'load_run': None,
    'integrator': 'GaussIntegrator',
    'integrator_params': {'s':3,'eps':1e-10},
}

laser_inputs = {
    'field_strength': 0.001,
    'omega': 1.0,
    'k_direction': [1.0,0.0,0.0],
    'polarization': [0.0,1j/1.4142135623730951,1/1.4142135623730951],
    'time_after_pulse': 100,
    'ncycles': 2.0,
    'phase':0,
    'sigma':0,
}

laser_classes = {
    'pulse':'square_velocity_dipole',
}

sample_settings = {
    'laser_pulse': True,
    'energy': True,
    'dipole_moment': True,
    'quadrupole_moment':True,
    'momentum': True,
    'kinetic_momentum': True,
    'auto_correlation': False,
    'EOM_projectors': False,
    'EOM2_projectors': False,
    'LR_projectors': False,
    'load_all_vectors': True,
    'final_state': True,
    'dipole_response': True,
    'general_response': True,
}