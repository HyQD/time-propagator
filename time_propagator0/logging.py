import inspect

style = {
    "empty": "{0}",
    "a": "# # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n# {0}\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # #",
    "b": " - {0}",
    "c": " ¤ {0}",
    "d": "# {0}",
    "e": " # {0}",
}

log_messages = {
    "__init__": {
        "string": "TimePropagator instance initiated with {0} method.",
        "style": "a",
        "print_level": 6,
    },
    "_init_from_output": {
        "string": "TimePropagator instance initiated from an output.",
        "style": "a",
        "print_level": 6,
    },
    "set_quantum_system": {
        "string": "Setting quantum system",
        "style": "b",
        "print_level": 6,
    },
    "set_projectors": {
        "string": "Setting projectors",
        "style": "b",
        "print_level": 6,
    },
    "set_initial_state": {
        "string": "Setting initial state",
        "style": "b",
        "print_level": 6,
    },
    "set_plane_wave_integrals": {
        "string": "Setting plane wave integrals",
        "style": "b",
        "print_level": 6,
    },
    "set_pulses": {
        "string": "Setting pulses",
        "style": "b",
        "print_level": 6,
    },
    "add_pulse": {
        "string": "Added pulse {0}",
        "style": "c",
        "print_level": 5,
    },
    "setup_quantum_system": {
        "string": "Setting up quantum system using {0}",
        "style": "c",
        "print_level": 5,
    },
    "setup_quantum_system hf_energy": {
        "string": "Hartree-Fock energy: {0}",
        "style": "c",
        "print_level": 2,
    },
    "setup_projectors": {
        "string": "Setting up {0} projectors",
        "style": "c",
        "print_level": 5,
    },
    "setup_projectors energies": {
        "string": "Stationary state energies: {0}",
        "style": "c",
        "print_level": 3,
    },
    "setup_projectors EOMCC_program": {
        "string": "Computing EOM-CC excited states using {0}",
        "style": "empty",
        "print_level": 5,
    },
    "setup_ground_state": {
        "string": "Setting up ground state",
        "style": "c",
        "print_level": 5,
    },
    "setup_initial_state": {
        "string": "Setting up initial state. State number {0}",
        "style": "c",
        "print_level": 5,
    },
    "setup_plane_wave_integrals": {
        "string": "Setting up plane wave integrals",
        "style": "c",
        "print_level": 5,
    },
    "setup_pulses": {
        "string": "Setting up pulses",
        "style": "b",
        "print_level": 5,
    },
    "build": {
        "string": "Building TimePropagator ...",
        "style": "d",
        "print_level": 5,
    },
    "_build_hamiltonian": {
        "string": "Building Hamiltonian",
        "style": "e",
        "print_level": 5,
    },
    "_build_integrator": {
        "string": "Building integrator",
        "style": "e",
        "print_level": 5,
    },
    "_build_sampling_arrays": {
        "string": "Building sample arrays",
        "style": "e",
        "print_level": 5,
    },
    "propagate": {
        "string": "Starting time propagation ...",
        "style": "empty",
        "print_level": 5,
    },
    "propagate init_step": {
        "string": "Initial time step: {0}",
        "style": "empty",
        "print_level": 5,
    },
    "propagate final_step": {
        "string": "Final time step: {0}",
        "style": "empty",
        "print_level": 5,
    },
    "propagate iterations": {
        "string": "Number of iterations: {0}",
        "style": "empty",
        "print_level": 5,
    },
    "propagate finished": {
        "string": "Time propagation finished at iteration number {0}. Success: {1}",
        "style": "empty",
        "print_level": 5,
    },
    "propagate run_time": {
        "string": "Run time: {0} seconds",
        "style": "empty",
        "print_level": 5,
    },
    "checkpoint": {
        "string": "Checkpoint at iteration number {0}",
        "style": "empty",
        "print_level": 3,
    },
}


class Logger:
    def __init__(self, log_messages, style, log=None):
        self._log_messages = log_messages
        self._style = style

        self._log = "" if log is None else log

    def set_log(self, log):
        self._log = log

    def log(self, print_level, name_ext="", values=[], new_lines=1):
        method_name = inspect.currentframe().f_back.f_code.co_name

        method_name += " " + name_ext if len(name_ext) > 0 else ""
        log_message = self._log_messages[method_name]

        string = log_message["string"].format(*values)
        style = log_message["style"]
        print_level_ = log_message["print_level"]

        vert_spaces = "\n" * new_lines
        s = self._style[style].format(string)
        self._log += vert_spaces + s

        if print_level >= print_level_:
            print(s)
