import numpy as np
import importlib
import pickle
import warnings


class Inputs:
    def __init__(self, inputs=None, input_requirements=None):
        """inputs: dict, list containing dicts, or "*.py" filename as str"""
        self.inputs = {}
        self.inputs["pulses"] = []

        if isinstance(inputs, dict):
            self.set_from_dict(inputs)
        elif isinstance(inputs, list):
            self.set_from_list(inputs)
        elif isinstance(inputs, str):
            self.set_from_file(inputs)

        self.input_requirements = input_requirements

    def set_from_list(self, list_):
        for el in list_:
            self.set_from_dict(el)

        return self

    def set_from_file(self, file_name):
        input_dict = load_inputs_from_py(file_name)
        self.set_from_dict(input_dict)

        return self

    def set_from_dict(self, input_dict):
        if "pulses" in input_dict.keys():
            self.set("pulses", input_dict["pulses"])
        for key in input_dict:
            self.set(key, input_dict[key])

        return self

    def set(self, key, value):
        if self.input_requirements is not None:
            self.check_validity(key, value)
        self.inputs[key] = self.format(key, value)

        return self

    def set_custom_input(self, key, value):
        if key in self.input_requirements:
            raise ValueError(f"Custom input {key} has same name as a defalt input.")
        self.inputs[key] = value

        return self

    def format(self, key, value):
        if key in self("pulses"):
            return self.format_pulse(value)
        else:
            return self.format_input(key, value)

    def format_input(self, key, input):
        return input

    def format_pulse(self, pulse):
        pulse["polarization"] = np.array(pulse["polarization"])
        if "k_direction" in pulse:
            pulse["k_direction"] = np.array(pulse["k_direction"])
        return pulse

    def check_validity(self, key, value):
        in_pulses = key in self("pulses")
        in_inputs = key in self.input_requirements
        if in_pulses and in_inputs:
            raise ValueError(
                f"You have used pulse name {k}, which is a valid input parameter."
            )
        if (not in_pulses) and (not in_inputs):
            raise ValueError(f"{key} is not a valid input parameter.")
        if in_inputs:
            self.validate_input(key, value)
        elif in_pulses:
            self.validate_pulse(value)

    def validate_pulse(self, pulse):
        if not isinstance(pulse, dict):
            raise TypeError(f"Pulse inputs must be of type dict.")
        if not "pulse_class" in pulse:
            raise KeyError(f"Pulse inputs must have key 'pulse_class'")
        if not "polarization" in pulse:
            raise KeyError(f"Pulse inputs must have key 'polarization'")
        if "k_direction" in pulse:
            eps = 1e-14
            dot_uk = np.abs(
                np.dot(np.array(pulse["polarization"]), np.array(pulse["k_direction"]))
            )
            if dot_uk > eps:
                warnings.warns(
                    f"Large dot product between the polarization vector \
                                and propagation direction: (u,k)={dot_uk}"
                )

    def validate_input(self, key, value):
        valid_types = self.input_requirements[key]["dtypes"]
        if not type(value).__name__ in valid_types:
            raise TypeError(f"{key} has to be one of the types {valid_types}")

    def has_key(self, key):
        return key in self.inputs.keys()

    def __call__(self, key):
        """Return an input value

        key: str
        """
        return self.inputs[key]


def load_inputs(inputs):
    """Return an input dict

    Atempts to load all input dictionaries from dict, file (str) or a list
    containing one or more dicts and file names. In case of list: If common
    key names, elements with higher list index have presedence.
    """
    if isinstance(inputs, str):
        if inputs[-4:] == ".py":
            inputs_ = load_inputs_from_py(inputs)
        elif inputs[-4:] == ".npz":
            inputs_ = np.load(inputs, allow_pickle=True)
        else:
            try:
                with open(inputs, "rb") as f:
                    inputs_ = pickle.load(inputs)
            except pickle.UnpicklingError:
                pass
    elif isinstance(inputs, list):
        inputs_ = {}
        for el in inputs:
            inputs_ = {**inputs_, **load_inputs(el)}
    else:
        inputs_ = inputs

    return cleanup_inputs(inputs_)


def load_inputs_from_py(file_name):
    """Return inputs dict
    file_name : str

    Collects all dicts from a .py file and generates an inputs dict
    """
    inputs = {}
    input_module = importlib.import_module(file_name.replace(".py", ""))
    input_attr_names = [el for el in dir(input_module) if not el.startswith("__")]
    for el in input_attr_names:
        input_dict = getattr(input_module, el)
        if type(input_dict) == dict:
            inputs = {**inputs, **input_dict}

    return inputs


def inspect_inputs(inputs):
    """determine if inputs argument is the output results of a simulation"""
    if inputs is None:
        return (True, False, False)
    if not isinstance(inputs, dict):
        return (False, False, False)
    init_from_output = all(
        el in inputs.keys() for el in ["samples", "inputs", "arrays", "log", "misc"]
    )
    return (True, init_from_output, not init_from_output)


def cleanup_inputs(inputs):
    if isinstance(inputs, np.lib.npyio.NpzFile) or isinstance(inputs, dict):
        inputs = dearrayfy_inputs(inputs)
    return inputs


def dearrayfy_inputs(inputs):
    inputs = dict(inputs)
    elems = ["samples", "inputs", "arrays", "log", "misc"]
    for el in elems:
        if el in inputs.keys() and isinstance(inputs[el], np.ndarray):
            inputs[el] = inputs[el].item()
    return inputs
