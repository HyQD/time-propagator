import numpy as np
import importlib


################################################################################
class Inputs:
    def __init__(self, a={}):
        """a: dict or list containing dicts"""
        if type(a) is dict:
            self.setup_dict(a)
        elif type(a) is list:
            self.setup_list(a)

    def format(self, dict_):
        if "pulses" in dict_.keys():
            for el in dict_["pulses"]:
                if el in dict_:
                    if "polarization" in dict_[el]:
                        dict_[el]["polarization"] = np.array(dict_[el]["polarization"])
                    if "k_direction" in dict_[el]:
                        dict_[el]["k_direction"] = np.array(dict_[el]["k_direction"])
        return dict_

    def setup_dict(self, a):
        self.inputs = self.format(a)

    def setup_list(self, a):
        dict_ = a[0]
        for i in np.arange(1, len(a)):
            dict_ = {**dict_, **a[i]}

        self.inputs = self.format(dict_)

    def set_from_file(self, file_name):
        input_module = importlib.import_module(file_name.replace(".py", ""))
        input_attr_names = [el for el in dir(input_module) if not el.startswith("__")]
        for el in input_attr_names:
            input_dict = getattr(input_module, el)
            if type(input_dict) == dict:
                self.set_from_dict(input_dict)

        return self

    def set_from_dict(self, input_dict):
        for key in input_dict:
            self.set(key, input_dict[key])

        return self

    def set(self, key, value):
        self.inputs[key] = value
        self.inputs = self.format(self.inputs)

    def has_key(self, key):
        return key in self.inputs.keys()

    def check_consistency(self):
        """method to check input consistency"""
        # check that only harmonic pulses have complex polarization vectors

        if self.has_key("pulses"):
            for el in self.inputs["pulses"]:

                # check that all pulses have input parameters
                s = "pulse must have defined inputs"
                assert self.has_key(el), s

                pulse_inputs = self(el)

                # check that pulses have defined polarization
                s = "pulses must have defined polarization"
                assert "polarization" in pulse_inputs.keys(), s

                # check that only pulses with defined frequency has complex polarization
                uI = (pulse_inputs["polarization"]).imag
                if np.max(np.abs(uI)) > 1e-14:
                    s = "only harmonic pulses can have complex polarization vectors"
                    assert "omega" in self.inputs[el], s

                if self("laser_approx") == "plane_wave":
                    # check that pulses have defined k_direction if plane waves
                    s = "plane waves need a propagation direction (k_direction)"
                    assert "k_direction" in pulse_inputs.keys(), s

                    # check that the k vector is orthogonal to the polarization vector
                    u = pulse_inputs["polarization"]
                    k = pulse_inputs["k_direction"]
                    s = "k_direction must be orthogonal to the polarization"
                    assert np.abs(np.dot(u, k)) < 1e-14, s

                    # check that gauge is 'length' if plane waves
                    s = "plane wave interaction is only implemented in velocity gauge"
                    assert self("gauge") == "velocity", s

    def __call__(self, key):
        """key: str"""
        return self.inputs[key]


def load_inputs(inputs):
    if isinstance(inputs, str):
        if inputs[-4:] == ".py":
            inputs = load_inputs_from_py(inputs)
        elif inputs[-4:] == ".npz":
            inputs = np.load(inputs, allow_pickle=True)
        else:
            try:
                with open(inputs, "rb") as f:
                    inputs = pickle.load(inputs)
            except pickle.UnpicklingError:
                pass

    return cleanup_inputs(inputs)


def load_inputs_from_py(file_name):
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
    type_check = isinstance(inputs, dict)
    init_from_output = all(
        el in inputs.keys() for el in ["samples", "inputs", "arrays", "log", "misc"]
    )
    return (type_check, init_from_output)


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
