import numpy as np

from time_propagator0.field_interaction import lasers

from time_propagator0.setup_daltonproject import (
    compute_plane_wave_integrals_from_molcas,
)

import importlib

from qcelemental import periodictable

################################################################################


def symbols2atomicnumbers(symbols):
    n = len(symbols)
    atomicnumbers = []
    for i in np.arange(n):
        atomicnumbers.append(periodictable.to_atomic_number(symbols[i]))
    return atomicnumbers


def symbols2nelectrons(symbols):
    return sum(symbols2atomicnumbers(symbols))


def get_atomic_symbols_from_xyz(molecule):
    symbols = []

    infile = open(molecule, "r")
    infile.readline()
    infile.readline()

    for line in infile:
        symbols.append(line.split()[0])

    infile.close()

    return symbols


def get_atomic_symbols_from_str(molecule):
    symbols = []

    atoms = molecule.split(";")
    for el in atoms:
        symbols.append(el.split()[0])

    return symbols


def get_coords_from_str(molecule):
    coords = []

    atoms = molecule.split(";")
    for el in atoms:
        coords.append([el.split()[1], el.split()[2], el.split()[3]])

    return coords


def get_atomic_symbols(molecule):
    if molecule[-4:] == ".xyz":
        return get_atomic_symbols_from_xyz(molecule)
    else:
        return get_atomic_symbols_from_str(molecule)


def get_basis(basis, program):
    if type(basis) == str:
        return basis
    else:
        return basis[program]


def inspect_inputs(inputs):
    """determine if inputs argument is the output results of a simulation"""
    if isinstance(inputs, dict) and all(
        el in inputs.keys() for el in ["samples", "inputs", "arrays", "log", "misc"]
    ):
        return True
    elif isinstance(inputs, str):
        if inputs[-4:] == ".npz":
            return True
        elif inputs[-7:] == ".pickle":
            return True
    return False


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


def load_inputs(inputs_):
    if inputs_[-4:] == ".npz":
        inputs = np.load(inputs_, allow_pickle=True)
        inputs = cleanup_inputs()

    return cleanup_inputs(inputs)
