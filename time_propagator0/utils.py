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
