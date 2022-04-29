import daltonproject as dp
import numpy as np
import os

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)


from qcelemental import PhysicalConstantsContext
constants = PhysicalConstantsContext('CODATA2018')


def construct_dalton_system_ao():
    pass

def construct_dalton_system_rhf():
    pass

def setup_dp_dalton(input_file,basis_name,n_excited_states,n_electrons,method='CCSD',custom_basis=False):
    from utils import symbols2nelectrons

    molecule = dp.Molecule(input_file=input_file)
    n_electrons_neutral = symbols2nelectrons(molecule.elements)
    charge = n_electrons_neutral - n_electrons
    molecule.charge = charge

    if not custom_basis:
        basis = dp.Basis(basis=basis_name,custom_basis=custom_basis)
    else:
        basis = dp.Basis(basis=basis_name+'.dalinp',custom_basis=custom_basis)


    ccsd = dp.QCMethod(method)
    prop = dp.Property(response_vectors=True)
    prop.excitation_energies(states=n_excited_states)
    result = dp.dalton.compute(molecule,basis,ccsd,prop)

    return dp.dalton.Arrays(result)

def setup_dp_molcas(input_file,basis_name,omega,k,custom_basis=False,verbose=False):
    speed_of_light = constants.get('c_au')
    wavelength = 2*np.pi*speed_of_light/omega

    if verbose:
        print ('wavelength: ',wavelength)

    molecule = dp.Molecule(input_file=input_file)
    if not custom_basis:
        basis = dp.Basis(basis=basis_name,custom_basis=custom_basis)
    else:
        basis = dp.Basis(basis=basis_name+'.molinp',custom_basis=custom_basis)


    ccsd = dp.QCMethod('CCSD')
    prop = dp.Property(vector_potential=True)
    prop.vector_potential(k_direction=list(k),wavelength=wavelength)

    result = dp.molcas.compute(molecule,basis,ccsd,prop)

    return dp.molcas.Arrays(result)

def setup_system_da(da,n_electrons,c=None,change_basis=False):
    """sets up a QuantumSystem object with arrays computed from a
    daltonproject.dalton.arrays.Arrays object.
    c: custom HF coefficient matrix"""
    h = da.h
    s = da.s
    u = da.u            #slow implementation
    if c == None:
        c = da.c.T

    x = da.position(0)
    y = da.position(1)
    z = da.position(2)

    px = da.momentum(0)
    py = da.momentum(1)
    pz = da.momentum(2)

    l = len(h)

    position = np.zeros((3, l, l))
    momentum = np.zeros((3, l, l),dtype=complex)

    position[0] = x
    position[1] = y
    position[2] = z

    momentum[0] = px
    momentum[1] = py
    momentum[2] = pz

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = 0.0
    bs.position = position
    bs.momentum = momentum
    bs.change_module(np=np)
    system = SpatialOrbitalSystem(n_electrons, bs)
    if change_basis:
        system.change_basis(c)

    return system



def get_amps(da):
    amps = np.zeros(1,dtype='complex128')
    amps = np.concatenate((amps,da.t1.flatten()))
    amps = np.concatenate((amps,da.t2.flatten()))
    amps = np.concatenate((amps,da.l1.flatten()))
    amps = np.concatenate((amps,da.l2.flatten()))
    return amps


def get_response_vectors(da,nr_of_excited_states,excitation_levels=None,M1=True,M2=True,L1=True,L2=True,R1=True,R2=True):
    M1_, M2_, L1_, L2_, R1_, R2_ = [], [], [], [], [], []

    if excitation_levels is None:
        levels = np.arange(1,nr_of_excited_states+1)
    else:
        levels = excitation_levels

    for n in levels:
        if M1:M1_.append(da.M1(n))
        if M2:M2_.append(da.M2(n))
        if L1:L1_.append(da.L1(n))
        if L2:L2_.append(da.L2(n))
        if R1:R1_.append(da.R1(n))
        if R2:R2_.append(da.R2(n))

    ret = []
    if M1:ret.append(M1_)
    if M2:ret.append(M2_)
    if L1:ret.append(L1_)
    if L2:ret.append(L2_)
    if R1:ret.append(R1_)
    if R2:ret.append(R2_)

    return tuple(ret)
