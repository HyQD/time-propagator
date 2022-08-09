import numpy as np

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)

from time_propagator0.utils import get_atomic_symbols


class QuantumSystemValues:
    def __init__(self):
        self.h = None
        self.u = None
        self.s = None
        self.position = None
        self.momentum = None
        self.quadrupole_moment = None
        self.C = None
        self.n = None
        self.l = None
        self.hf_energy = None
        self.nuclear_repulse_energy = None
        self.converged = None

    def set_pyscf_values(obj):
        pass

    def set_pyscf_values_ao(self, mol):
        # charges = mol.atom_charges()
        # coords = mol.atom_coords()
        # nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
        # mol.set_common_orig_(nuc_charge_center)

        self.n = mol.nelectron
        self.l = mol.nao

        self.nuclear_repulsion_energy = mol.energy_nuc()

        self.h = mol.get_hcore()
        self.s = mol.intor_symmetric("int1e_ovlp")

        l = self.l
        self.u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
        self.position = mol.intor("int1e_r").reshape(3, l, l)
        self.momentum = 1j * mol.intor("int1e_ipovlp").reshape(3, l, l)

        return self

    def set_pyscf_values_rhf(self, hf):
        self.set_pyscf_values_ao(hf.mol)

        self.hf_energy = hf.e_tot
        self.C = hf.mo_coeff

        self.converged = hf.converged

        return self

    def set_dalton_values_rhf(self, a):
        import daltonproject as dp

        self.nuclear_repulsion_energy = a.nuclear_repulsion_energy
        self.hf_energy = a.electronic_energy
        self.n = a.num_electrons
        self.l = a.num_orbitals.tot_num_orbitals

        da = dp.dalton.Arrays(a)
        self.h = da.h
        self.s = da.s
        self.u = da.u
        self.C = da.c.T

        l = self.l

        position = np.zeros((3, l, l))
        momentum = np.zeros((3, l, l), dtype=complex)

        position[0] = da.position(0)
        position[1] = da.position(1)
        position[2] = da.position(2)

        momentum[0] = da.momentum(0)
        momentum[1] = da.momentum(1)
        momentum[2] = da.momentum(2)

        self.position = position
        self.momentum = momentum

        return self


def run_pyscf_ao(
    molecule,
    basis="cc-pvdz",
    charge=0,
    cart=False,
    **kwargs,
):
    import pyscf
    import basis_set_exchange as bse
    from time_propagator0.utils import get_atomic_symbols

    atomic_symbols = get_atomic_symbols(molecule)

    mol = pyscf.gto.Mole()
    mol.charge = charge
    mol.cart = cart
    mol.unit = "bohr"
    mol.basis = bse.api.get_basis(name=basis, fmt="nwchem", elements=atomic_symbols)
    mol.build(atom=molecule, verbose=False, **kwargs)

    return mol


def run_pyscf_rhf(
    molecule,
    basis="cc-pvdz",
    charge=0,
    cart=False,
    conv_tol_grad=1e-10,
    **kwargs,
):
    import pyscf

    mol = run_pyscf_ao(
        molecule,
        basis=basis,
        charge=charge,
        cart=cart,
        **kwargs,
    )

    hf = pyscf.scf.RHF(mol)
    hf.conv_tol_grad = conv_tol_grad
    hf_energy = hf.kernel()

    return hf


def run_dalton_rhf(
    molecule,
    basis="cc-pvdz",
    charge=0,
    custom_basis=False,
    **kwargs,
):
    import daltonproject as dp
    from time_propagator0.utils import symbols2nelectrons

    if molecule[-4:] == ".xyz":
        mol = dp.Molecule(input_file=molecule)
    else:
        mol = dp.Molecule(atoms=molecule)

    mol.charge = charge

    n_electrons_neutral = symbols2nelectrons(mol.elements)
    n_electrons = n_electrons_neutral - charge

    basis_set = dp.Basis(basis=basis, custom_basis=custom_basis)

    ccsd = dp.QCMethod("CCS")
    prop = dp.Property(response_vectors=True)
    prop.excitation_energies(states=0)
    result = dp.dalton.compute(mol, basis_set, ccsd, prop, verbose=False)

    return result


def construct_quantum_system(qsv, add_spin=False, anti_symmetrize=False):
    bs = BasisSet(qsv.l, dim=3, np=np)
    bs.h = qsv.h
    bs.s = qsv.s
    bs.u = qsv.u
    bs.nuclear_repulsion_energy = qsv.nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = qsv.position
    bs.momentum = qsv.momentum
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(qsv.n, bs)
    system.change_basis(qsv.C)

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )
