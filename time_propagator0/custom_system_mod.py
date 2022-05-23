import warnings

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)

from time_propagator0.utils import get_atomic_symbols


def construct_pyscf_system_ao(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    **kwargs,
):
    """Convenience function setting up an atom or a molecule from PySCF as a
    ``QuantumSystem``.

    Parameters
    ----------
    molecule : str
        String describing the atom or molecule. This gets passed to PySCF which
        means that we support all the same string options as PySCF.
    basis : str
        String describing the basis set. PySCF determines which options are
        available.
    add_spin : bool
        Whether or not to return a ``SpatialOrbitalSystem`` (``False``) or a
        ``GeneralOrbitalSystem`` (``True``). Default is ``True``.
    anti_symmetrize : bool
        Whether or not to anti-symmetrize the two-body elements in a
        ``GeneralOrbitalSystem``. This only applies if ``add_spin = True``.
        Default is ``True``.
    np : module
        Array- and linear algebra module.

    Returns
    -------
    SpatialOrbitalSystem, GeneralOrbitalSystem
        Depending on the choice of ``add_spin`` we return a
        ``SpatialOrbitalSystem`` (``add_spin = False``), or a
        ``GeneralOrbitalSystem`` (``add_spin = True``).

    See Also
    -------
    PySCF

    Example
    -------
    >>> # Set up the Beryllium atom centered at (0, 0, 0)
    >>> system = construct_pyscf_system_ao(
    ...     "be 0 0 0", basis="cc-pVDZ", add_spin=False
    ... )
    >>> # Compare the number of occupied basis functions
    >>> system.n == 4 // 2
    True
    >>> gos = system.construct_general_orbital_system()
    >>> gos.n == 4
    True
    """

    import pyscf
    import basis_set_exchange as bse
    from time_propagator0.utils import get_atomic_symbols

    if np is None:
        import numpy as np

    atomic_symbols = get_atomic_symbols(molecule)

    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.basis = bse.api.get_basis(name=basis, fmt="nwchem", elements=atomic_symbols)
    mol.build(atom=molecule, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    l = mol.nao

    # n_a = (mol.nelectron + mol.spin) // 2
    # n_b = n_a - mol.spin

    # assert n_b == n - n_a

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)
    momentum = mol.intor("int1e_ipovlp").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.momentum = 1j * momentum
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )


def construct_pyscf_system_rhf(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    **kwargs,
):
    """Convenience function setting up a closed-shell atom or a molecule from
    PySCF as a ``QuantumSystem`` in RHF-basis using PySCF's RHF-solver.

    Parameters
    ----------
    molecule : str
        String describing the atom or molecule. This gets passed to PySCF which
        means that we support all the same string options as PySCF.
    basis : str
        String describing the basis set. PySCF determines which options are
        available.
    add_spin : bool
        Whether or not to return a ``SpatialOrbitalSystem`` (``False``) or a
        ``GeneralOrbitalSystem`` (``True``). Default is ``True``.
    anti_symmetrize : bool
        Whether or not to anti-symmetrize the two-body elements in a
        ``GeneralOrbitalSystem``. This only applies if ``add_spin = True``.
        Default is ``True``.
    np : module
        Array- and linear algebra module.

    Returns
    -------
    SpatialOrbitalSystem, GeneralOrbitalSystem
        Depending on the choice of ``add_spin`` we return a
        ``SpatialOrbitalSystem`` (``add_spin = False``), or a
        ``GeneralOrbitalSystem`` (``add_spin = True``).

    See Also
    -------
    PySCF

    Example
    -------
    >>> # Set up the Beryllium atom centered at (0, 0, 0)
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ", add_spin=False
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> # Compare the number of occupied basis functions
    >>> system.n == 4 // 2
    True
    >>> gos = system.construct_general_orbital_system()
    >>> gos.n == 4
    True
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ"
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> system.n == gos.n
    True
    """

    import pyscf
    import basis_set_exchange as bse
    from time_propagator0.utils import get_atomic_symbols

    if np is None:
        import numpy as np

    atomic_symbols = get_atomic_symbols(molecule)

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.basis = bse.api.get_basis(name=basis, fmt="nwchem", elements=atomic_symbols)
    mol.charge = charge
    mol.cart = cart
    mol.build(atom=molecule, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    assert n % 2 == 0, "We require closed shell, with an even number of particles"

    l = mol.nao

    hf = pyscf.scf.RHF(mol)
    hf_energy = hf.kernel()

    if not hf.converged:
        warnings.warn("RHF calculation did not converge")

    if verbose:
        print(f"RHF energy: {hf.e_tot}")

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)

    C = np.asarray(hf.mo_coeff)

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)
    momentum = mol.intor("int1e_ipovlp").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.momentum = 1j * momentum
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(C)

    return (
        (system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize), C)
        if add_spin
        else (system, C)
    )


def construct_dalton_system_ao(
    molecule,
    basis,
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    charge=0,
    custom_basis=False,
):
    """sets up a QuantumSystem object with arrays computed from a
    daltonproject.dalton.arrays.Arrays object.
    c: custom HF coefficient matrix"""
    import daltonproject as dp
    from time_propagator0.utils import symbols2nelectrons

    if np is None:
        import numpy as np

    if molecule[-4:] == ".xyz":
        mol = dp.Molecule(input_file=molecule)
    else:
        mol = dp.Molecule(atoms=molecule)

    mol.charge = charge

    n_electrons_neutral = symbols2nelectrons(mol.elements)
    n_electrons = n_electrons_neutral - charge

    basis_set = dp.Basis(basis=basis, custom_basis=custom_basis)

    ccsd = dp.QCMethod("CC2")
    prop = dp.Property(response_vectors=True)
    prop.excitation_energies(states=0)
    result = dp.dalton.compute(mol, basis_set, ccsd, prop)

    da = dp.dalton.Arrays(result)

    h = da.h
    s = da.s
    u = da.u  # slow implementation

    x = da.position(0)
    y = da.position(1)
    z = da.position(2)

    px = da.momentum(0)
    py = da.momentum(1)
    pz = da.momentum(2)

    l = len(h)

    position = np.zeros((3, l, l))
    momentum = np.zeros((3, l, l), dtype=complex)

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

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )


def construct_dalton_system_rhf(
    molecule,
    basis,
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    charge=0,
    custom_basis=False,
):
    """sets up a QuantumSystem object with arrays computed from a
    daltonproject.dalton.arrays.Arrays object.
    c: custom HF coefficient matrix"""
    import daltonproject as dp
    from time_propagator0.utils import symbols2nelectrons

    if np is None:
        import numpy as np

    if molecule[-4:] == ".xyz":
        mol = dp.Molecule(input_file=molecule)
    else:
        mol = dp.Molecule(atoms=molecule)

    mol.charge = charge

    n_electrons_neutral = symbols2nelectrons(mol.elements)
    n_electrons = n_electrons_neutral - charge

    basis_set = dp.Basis(basis=basis, custom_basis=custom_basis)

    ccsd = dp.QCMethod("CC2")
    prop = dp.Property(response_vectors=True)
    prop.excitation_energies(states=0)
    result = dp.dalton.compute(mol, basis_set, ccsd, prop)

    da = dp.dalton.Arrays(result)

    h = da.h
    s = da.s
    u = da.u  # slow implementation
    C = da.c.T

    x = da.position(0)
    y = da.position(1)
    z = da.position(2)

    px = da.momentum(0)
    py = da.momentum(1)
    pz = da.momentum(2)

    l = len(h)

    position = np.zeros((3, l, l))
    momentum = np.zeros((3, l, l), dtype=complex)

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
    system.change_basis(C)
    return (
        (system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize), C)
        if add_spin
        else (system, C)
    )
