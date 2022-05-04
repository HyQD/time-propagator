import numpy as np


def two_component_EOM_projector(
    L1, L2, l1_t, l2_t, t0_t, t1_t, t2_t, t1, t2, R0, R1, R2
):
    """
    Eq. 45
    """
    psi_tilde_psi_n = func_psi_tilde_psi_n(
        l1_t, l2_t, t1_t, t2_t, t1, t2, R1, R2
    ) * np.exp(-t0_t)

    """
    Eq. 46
    """
    psi_tilde_n_psi = func_psi_tilde_n_psi(L1, L2, t1_t, t2_t, t1, t2) * np.exp(t0_t)

    """
    Eq 61
    """
    psi_tilde_t_psi = tilde_psi_t_psi_0(l1_t, l2_t, t1_t, t2_t, t1, t2) * np.exp(-t0_t)

    """
    Eq. 60
    """
    projector = psi_tilde_psi_n
    projector += psi_tilde_n_psi.conj()
    projector += psi_tilde_t_psi * R0

    return 0.25 * np.abs(projector) ** 2


def conventional_EOM_projector(
    L1, L2, l1_t, l2_t, t0_t, t1_t, t2_t, t1, t2, R0, R1, R2
):
    """
    Eq. 45
    """
    psi_tilde_psi_n = func_psi_tilde_psi_n(
        l1_t, l2_t, t1_t, t2_t, t1, t2, R1, R2
    ) * np.exp(-t0_t)

    """
    Eq. 46
    """
    psi_tilde_n_psi = func_psi_tilde_n_psi(L1, L2, t1_t, t2_t, t1, t2) * np.exp(t0_t)

    """
    Eq 61
    """
    psi_tilde_t_psi = tilde_psi_t_psi_0(l1_t, l2_t, t1_t, t2_t, t1, t2) * np.exp(-t0_t)

    """
    Eq. 66
    """
    projector = (psi_tilde_psi_n + R0 * psi_tilde_t_psi) * psi_tilde_n_psi
    return projector.real


def EOM_projector(L1, L2, l1_t, l2_t, t0_t, t1_t, t2_t, t1, t2, R1, R2):
    psi_tilde_psi_n = func_psi_tilde_psi_n(
        l1_t, l2_t, t1_t, t2_t, t1, t2, R1, R2
    ) * np.exp(t0_t)
    psi_tilde_n_psi_star = (
        func_psi_tilde_n_psi(L1, L2, t1_t, t2_t, t1, t2) * np.exp(-t0_t)
    ).conj()
    return 0.25 * np.abs(psi_tilde_psi_n + psi_tilde_n_psi_star) ** 2


def LR_projector(
    M1, M2, L1, L2, l1, l2, l1_t, l2_t, t1, t2, t0_t, t1_t, t2_t, R0, R1, R2
):

    """
    Eq. [45]
    """
    psi_tilde_psi_n = func_psi_tilde_psi_n(
        l1_t, l2_t, t1_t, t2_t, t1, t2, R1, R2
    ) * np.exp(-t0_t)

    """
    Eq. [46]
    """
    psi_tilde_n_psi = func_psi_tilde_n_psi(L1, L2, t1_t, t2_t, t1, t2) * np.exp(t0_t)

    """
    Eq. [44]
    """
    psi_bar_n_psi = func_psi_bar_n_psi(
        M1, M2, l1, l2, t1_t, t2_t, t1, t2, R1, R2
    ) * np.exp(t0_t)

    tpsi_psi0 = rccsd_overlap(l1_t, l2_t, t1_t, t2_t, t1, t2) * np.exp(-t0_t)
    tpsi0_psi = rccsd_overlap(l1, l2, t1, t2, t1_t, t2_t) * np.exp(t0_t)
    val = R0 * (tpsi_psi0 - tpsi0_psi)

    lr_projection = (psi_bar_n_psi + psi_tilde_psi_n + val) * psi_tilde_n_psi

    return lr_projection.real


def rccsd_overlap(l1, l2, t1_bra, t2_bra, t1_ket, t2_ket):
    dt1 = t1_ket - t1_bra
    val = 1
    val += np.einsum("ia,ai->", l1, dt1)
    val += 0.5 * np.einsum("ijab,abij->", l2, t2_ket - t2_bra)
    val += 0.5 * np.einsum("ijab,ai,bj->", l2, dt1, dt1, optimize=True)
    return val


def tilde_psi_t_psi_0(l_1_t2, l_2_t2, t_1_t2, t_2_t2, t_1_t1, t_2_t1):
    """
    Eq. 61
    """
    psi_t2_t1 = 1
    psi_t2_t1 += 0.5 * np.einsum("ijab,abij->", l_2_t2, t_2_t1)
    psi_t2_t1 -= 0.5 * np.einsum("ijab,abij->", l_2_t2, t_2_t2)

    psi_t2_t1 += 0.5 * np.einsum("ai,bj,ijab->", t_1_t2, t_1_t2, l_2_t2, optimize=True)

    psi_t2_t1 += 0.5 * np.einsum("ai,bj,ijab->", t_1_t1, t_1_t1, l_2_t2, optimize=True)

    psi_t2_t1 -= np.einsum("ai,bj,ijab->", t_1_t2, t_1_t1, l_2_t2, optimize=True)

    psi_t2_t1 += np.einsum("ia,ai->", l_1_t2, t_1_t1)
    psi_t2_t1 -= np.einsum("ia,ai->", l_1_t2, t_1_t2)

    return psi_t2_t1


def func_psi_bar_n_psi(M1, M2, l1, l2, t1_t, t2_t, t1, t2, R1, R2):
    val = func_psi_tilde_n_psi(M1, M2, t1_t, t2_t, t1, t2)
    val -= func_psi_tilde_psi_n(l1, l2, t1, t2, t1_t, t2_t, R1, R2)
    return val


def func_psi_tilde_psi_n(l1_t, l2_t, t1_t, t2_t, t1, t2, R1, R2):
    val = np.einsum("ia,ai->", l1_t, R1, optimize=True)
    val += 0.5 * np.einsum("ijab,abij->", l2_t, R2, optimize=True)
    val += np.einsum("ai,bj,ijab->", R1, t1, l2_t, optimize=True)
    val -= np.einsum("ai,bi,ijab->", R1, t1_t, l2_t, optimize=True)
    return val


def func_psi_tilde_n_psi(L1, L2, t1_t, t2_t, t1, t2):
    val = np.einsum("ia,ai->", L1, t1_t, optimize=True)
    val -= np.einsum("ia,ai->", L1, t1, optimize=True)
    val += 0.5 * np.einsum("ijab,abij->", L2, t2_t, optimize=True)
    val += 0.5 * np.einsum("ai,bj,ijab->", t1_t, t1_t, L2, optimize=True)
    val -= 0.5 * np.einsum("ijab,abij->", L2, t2, optimize=True)
    val -= np.einsum("ai,bj,ijab->", t1, t1_t, L2, optimize=True)
    val += 0.5 * np.einsum("ai,bj,ijab->", t1, t1, L2, optimize=True)
    return val
