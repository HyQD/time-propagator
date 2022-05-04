import numpy as np


def T1_transform(a, t1):
    nvir = t1.shape[0]
    nocc = t1.shape[1]
    n = nocc + nvir
    o = slice(0, nocc)
    v = slice(nocc, n)
    tt1 = np.zeros((n, n))
    tt1[v, o] += t1
    A = np.einsum("pr,rq->pq", a, tt1)
    A -= np.einsum("pr,rq->pq", tt1, a)
    A -= np.einsum("pr,rs,sq->pq", tt1, a, tt1, optimize=True)
    A += a
    return A


def compute_R0_(l1, l2, R1, R2):
    val1 = -np.einsum("ia,ai->", l1, R1)
    val2 = -0.5 * np.einsum("ijab,abij->", l2, R2)
    val = val1 + val2
    return val


def eom_transition_moment_(l1, l2, t1, t2, R0, R1, R2, a, o, v):
    """
    R0<psi0|A|psi0>
    """
    val = 2 * np.einsum("ii->", a[o, o])
    val -= np.einsum("ij,kjab,baik->", a[o, o], l2, t2, optimize=True)
    val -= np.einsum("ij,ja,ai->", a[o, o], l1, t1, optimize=True)

    val += np.einsum("ia,ia->", a[o, v], l1)

    val += 2 * np.einsum("ai,ai->", a[v, o], t1)
    val -= np.einsum("ai,ak,jkcb,bcij->", a[v, o], t1, l2, t2, optimize=True)
    val -= np.einsum("ai,ci,jkcb,abjk->", a[v, o], t1, l2, t2, optimize=True)
    val -= np.einsum("ai,jb,abji->", a[v, o], l1, t2, optimize=True)
    val += 2 * np.einsum("ai,jb,abij->", a[v, o], l1, t2, optimize=True)
    val -= np.einsum("ai,jb,aj,bi->", a[v, o], l1, t1, t1, optimize=True)

    val += np.einsum("ab,ia,bi->", a[v, v], l1, t1, optimize=True)
    val += np.einsum("ab,ijac,bcij->", a[v, v], l2, t2, optimize=True)

    val *= R0

    # print('Mine: reference contribution= ', val)

    xxx = eom_trm(l1, l2, t1, t2, R0, R1, R2, a, o, v)
    # print('      trans.mom. contribution=', xxx)

    val += xxx

    return val


def eom_trm(l1, l2, t1, t2, R0, R1, R2, a, o, v, biorthonormal=False):
    A = T1_transform(a, t1)

    trA_occ = np.einsum("ii->", A[o, o])

    val_0_A_R1_0 = 2 * np.einsum("ia,ai->", A[o, v], R1)

    val_0_l1_A_R1_0 = 2 * trA_occ * np.einsum("ia,ai->", l1, R1)
    val_0_l1_A_R1_0 += np.einsum("ia,ab,bi->", l1, A[v, v], R1, optimize=True)
    val_0_l1_A_R1_0 -= np.einsum("ia,ji,aj->", l1, A[o, o], R1, optimize=True)

    val_0_l2_A_R1_0 = 0.5 * np.einsum("ijab,ai,bj->", l2, R1, A[v, o], optimize=True)

    val_0_l2_AT2_R1_0 = np.einsum(
        "ijab,bj,acik,kc->", l2, R1, t2, A[o, v], optimize=True
    )
    val_0_l2_AT2_R1_0 -= 0.5 * np.einsum(
        "ijab,bj,acki,kc->", l2, R1, t2, A[o, v], optimize=True
    )
    val_0_l2_AT2_R1_0 -= 0.5 * np.einsum(
        "ijab,acij,kc,bk->", l2, t2, A[o, v], R1, optimize=True
    )
    val_0_l2_AT2_R1_0 -= 0.5 * np.einsum(
        "ijab,abik,kc,cj->", l2, t2, A[o, v], R1, optimize=True
    )

    val_0_l1_A_R2_0 = 2 * np.einsum("ia,abij,jb->", l1, R2, A[o, v], optimize=True)
    val_0_l1_A_R2_0 -= np.einsum("ia,abji,jb->", l1, R2, A[o, v], optimize=True)

    val_0_l2_A_R2_0 = 0.5 * trA_occ * np.einsum("ijab,abij->", l2, R2)
    val_0_l2_A_R2_0 += 0.5 * np.einsum("ijab,acij,bc->", l2, R2, A[v, v], optimize=True)
    val_0_l2_A_R2_0 -= 0.5 * np.einsum("ijab,abik,kj->", l2, R2, A[o, o], optimize=True)

    if biorthonormal:  # biorthonormal projection manifold
        val = val_0_A_R1_0
        val += val_0_l1_A_R1_0
        val += val_0_l2_A_R1_0
        val += val_0_l2_AT2_R1_0
        val += val_0_l1_A_R2_0
        val += val_0_l2_A_R2_0
    else:  # biorthogonal projection manifold
        val = val_0_A_R1_0
        val += val_0_l1_A_R1_0
        val += 2 * val_0_l2_A_R1_0
        val += 2 * val_0_l2_AT2_R1_0
        val += val_0_l1_A_R2_0
        val += 2 * val_0_l2_A_R2_0

    return val


def print_eta(f, u, t1, o, v):
    eta1 = (2 * f[o, v]).T
    eta1 += 4 * np.einsum("bj,ijab->ai", t1, u[o, o, v, v])
    eta1 -= 2 * np.einsum("bj,ijba->ai", t1, u[o, o, v, v])
    eta2 = 4 * u[v, v, o, o]
    eta2 -= 2 * np.swapaxes(u[v, v, o, o], 0, 1)
    nocc = t1.shape[1]
    nvir = t1.shape[0]
    print("eta1:")
    for a in range(nvir):
        if a == 0:
            for i in range(nocc):
                print("        ", i + 1, end="")
            print()
        for i in range(nocc):
            if i == 0:
                print(a + 1, " ", eta1[a, i], end="")
            else:
                print("  ", eta1[a, i], end="")
        print()
    for ii in range(nocc):
        i = ii + 1
        for aa in range(nvir):
            a = aa + 1
            ai = nvir * (i - 1) + a
            for jj in range(nocc):
                j = jj + 1
                for bb in range(nvir):
                    b = bb + 1
                    bj = nvir * (j - 1) + b
                    if ai >= bj and np.abs(eta2[aa, bb, ii, jj]) > 1.0e-8:
                        print("eta2(ai,bj):", ai, bj, eta2[aa, bb, ii, jj])


def compute_R0(f, u, t1, t2, R1, R2, dE_n, o, v):
    val1 = 2 * np.einsum("ai,ia->", R1, f[o, v])
    val1 += 4 * np.einsum("ai,bj,ijab->", R1, t1, u[o, o, v, v], optimize=True)
    val1 -= 2 * np.einsum("ai,bj,ijba->", R1, t1, u[o, o, v, v], optimize=True)
    val1 /= dE_n
    val2 = 2 * np.einsum("abij,ijab->", R2, u[o, o, v, v])
    val2 -= np.einsum("abij,ijba->", R2, u[o, o, v, v])
    val2 /= dE_n
    val = val1 + val2
    return val


def eom_transition_moment(l1, l2, t1, t2, R0, R1, R2, a, o, v):
    """
    Equation [55] in notes
    """

    """
    R0<psi0|A|psi0>
    """
    xxx = 2 * np.einsum("ii->", a[o, o])
    xxx -= np.einsum("ij,kjab,baik->", a[o, o], l2, t2, optimize=True)
    xxx -= np.einsum("ij,ja,ai->", a[o, o], l1, t1, optimize=True)

    xxx += np.einsum("ia,ia->", a[o, v], l1)

    xxx += 2 * np.einsum("ai,ai->", a[v, o], t1)
    xxx -= np.einsum("ai,ak,jkcb,bcij->", a[v, o], t1, l2, t2, optimize=True)
    xxx -= np.einsum("ai,ci,jkcb,abjk->", a[v, o], t1, l2, t2, optimize=True)
    xxx -= np.einsum("ai,jb,abji->", a[v, o], l1, t2, optimize=True)
    xxx += 2 * np.einsum("ai,jb,abij->", a[v, o], l1, t2, optimize=True)
    xxx -= np.einsum("ai,jb,aj,bi->", a[v, o], l1, t1, t1, optimize=True)

    xxx += np.einsum("bc,ib,ci->", a[v, v], l1, t1, optimize=True)
    xxx += np.einsum("bc,ijbd,cdij->", a[v, v], l2, t2, optimize=True)

    xxx *= R0

    # print('Håkon: reference contribution= ',xxx)

    """
    <phi0|Abar*R1|phi0>

    <phi0|Abar*R2|phi0> = 0
    """
    val = 2 * np.einsum("bi,ib->", R1, a[o, v])

    """
    <phi0|Lambda*Abar*R1|phi0>
    """
    val += np.einsum("bi,cb,ic->", R1, a[v, v], l1, optimize=True)
    val += np.einsum("bi,cd,dj,ijbc->", R1, a[v, v], t1, l2, optimize=True)
    val -= np.einsum("bi,ic,jkbd,cdjk->", R1, a[o, v], l2, t2, optimize=True)
    val -= np.einsum("bi,jb,ikcd,cdjk->", R1, a[o, v], l2, t2, optimize=True)
    val -= np.einsum("bi,jc,ikbd,cdkj->", R1, a[o, v], l2, t2, optimize=True)
    val += 2 * np.einsum("bi,jc,ikbd,cdjk->", R1, a[o, v], l2, t2, optimize=True)
    val -= np.einsum("bi,jc,ck,dj,ikbd->", R1, a[o, v], t1, t1, l2, optimize=True)
    val -= np.einsum("bi,jk,cj,ikbc->", R1, a[o, o], t1, l2, optimize=True)
    val -= np.einsum("bi,ij,jb->", R1, a[o, o], l1, optimize=True)
    val += 2 * np.einsum("bi,jj,ib->", R1, a[o, o], l1, optimize=True)
    val += np.einsum("bi,cj,ijbc->", R1, a[v, o], l2, optimize=True)
    val -= np.einsum("bi,ic,jb,cj->", R1, a[o, v], l1, t1, optimize=True)
    val -= np.einsum("bi,jb,ic,cj->", R1, a[o, v], l1, t1, optimize=True)
    val += 2 * np.einsum("bi,jc,ib,cj->", R1, a[o, v], l1, t1, optimize=True)

    """
    <phi0|Lambda*Abar*R2|phi0>
    """
    val += np.einsum("db,bcij,jicd->", a[v, v], R2, l2, optimize=True)
    val += np.einsum("kk,bcij,ijbc->", a[o, o], R2, l2, optimize=True)
    val -= np.einsum("ik,bcij,jkcb->", a[o, o], R2, l2, optimize=True)
    val += np.einsum("kd,dk,bcij,ijbc->", a[o, v], t1, R2, l2, optimize=True)
    val -= np.einsum("id,dk,bcij,jkcb->", a[o, v], t1, R2, l2, optimize=True)
    val -= np.einsum("kb,dk,bcij,jicd->", a[o, v], t1, R2, l2, optimize=True)
    val -= np.einsum("ib,jc,bcji->", a[o, v], l1, R2, optimize=True)
    val += 2 * np.einsum("ib,jc,bcij->", a[o, v], l1, R2, optimize=True)

    # print('       trans.mom. contribution=', val)
    val += xxx

    return val


"""
The ksi functions are
    ksi_A_mu = <Phi0|Ymu*exp(-T)*A*exp(t)|Phi0>

and the eta functions are
    eta_A_mu = <Phi0|(1+Lambda)*exp(-T)*[A,Xmu]*exp(T)|Phi0>

where A = a_pq * c_p^\dagger c_q is a one-body operator with matrix
elements a_pq.
"""


def ksi_A_0(t1, a, o, v):
    value = 2 * np.einsum("ib,bi->", a[o, v], t1)
    value += 2 * np.einsum("ii->", a[o, o])
    return value


def ksi_A_1(t1, t2, a, o, v):

    no, nv = t1.shape[1], t1.shape[0]
    rhs = np.zeros((nv, no))

    rhs += np.einsum("ab,bi->ai", a[v, v], t1)

    rhs += np.einsum("ai->ai", a[v, o])

    rhs -= np.einsum("jb,abji->ai", a[o, v], t2)

    rhs += 2 * np.einsum("jb,abij->ai", a[o, v], t2)

    rhs -= np.einsum("jb,aj,bi->ai", a[o, v], t1, t1, optimize=True)

    rhs -= np.einsum("ji,aj->ai", a[o, o], t1)

    return rhs


def ksi_A_2(t1, t2, a, o, v):
    no, nv = t1.shape[1], t1.shape[0]
    rhs = np.zeros((nv, nv, no, no))

    rhs += np.einsum("ac,bcji->abij", a[v, v], t2)

    rhs += np.einsum("bc,acij->abij", a[v, v], t2)

    rhs -= np.einsum("kc,ak,bcji->abij", a[o, v], t1, t2, optimize=True)

    rhs -= np.einsum("kc,bk,acij->abij", a[o, v], t1, t2, optimize=True)

    rhs -= np.einsum("kc,ci,abkj->abij", a[o, v], t1, t2, optimize=True)

    rhs -= np.einsum("kc,cj,abik->abij", a[o, v], t1, t2, optimize=True)

    rhs -= np.einsum("ki,abkj->abij", a[o, o], t2)

    rhs -= np.einsum("kj,abik->abij", a[o, o], t2)
    # scale diagonal by 1/2: required to get same result as Dalton CCLR
    rhs = np.swapaxes(rhs, 1, 2).reshape((nv * no, nv * no))
    rhs[np.diag_indices_from(rhs)] *= 0.5
    rhs = np.swapaxes(rhs.reshape((nv, no, nv, no)), 1, 2)
    return rhs


def eta_A_1(l1, l2, t1, t2, a, o, v):

    no, nv = t1.shape[1], t1.shape[0]

    eta = np.zeros((no, nv))

    eta += 2 * np.einsum("ia->ia", a[o, v])

    eta += np.einsum("ba,ib->ia", a[v, v], l1)

    eta -= np.einsum("ib,jkac,bcjk->ia", a[o, v], l2, t2, optimize=True)

    eta -= np.einsum("ja,ikbc,bcjk->ia", a[o, v], l2, t2, optimize=True)

    eta -= np.einsum("ib,ja,bj->ia", a[o, v], l1, t1, optimize=True)

    eta -= np.einsum("ja,ib,bj->ia", a[o, v], l1, t1, optimize=True)

    eta -= np.einsum("ij,ja->ia", a[o, o], l1)

    return eta


def eta_A_2(l1, l2, t1, t2, a, o, v):

    no, nv = t1.shape[1], t1.shape[0]

    eta = np.zeros((no, no, nv, nv))

    eta += np.einsum("ca,ijcb->ijab", a[v, v], l2)

    eta += np.einsum("cb,ijac->ijab", a[v, v], l2)

    eta -= np.einsum("ib,ja->ijab", a[o, v], l1)

    eta -= np.einsum("ja,ib->ijab", a[o, v], l1)

    eta += 2 * np.einsum("ia,jb->ijab", a[o, v], l1)

    eta += 2 * np.einsum("jb,ia->ijab", a[o, v], l1)

    eta -= np.einsum("ic,ck,kjab->ijab", a[o, v], t1, l2, optimize=True)

    eta -= np.einsum("jc,ck,ikab->ijab", a[o, v], t1, l2, optimize=True)

    eta -= np.einsum("ka,ck,ijcb->ijab", a[o, v], t1, l2, optimize=True)

    eta -= np.einsum("kb,ck,ijac->ijab", a[o, v], t1, l2, optimize=True)

    eta -= np.einsum("ik,kjab->ijab", a[o, o], l2)

    eta -= np.einsum("jk,ikab->ijab", a[o, o], l2)

    return eta
