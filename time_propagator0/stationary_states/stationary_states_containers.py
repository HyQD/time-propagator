import abc


class StationaryStatesContainer(metaclass=abc.ABCMeta):
    @property
    def L1(self):
        return self._L1

    @property
    def L2(self):
        return self._L2

    @property
    def R1(self):
        return self._R1

    @property
    def R2(self):
        return self._R2

    @property
    def M1(self):
        return self._M1

    @property
    def M2(self):
        return self._M2

    @property
    def R0(self):
        return self._R0

    @property
    def t(self):
        return self._t

    @property
    def l(self):
        return self._l

    @property
    def C(self):
        return self._C

    @property
    def n_states(self):
        return self._n_states

    @L1.setter
    def L1(self, L1):
        self._L1 = L1

    @L2.setter
    def L2(self, L2):
        self._L2 = L2

    @R1.setter
    def R1(self, R1):
        self._R1 = R1

    @R2.setter
    def R2(self, R2):
        self._R2 = R2

    @M1.setter
    def M1(self, M1):
        self._M1 = M1

    @M2.setter
    def M2(self, M2):
        self._M2 = M2

    @R0.setter
    def R0(self, R0):
        self._R0 = R0

    @t.setter
    def t(self, t):
        self._t = t

    @l.setter
    def l(self, l):
        self._l = l

    @C.setter
    def C(self, C):
        self._C = C


class CIStatesContainer(StationaryStatesContainer):
    def __init__(self, C=None):
        self._C = C
        self._n_states = len(C[0, :])


class CCStatesContainer(StationaryStatesContainer):
    def __init__(
        self,
        L1,
        L2,
        R1,
        R2,
        M1=None,
        M2=None,
        R0=None,
        t=None,
        l=None,
    ):
        self._L1 = L1
        self._L2 = L2
        self._R1 = R1
        self._R2 = R2
        self._M1 = M1
        self._M2 = M2
        self._R0 = R0
        self._t = t
        self._l = l
        self._n_states = len(self._L1)


class CCStatesContainerMemorySaver(StationaryStatesContainer):
    def __init__(self, da, R0=None, t=None, l=None):
        class MakeSubscribtable:
            def __init__(self, f):
                self.f = f

            def __getitem__(self, n):
                return self.f(n + 1)

        self._L1 = MakeSubscribtable(da.L1)
        self._L2 = MakeSubscribtable(da.L2)
        self._R1 = MakeSubscribtable(da.R1)
        self._R2 = MakeSubscribtable(da.R2)
        self._M1 = MakeSubscribtable(da.M1)
        self._M2 = MakeSubscribtable(da.M2)
        self._R0 = None
        self._t = None
        self._l = None
        self._n_states = len(da.state_energies) - 1


def setup_CCStatesContainer_from_dalton(da, LR_projectors=False):
    n_states = len(da.state_energies) - 1

    L1, L2, R1, R2 = [], [], [], []
    if LR_projectors:
        M1, M2 = [], []

    for n in range(1, n_states + 1):
        L1.append(da.L1(n))
        L2.append(da.L2(n))
        R1.append(da.R1(n))
        R2.append(da.R2(n))
        if LR_projectors:
            M1.append(da.M1(n))
            M2.append(da.M2(n))

    return (
        CCStatesContainer(L1, L2, R1, R2, M1, M2)
        if LR_projectors
        else CCStatesContainer(L1, L2, R1, R2)
    )


def setup_CCStatesContainerMemorySaver_from_dalton(da):
    return CCStatesContainerMemorySaver(da)
