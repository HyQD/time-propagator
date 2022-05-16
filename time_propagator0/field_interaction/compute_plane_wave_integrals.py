def compute_vpi(self, molecule, omega, k_direction):
    print("Running Molcas, omega: ", omega)
    self.ma = setup_plane_wave_integrals_from_molcas(
        self.input_file,
        self.basis,
        omega,
        k_direction,
        custom_basis=self.custom_basis,
    )

    l = self.n_basis
    cosp = np.zeros((3, l, l), dtype=complex)
    sinp = np.zeros((3, l, l), dtype=complex)
    cosp[0] = self.transform_operator(self.ma.cosp(0))
    cosp[1] = self.transform_operator(self.ma.cosp(1))
    cosp[2] = self.transform_operator(self.ma.cosp(2))
    sinp[0] = self.transform_operator(self.ma.sinp(0))
    sinp[1] = self.transform_operator(self.ma.sinp(1))
    sinp[2] = self.transform_operator(self.ma.sinp(2))

    cos2 = self.transform_operator(self.ma.cos2)
    sin2 = self.transform_operator(self.ma.sin2)

    return cosp, sinp, cos2, sin2
