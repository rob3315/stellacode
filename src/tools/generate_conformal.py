def generate_conformal_non_constant_separation(vmec_wout_file, fa, filepath="conformal_cws.txt"):
    import numpy as np
    from scipy.io import netcdf_file
    f = netcdf_file(vmec_wout_file, 'r', mmap=False)

    # Number of toroidal modes
    ntor = f.variables['ntor'][()]
    nmax = ntor + 1

    # Stellarator symmetry
    lasym = f.variables['lasym__logical__'][()]

    # Fourier coefficients of magnetic axis
    raxis_cc = f.variables['raxis_cc'][()]
    zaxis_cs = f.variables['zaxis_cs'][()]
    if lasym == 1:
        raxis_cs = f.variables['raxis_cs'][()]
        zaxis_cc = f.variables['zaxis_cc'][()]
    else:
        raxis_cs = np.zeros_like(raxis_cc)
        zaxis_cc = np.zeros_like(zaxis_cs)

    # Fourier coefficients of LCFS
    rmnc = f.variables['rmnc'][()][-1]
    zmns = f.variables['zmns'][()][-1]
    if lasym == 1:
        rmns = f.variables['rmns'][()]
        zmnc = f.variables['zmnc'][()]
    else:
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(zmns)

    # Initialize scaled coefficients
    rmnc_scaled = np.zeros_like(rmnc)
    zmns_scaled = np.zeros_like(zmns)
    rmns_scaled = np.zeros_like(rmns)
    zmnc_scaled = np.zeros_like(zmnc)

    # Scaling factors
    fR = 1

    # Compute scaled coefficients when m = 0
    rmnc_scaled[:nmax:] = fa * rmnc[:nmax:] + (fR - fa) * raxis_cc[:nmax:]
    zmns_scaled[:nmax:] = fa * zmns[:nmax:] + (fR - fa) * zaxis_cs[:nmax:]
    rmns_scaled[:nmax:] = fa * rmns[:nmax:] + (fR - fa) * raxis_cs[:nmax:]
    zmnc_scaled[:nmax:] = fa * zmnc[:nmax:] + (fR - fa) * zaxis_cc[:nmax:]

    # Compute scaled coefficients when m != 0
    rmnc_scaled[nmax::] = fa * rmnc[nmax::]
    zmns_scaled[nmax::] = fa * zmns[nmax::]
    rmns_scaled[nmax::] = fa * rmns[nmax::]
    zmnc_scaled[nmax::] = fa * zmnc[nmax::]

    # m and n arrays
    xm = f.variables['xm'][()]
    xn = f.variables['xn'][()] / f.variables['nfp'][()]

    # Close NetCDF file
    f.close()

    # Write text file
    if lasym == 0:
        data = np.column_stack((xm, -xn, rmnc_scaled, zmns_scaled))
    else:
        data = np.column_stack(
            (xm, -xn, rmnc_scaled, rmns_scaled, zmnc_scaled, zmns_scaled))

    np.savetxt(filepath, data, header="fourier")
