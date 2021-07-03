Set your own settings for your simulations
=============================================
The input of stellacode is a config file with INI text-structure.
This file contain all the numerical parameters and a link to the surfaces files and the bnorm file.

the config file
-----------------
::

    [geometry]
    np = 3 # The number of discrete symmetry needed to obtain the full torus
    ntheta_plasma = 64 # nb of toroidal points of the plasma used numerically (on one section)
    ntheta_coil = 64 # nb of toroidal points of the CWS used numerically (on one section)
    nzeta_plasma = 64  # nb of poloidal points of the plasma used numerically
    nzeta_coil = 64 # nb of poloidal points of the plasma used numerically
    mpol_coil = 12 # higher order of the Fourier harmonics in the toroidal(?) direction
    ntor_coil = 12 # higher order of the Fourier harmonics in the poloidal(?) direction
    path_plasma = data/li383/plasma_surf.txt
    path_cws = data/li383/cws.txt

    [other]
    path_bnorm = data/li383/bnorm.txt
    net_poloidal_current_amperes = 11884578.094260072
    net_toroidal_current_amperes = 0.
    curpol = 4.9782004309255496 # an unknown constant in front of bnorm...
    path_output = with_curv # output folder
    lamb = 2.5e-16 # lambda used in EM cost for the Tychonov regularization
    dask = True # Do not change
    cupy = False # Do not change


    [dask_parameters] # chunk for dask, do not modify if you do not know what you are doing
    chunk_theta_coil=32
    chunk_zeta_coil=32
    chunk_theta_plasma=32
    chunk_zeta_plasma=32
    chunk_theta=17

    [optimization_parameters] # The important section ;-)
    freq_save=100 #
    max_iter=2000 # number of iteration of the optimization algorithm (lower than the number of cost call)
    d_min=True # penalize low distance to plasma
    d_min_hard = 0.18 # wall
    d_min_soft= 0.20 # beginning of the penalization
    d_min_penalization=1000 # constant factor, a priori not very useful
    perim=True # penalization of the perimeter
    perim_c0=56 # beginning of the penalization
    perim_c1=60 # wall
    curvature=True # penalization of the inverse of the curvature radii
    curvature_c0=13 # beginning of the penalization
    curvature_c1=15 # beginning of wall

Note that the unit of the different quantities are
    - Ampere (A) for the currents
    - unknown for curpol...
    - metter (m) for the distance to the plasma
    - metter square (m^2) for the perimeter
    - inverse of the metter (m^-1) for the curvature (i.e. principal curvatures)

the surface files
-----------------
The surface files have the form :
::

    0    0  1.3278E+00  0.0000E+00
    0    1  4.2527E-03 -2.4153E-02
    0    2  4.6118E-05 -9.3831E-03
    0    3  7.1383E-03  3.4517E-03
    0    4  5.0157E-03  2.3650E-03
    0    5  1.6390E-03  9.4419E-04
    0    6 -2.4003E-04 -9.4647E-04
    0    7 -1.2541E-04  1.8339E-04

where for each row we have
::

    m   n   Rmn     Zmn

the bnorm file
-----------------
the bnorm file has a similar structure
::

    0  -24  1.7668311833075789E-11

where for each row we have
::

    m   n   Bmn