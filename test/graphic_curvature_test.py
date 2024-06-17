
import unittest
from src.surface.surface_Fourier import *
import numpy as np
import logging


def surface_curvature_num(X, Y, Z):
    """
    Calculate the principle curvatures of a surface using numerical differentiation.

    Parameters:
    X (numpy.ndarray): The x coordinates of the surface grid.
    Y (numpy.ndarray): The y coordinates of the surface grid.
    Z (numpy.ndarray): The z coordinates of the surface grid.

    Returns:
    list: A list containing the maximum and minimum principle curvatures.
    """
    # Check if all arrays have the same shape
    assert X.shape == Y.shape == Z.shape

    # Unpack the shape of the surface grid
    lr, lb = X.shape

    # Calculate first derivatives of the surface coordinates
    Xu, Xv = np.gradient(X, 1/lr, 1/lb)
    Yu, Yv = np.gradient(Y, 1/lr, 1/lb)
    Zu, Zv = np.gradient(Z, 1/lr, 1/lb)

    # Calculate second derivatives of the surface coordinates
    Xuu, Xuv = np.gradient(Xu, 1/lr, 1/lb)
    Yuu, Yuv = np.gradient(Yu, 1/lr, 1/lb)
    Zuu, Zuv = np.gradient(Zu, 1/lr, 1/lb)

    Xvu, Xvv = np.gradient(Xv, 1/lr, 1/lb)
    Yvu, Yvv = np.gradient(Yv, 1/lr, 1/lb)
    Zvu, Zvv = np.gradient(Zv, 1/lr, 1/lb)

    # Create matrices for surface derivatives
    Pu = np.array([Xu, Yu, Zu])
    Pv = np.array([Xv, Yv, Zv])
    Puu = np.array([Xuu, Yuu, Zuu])
    Puv = np.array([Xuv, Yuv, Zuv])
    Pvv = np.array([Xvv, Yvv, Zvv])

    # Calculate coefficients of the surface
    E = np.einsum('lij,lij->ij', Pu, Pu)  # E
    F = np.einsum('lij,lij->ij', Pu, Pv)  # F
    G = np.einsum('lij,lij->ij', Pv, Pv)  # G

    # Calculate normal to the surface
    m = np.cross(Pu, Pv, axisa=0, axisb=0)
    p = np.sqrt(np.einsum('ijl,ijl->ij', m, m))
    n = m / p[:, :, np.newaxis]

    # Calculate coefficients of the surface tangent plane
    L = np.einsum('lij,ijl->ij', Puu, n)  # e
    M = np.einsum('lij,ijl->ij', Puv, n)  # f
    N = np.einsum('lij,ijl->ij', Pvv, n)  # g

    # Calculate Gaussian curvature
    K = (L * N - M**2) / (E * G - F**2)

    # Calculate Mean Curvature
    H = (E * N + G * L - 2*F * M) / (2*(E * G - F**2))

    # Calculate Principle Curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)

    # Return the principle curvatures
    return [Pmax, Pmin]


lu, lv = 128, 128
surface_parametrization = Surface_Fourier.load_file('data/li383/cws.txt')
surface_parametrization = Surface_Fourier.load_file('data/li383/cws.txt')
S = Surface_Fourier(surface_parametrization, (lu, lv), 3)
tmp = surface_curvature_num(S.X, S.Y, S.Z)
# Plot numerical vs analytical curvature
if False:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(S.grids[0], S.grids[1], tmp[1], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(S.grids[0], S.grids[1], S.principles[1], cmap=cm.PiYG,
                            linewidth=0, antialiased=False)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

plot_function_on_surface(S, np.maximum(
    np.abs(S.principles[0]), np.abs(S.principles[1])))
