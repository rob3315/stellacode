import numpy as np

from stellacode.surface.fourier import *


def surface_curvature_num(X, Y, Z):
    # from  sujithTSR /surface-curvature
    (lr, lb) = X.shape
    print(lr)
    print(lb)
    # First Derivatives
    Xu, Xv = np.gradient(X, 1 / lr, 1 / lb)
    Yu, Yv = np.gradient(Y, 1 / lr, 1 / lb)
    Zu, Zv = np.gradient(Z, 1 / lr, 1 / lb)
    # 	print(Xu)

    # Second Derivatives
    Xuu, Xuv = np.gradient(Xu, 1 / lr, 1 / lb)
    Yuu, Yuv = np.gradient(Yu, 1 / lr, 1 / lb)
    Zuu, Zuv = np.gradient(Zu, 1 / lr, 1 / lb)

    Xvu, Xvv = np.gradient(Xv, 1 / lr, 1 / lb)
    Yvu, Yvv = np.gradient(Yv, 1 / lr, 1 / lb)
    Zvu, Zvv = np.gradient(Zv, 1 / lr, 1 / lb)

    Pu = np.array([Xu, Yu, Zu])
    Pv = np.array([Xv, Yv, Zv])
    Puu = np.array([Xuu, Yuu, Zuu])
    Puv = np.array([Xuv, Yuv, Zuv])
    Pvv = np.array([Xvv, Yvv, Zvv])

    # % First fundamental Coeffecients of the surface (E,F,G)

    E = np.einsum("lij,lij->ij", Pu, Pu)
    F = np.einsum("lij,lij->ij", Pu, Pv)
    G = np.einsum("lij,lij->ij", Pv, Pv)

    m = np.cross(Pu, Pv, axisa=0, axisb=0)
    p = np.sqrt(np.einsum("ijl,ijl->ij", m, m))
    n = m / p[:, :, np.newaxis]
    # n is the normal
    # % Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
    L = np.einsum("lij,ijl->ij", Puu, n)  # e
    M = np.einsum("lij,ijl->ij", Puv, n)  # f
    N = np.einsum("lij,ijl->ij", Pvv, n)  # g

    # Alternative formula for gaussian curvature in wiki
    # K = det(second fundamental) / det(first fundamental)
    # % Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)
    # 	print(K.size)
    # wiki trace of (second fundamental)(first fundamental inverse)
    # % Mean Curvature
    H = ((E * N + G * L - 2 * F * M) / ((E * G - F**2))) / 2
    # 	print(H.size)

    # % Principle Curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)
    # [Pmax, Pmin]
    Principle = [Pmax, Pmin]
    return Principle


def test_graphic_curvature():
    lu, lv = 128, 128
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)

    tmp = surface_curvature_num(S.P[:, :, 0], S.P[:, :, 1], S.P[:, :, 2])
    # Plot numerical vs analytical curvature
    if False:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            S.grids[0],
            S.grids[1],
            tmp[1],
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        surf2 = ax.plot_surface(
            S.grids[0],
            S.grids[1],
            S.principles[1],
            cmap=cm.PiYG,
            linewidth=0,
            antialiased=False,
        )
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter("{x:.02f}")
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    # plot_function_on_surface(S,np.maximum(np.abs(S.principles[0]),np.abs(S.principles[1])))
