# import jax
# jax.config.update("jax_enable_x64", True)


# from stellacode import np
# from stellacode.costs.em_cost import EMCost, get_b_field_err
# from stellacode.definitions import w7x_plasma, ncsx_plasma
# from stellacode.surface import (
#     Current,
#     CurrentZeroTorBC,
#     IntegrationParams,
#     ToroidalSurface,
# )
# from stellacode.surface import CylindricalSurface, FourierSurface, CoilSurface, MultipleCoils
# from stellacode.surface.imports import (
#     get_current_potential,
#     get_cws,
#     get_net_current,
#     get_plasma_surface,
#     get_cws_from_plasma_config,
# )
# from stellacode.surface.rotated_surface import RotatedSurface

# def test_multiple_tore():
#     n_harmonics = 4
#     factor = 4
#     surf_plasma = w7x_plasma
#     net_currents = get_net_current(surf_plasma.file_path)
#     current = Current(
#         num_pol=n_harmonics,
#         num_tor=n_harmonics,
#         # sin_basis=sin_basis,
#         # cos_basis=cos_basis,
#         net_currents=net_currents,
#     )

#     # minor_radius = surf_plasma.get_minor_radius()
#     # major_radius = surf_plasma.get_major_radius()
#     tor_surf = ToroidalSurface(
#         num_tor_symmetry=surf_plasma.num_tor_symmetry,
#         major_radius=5,
#         minor_radius=1,
#         integration_par=current.get_integration_params(factor=factor),
#     )
#     coil1 =  RotatedSurface(
#         surface=tor_surf,
#         num_tor_symmetry=surf_plasma.num_tor_symmetry,
#         rotate_diff_current=1,
#         current=current,
#     )
#     multiple_coils = [coil1]


# def test_cylindrical_coil_non_axisym():
#     current_n_coeff = 8
#     n_points = current_n_coeff * 4
#     # em_cost = EMCost.from_plasma_config(
#     #     plasma_config=w7x_plasma,
#     #     integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
#     #     lamb=1e-20,
#     # )

#     current = CurrentZeroTorBC(
#         num_pol=current_n_coeff,
#         num_tor=current_n_coeff,
#         sin_basis=True,
#         cos_basis=True,
#         net_currents=get_net_current(w7x_plasma.path_plasma),
#     )

#     fourier_coeffs = np.zeros((0, 2))

#     coil_surf = CoilSurface(
#         surface=CylindricalSurface(
#             fourier_coeffs=fourier_coeffs,
#             integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
#             num_tor_symmetry=9,
#         ),
#         # num_tor_symmetry=3,
#         # rotate_diff_current=3,
#         current=current,
#     )

#     coil_surf.surface.plot()
#     import pdb;pdb.set_trace()