import matplotlib.pyplot as plt
from stellacode.surface import Surface, CoilFactory
import numpy as onp


def plot_cross_sections(coil_factory: CoilFactory, Sp: Surface,
                        save_fig: bool = False, output_path: str = ""):
    """
    Plot a cross section of the CWS and several cross sections of the LCFS

    Args:
        coil_factory (CoilFactory): CoilFactory object for the CWS
        Sp (Surface): Surface object for the LCFS
        save_fig (bool, optional): Flag to save the figure. Defaults to False.
        output_path (str, optional): Path to save the figure. Defaults to "".

    Returns:
        Axes: Matplotlib axes object containing the plot
    """
    # Plot cross section of the CWS
    ax = coil_factory.plot_cross_section(c="r")

    # Plot several cross sections of the LCFS
    Sp.plot_cross_sections(convex_envelope=False, ax=ax)

    # Save the figure if requested
    if save_fig:
        ax.savefig(output_path)

    return ax


def plot_CWS_LCFS(S: Surface, Sp: Surface, n_cyl: int, save_fig: bool = False, output_path: str = ""):
    """
    Plot in the XY plane:
    * the CWS boundaries projection
    * the contour lines of the LCFS Z param

    Args:
        S (Surface): Surface object representing the CWS
        Sp (Surface): Surface object representing the LCFS
        n_cyl (int): Number of cylindrical segments to plot
        save_fig (bool, optional): Flag to save the figure. Defaults to False.
        output_path (str, optional): Path to save the figure. Defaults to "".

    Returns:
        fig, ax: Matplotlib figure and axes objects containing the plot
    """
    # Compute the step size for each segment
    n_total_cyl = Sp.nfp*n_cyl+(n_cyl == 0)
    step_v = S.integration_par.num_points_v // n_total_cyl

    # Create a figure and axes object
    fig, ax = plt.subplots()

    for cyl in range(n_total_cyl):
        # Calculate indices for segment boundaries
        minus_idx = step_v * cyl
        plus_idx = step_v * (cyl+1)-1

        # Collect boundary points for minus and plus toroidal boundaries
        cyl_2d_external_boundary = S.xyz[0,
                                         (step_v * cyl):(step_v * (cyl + 1)), :]
        cyl_2d_internal_boundary = S.xyz[S.integration_par.num_points_u // (
            2), (step_v * cyl):(step_v * (cyl + 1)), :]
        cyl_2d_minus_tor_boundary = S.xyz[:, minus_idx, :]
        cyl_2d_plus_tor_boundary = S.xyz[:, plus_idx, :]

        # Plot segment boundaries
        ax.plot(cyl_2d_internal_boundary[:, 0], cyl_2d_internal_boundary[:,
                1], 'k-', label="internal", markersize=3)
        ax.plot(cyl_2d_external_boundary[:, 0], cyl_2d_external_boundary[:,
                1], 'k-', label="external", markersize=3)
        ax.plot(cyl_2d_minus_tor_boundary[:, 0], cyl_2d_minus_tor_boundary[:,
                1], 'k-', label="minus toroidal", markersize=3)
        ax.plot(cyl_2d_plus_tor_boundary[:, 0], cyl_2d_plus_tor_boundary[:,
                1], 'k-', label="plus toroidal", markersize=3)

    # Expand LCFS coordinates to plot the whole surface
    xyz = Sp.expand_for_plot_whole(nfp=5)[0]
    # Plot LCFS contours
    ax.contourf(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2],
                cmap='Reds', alpha=1, levels=20)

    # Set axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', 'datalim')
    ax.set_title("CWS and LCFS")

    # Save the figure if requested
    if save_fig:
        fig.savefig(output_path)

    return fig, ax


def plot_j2D_CWS(S: Surface, save_fig: bool = False, output_path: str = "") -> Tuple[plt.Figure, plt.Axes]:
    """
    Unwrap the CWS to plot j in 2D.

    Args:
        S (Surface): The surface object containing the CWS and its grids.
        save_fig (bool, optional): Whether to save the plot. Defaults to False.
        output_path (str, optional): The path to save the plot. Defaults to "".

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes object.
    """
    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(20, 20))

    # Normalize arrow length in such a way that the longest arrow is still visible
    scale_factor = 1 / onp.max(onp.linalg.norm(S.j_surface, axis=-1))

    # Introduce a downsampling factor to reduce the number of arrows
    downsampling_factor = 2
    ax.quiver(
        S.grids[1][::downsampling_factor, ::downsampling_factor],
        S.grids[0][::downsampling_factor, ::downsampling_factor],
        S.j_surface[::downsampling_factor,
                    ::downsampling_factor, 1] * scale_factor,
        S.j_surface[::downsampling_factor,
                    ::downsampling_factor, 0] * scale_factor,
        scale=20,
        angles='xy',
        color='r',
        alpha=0.75,
    )

    # Set axis labels and title
    ax.set_xlabel("v")
    ax.set_ylabel("u")
    ax.set_aspect('equal', 'box')
    ax.set_title('Current over the unwrapped coil winding surface')

    # Save the figure if requested
    if save_fig:
        fig.savefig(output_path)

    return fig, ax
