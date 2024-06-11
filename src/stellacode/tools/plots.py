import matplotlib.pyplot as plt
from stellacode.surface import Surface
import numpy as onp

def plot_CWS_LCFS(S: Surface, Sp: Surface, n_cyl :int , ax=None, save_fig:bool=False, output_path:str=""):
    """
    Plot in the XY plane:
    * the CWS boundaries projection
    * the contour lines of the LCFS Z param
    """
    # Compute the step size for each segment
    n_total_cyl = Sp.nfp*n_cyl+(n_cyl==0)
    step_v = S.integration_par.num_points_v // n_total_cyl

    fig, ax = plt.subplots()

    for cyl in range(n_total_cyl):
        minus_idx = step_v * cyl
        plus_idx = step_v * (cyl+1)-1

        # Collect boundary points for minus and plus toroidal boundaries
        cyl_2d_external_boundary = S.xyz[0,(step_v * cyl) :(step_v * (cyl + 1)),:]
        cyl_2d_internal_boundary = S.xyz[S.integration_par.num_points_u // (2),(step_v * cyl) :(step_v * (cyl + 1)),:]
        cyl_2d_minus_tor_boundary=S.xyz[:, minus_idx, :]
        cyl_2d_plus_tor_boundary=S.xyz[:, plus_idx, :]

        ax.plot(cyl_2d_internal_boundary[ :,0], cyl_2d_internal_boundary[: ,1],'k-', label="internal",markersize=3)
        ax.plot(cyl_2d_external_boundary[: ,0], cyl_2d_external_boundary[: ,1],'k-', label="external",markersize=3)
        ax.plot(cyl_2d_minus_tor_boundary[:,0], cyl_2d_minus_tor_boundary[:,1],'k-', label="minus toroidal",markersize=3)
        ax.plot(cyl_2d_plus_tor_boundary[:,0], cyl_2d_plus_tor_boundary[:,1],'k-', label="plus toroidal",markersize=3)

    xyz = Sp.expand_for_plot_whole(nfp=5)[0]
    ax.contourf(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], cmap='Reds', alpha=1, levels=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', 'datalim')
    ax.set_title("CWS and LCFS")

    if save_fig:
        fig.savefig(output_path)

    return fig, ax

def plot_j2D_CWS(S: Surface,ax=None, save_fig: bool=False, output_path:str=""):
    """
    Unwrap the CWS to plot j in 2D
    """
    # Plot the points contained in the variable "coil_surface.grids" over a 2D plane defined by 'u' and 'v'
    fig,ax = plt.subplots(figsize=(20,20))
    

    # normalize arrow lenght in such a way that the longest arrow is still visible
    scale_factor = 1 / onp.max(onp.linalg.norm(S.j_surface, axis=-1))

    # introduce in the line below a downsampling factor to reduce the number of arrows
    downsampling_factor = 2
    ax.quiver(
        S.grids[1][::downsampling_factor,::downsampling_factor],
        S.grids[0][::downsampling_factor,::downsampling_factor], 
        S.j_surface[::downsampling_factor,::downsampling_factor,1]*scale_factor, 
        S.j_surface[::downsampling_factor,::downsampling_factor,0]*scale_factor, 
        scale=20,
        angles='xy', 
        color='r', 
        alpha=0.75, 
    )

    ax.set_xlabel("v")
    ax.set_ylabel("u")

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', 'box')
    ax.set_title('Current over the unwrapped coil winding surface')

    if save_fig:
        fig.savefig(output_path)
    
    return fig,ax