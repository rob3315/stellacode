from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs import EMCost
from stellacode.surface.coil_surface import GroovedCoilFactory
from jax import Array

class GroovesCost(AbstractCost):
    """
    Field precision for a given set of grooves.

    Args:
        * nfp: number of field periods
        * constraint: 
    """
    cyl_per_fp: int
    em_cost: EMCost
    groove_sets: list = []
    ground_truth_B: Array

    @classmethod
    def from_params(cls, em_cost, nb_cyl_per_fp, **kwargs):
        # Defining grooving parameters
        groove_sets = []
        for _ in range(nb_cyl_per_fp):
            groove_sets += [GroovedCoilFactory.from_params(**kwargs)]

        gt_b = np.array(em_cost.Sp.get_gt_b_field(b_norm_file=None))

        return cls(
            cyl_per_fp=nb_cyl_per_fp,
            em_cost=em_cost,
            groove_sets=groove_sets,
            ground_truth_B = gt_b,
        )
    def cost(self, S, results: Results = Results(),x = None):
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        nfp = self.em_cost.Sp.nfp # number of plasma field periods
        coil_surf = S().get_coil() # S.get_coil() #
        n_pts_v = coil_surf.integration_par.__dict__['num_points_v']
        ugrid, vgrid = coil_surf.grids
        # updated_params = S.phi_mn # S.get_phi_mn()
        # kwargs = dict(current = updated_params[0] / self.groove_sets[0].nb_grooves * (1 + updated_params[2]), v_ctrl_points_w = updated_params[3:])

        kwargs = dict(current = x[0] / self.groove_sets[0].nb_grooves, v_ctrl_points_w = x[1:])
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        j_2d_BE_cyl = []
        for cyl in range(self.cyl_per_fp):
            self.groove_sets[cyl].update_params(**kwargs)

            grid_uv_unwrapped = np.stack([ugrid[:,(cyl * n_pts_v // (nfp * self.cyl_per_fp)):((cyl + 1) * n_pts_v // (nfp * self.cyl_per_fp))], \
            vgrid[:,(cyl * n_pts_v // (nfp * self.cyl_per_fp)):((cyl + 1) * n_pts_v // (nfp * self.cyl_per_fp))] - cyl],axis=2)

            # Displacing the grid by half spatial step in both directions to avoid edge effects
            grid_xy = grid_uv_unwrapped.reshape(-1,2)[:,::-1]
            
            # Calculating the spatial step on u and v directions
            delta_u = np.mean(np.diff(grid_uv_unwrapped, axis=0).reshape(-1,2),axis=0)[0] * 0.5
            delta_v = np.mean(np.diff(grid_uv_unwrapped, axis=1).reshape(-1,2),axis=0)[1] * 0.5

            grid_xy += np.array([delta_v, delta_u])

            # Post-process the results for all subdomains
            e_x, e_y = np.array([*zip(*self.groove_sets[cyl].bem_solver.post_process(grid_xy)[1])])

            # Creating a j_2d per cylinder
            j_2d_BE_cyl += [np.stack((e_y.reshape(grid_uv_unwrapped.shape[0:2]),e_x.reshape(grid_uv_unwrapped.shape[0:2])),axis=2)]


        # Creating a j_2d that looks like the original stellacode S.j_surface
        j_2d_BE = np.empty((j_2d_BE_cyl[0].shape[0], 0, j_2d_BE_cyl[0].shape[2]))
        for cyl in range(self.cyl_per_fp):
            j_2d_BE = np.append(j_2d_BE,j_2d_BE_cyl[cyl],axis=1)

        j_2d_BE = np.tile(j_2d_BE, (1, nfp, 1))


        #  Creating a j_3d that looks like the original stellacode S.j_3d 
        j_3d_BE = np.einsum('ijkl,ijl,ij->ijk', coil_surf.jac_xyz, j_2d_BE,1 / coil_surf.ds)
        # j_3d_BE = np.einsum('ijkl,ijl->ijk', S.jac_xyz, j_2d_BE)

        coil_surf.j_3d = j_3d_BE.copy()

        pred_b = coil_surf.get_b_field(self.em_cost.Sp.xyz)
        deltaB_B_vec = np.linalg.norm(pred_b - self.ground_truth_B[:, : pred_b.shape[1]], axis=-1) / np.linalg.norm(self.ground_truth_B[:, : pred_b.shape[1]], axis=-1)
        
        # print(f"gt_b: {gt_b} [T]")
        # print(f"pred_b: {pred_b} [T]")
        # print(f"deltaB_B_vec: {deltaB_B_vec} [T]")

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        loss = np.linalg.norm(deltaB_B_vec)

        return loss, {"deltaB_B_groove_sets": loss}, results, S
