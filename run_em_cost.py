from stellacode.costs import EMCost, AggregateCost
from stellacode.surface import IntegrationParams
from stellacode.surface import WrappedCoil, FourierSurfaceFactory
from stellacode.optimizer import Optimizer
from stellacode.definitions import w7x_plasma

n_harmonics_phi = 4
n_harmonics_u = 4
n_harmonics_v = 4
factor = 4

num_cyl = 1  # num_cyl
make_joints = False  # oblique cut to make joints btw cylinders
init_dist = 0  # initial distance LCFS-CWS

lamb = 1e-27
train_currents = True
currentit = 1e8
is_convex = False

maxiter = 2
optim_method = "L-BFGS-B"

integration_par = IntegrationParams(
    num_points_u=n_harmonics_u * factor, num_points_v=n_harmonics_v * factor)

if make_joints:
    cut_tor = None
else:
    cut_tor = n_harmonics_phi * factor // num_cyl

plasma_config = w7x_plasma

Sp = FourierSurfaceFactory.from_file(
    plasma_config.path_plasma, integration_par=integration_par)

costs = []
em_cost = EMCost.from_plasma_config(
    lamb=lamb,
    plasma_config=plasma_config,
    integration_par=integration_par,
    train_currents=train_currents,
)
costs.append(em_cost)
agg_cost = AggregateCost(
    costs=costs
)

coil_factory = WrappedCoil.from_plasma(
    surf_plasma=em_cost.Sp,
    surf_type="cylindrical",
    sin_basis=True,
    cos_basis=False,
    make_joints=make_joints,
    distance=init_dist,
    match_surface=False,
    num_cyl=num_cyl,
    convex=is_convex,
    common_current_on_each_rot=False,
    n_harmonics=n_harmonics_phi,  # Number of harmonics in the current fourier expansion
    factor=factor,  # The number of points on the grid is n_harmonics * factor
)

opt = Optimizer.from_cost(
    agg_cost,
    coil_factory,
    method=optim_method,  # Quasi-Newton method very good for non constrained optimization
    kwargs=dict(
        options={"disp": True, "maxiter": maxiter},
    ),
    save_res=True,
    output_folder_name="outputs/"
)

cost, metrics, results, optimized_params = opt.optimize()
