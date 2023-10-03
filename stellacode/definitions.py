import typing as tp
from os.path import dirname, realpath, join

from pydantic import BaseModel


class PlasmaConfig(BaseModel):
    """
    Configuration of a plasma

    Args:
        * path_plasma: path to the plasma equilibrium result (vmec wout file)
        * minor_radius: average minor radius of the plasma
        * path_bnorm: patht to the bnorm file
        * path_cws: path to the coil winding surface
    """

    path_plasma: str
    minor_radius: float
    path_bnorm: tp.Optional[str] = None
    path_cws: tp.Optional[str] = None


configs_folder = join(f"{dirname(dirname(realpath(__file__)))}", "data")
w7x_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "w7x", "wout_d23p4_tm.nc"),
    minor_radius=0.53,
    path_bnorm=join(configs_folder, "w7x", "bnorm.d23p4_tm"),
    path_cws=join(configs_folder, "w7x", "nescin.w7x_winding_surface_from_Drevlak"),
)
ncsx_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "li383", "wout_li383_1.4m.nc"),
    minor_radius=0.33,
    path_bnorm=join(configs_folder, "li383", "bnorm.li383_1.4m"),
    path_cws=join(configs_folder, "li383", "nescin.li383_realWindingSurface"),
)
hsx_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "hsx", "wout_HSX_QHS.nc"),
    minor_radius=0.15,
    path_bnorm=None,
    path_cws=None,
)
