import typing as tp
from os.path import dirname, realpath, join

from pydantic import BaseModel, Extra


class PlasmaConfig(BaseModel):
    path_plasma: str
    path_bnorm: tp.Optional[str] = None
    path_cws: tp.Optional[str] = None


configs_folder = join(f"{dirname(dirname(realpath(__file__)))}", "data")
w7x_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "w7x", "wout_d23p4_tm.nc"),
    path_bnorm=join(configs_folder, "w7x", "bnorm.d23p4_tm"),
    path_cws=join(configs_folder, "w7x", "nescin.w7x_winding_surface_from_Drevlak"),
)
ncsx_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "li383", "wout_li383_1.4m.nc"),
    path_bnorm=join(configs_folder, "li383", "bnorm.li383_1.4m"),
    path_cws=join(configs_folder, "li383", "nescin.li383_realWindingSurface"),
)
hsx_plasma = PlasmaConfig(
    path_plasma=join(configs_folder, "hsx", "wout_HSX_QHS.nc"),
    path_bnorm=None,
    path_cws=None,
)
