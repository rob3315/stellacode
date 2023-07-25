from pydantic import BaseModel, Extra
import typing as tp
from os.path import dirname, realpath


class PlasmaConfig(BaseModel):
    path_plasma: str
    path_bnorm: tp.Optional[str] = None
    path_cws: tp.Optional[str] = None


configs_folder = f"{dirname(dirname(realpath(__file__)))}/data/"
w7x_plasma = PlasmaConfig(
    path_plasma=f"{configs_folder}w7x/wout_d23p4_tm.nc",
    path_bnorm=f"{configs_folder}w7x/bnorm.d23p4_tm",
    path_cws=f"{configs_folder}w7x/nescin.w7x_winding_surface_from_Drevlak",
)
ncsx_plasma = PlasmaConfig(
    path_plasma=f"{configs_folder}li383/wout_li383_1.4m.nc",
    path_bnorm=f"{configs_folder}li383/bnorm.li383_1.4m",
    path_cws=f"{configs_folder}li383/nescin.li383_realWindingSurface",
)
hsx_plasma = PlasmaConfig(
    path_plasma=f"{configs_folder}HSX_QHS_vac_ns201_fixed/wout_HSX_QHS_vacuum_ns201.nc",
    # path_bnorm=f"{configs_folder}HSX_QHS_vac_ns201_fixed/bnorm.HSX_QHS_vacuum_ns201",
    # path_cws=f"{configs_folder}HSX_QHS_vac_ns201_fixed/nescin.HSX_QHS_vacuum_ns201",
)
