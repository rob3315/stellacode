from pydantic import BaseModel, Extra
import typing as tp
from os.path import dirname, realpath


class PlasmaConfig(BaseModel):
    path_plasma: str
    path_bnorm: tp.Optional[str] = None


configs_folder = f"{dirname(dirname(realpath(__file__)))}/data/"
w7x_plasma = PlasmaConfig(
    path_plasma=f"{configs_folder}w7x/wout_d23p4_tm.nc", path_bnorm=f"{configs_folder}w7x/bnorm.d23p4_tm"
)
ncsx_plasma = PlasmaConfig(
    path_plasma=f"{configs_folder}li383/wout_li383_1.4m.nc", path_bnorm=f"{configs_folder}li383/bnorm.li383_1.4m"
)
