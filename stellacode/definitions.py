from pydantic import BaseModel, Extra
import typing as tp


class PlasmaConfig(BaseModel):
    path_plasma: str
    path_bnorm: tp.Optional[str] = None


w7x_plasma = PlasmaConfig(path_plasma="../data/w7x/wout_d23p4_tm.nc", path_bnorm="../data/w7x/bnorm.d23p4_tm")
ncsx_plasma = PlasmaConfig(path_plasma="../data/li383/wout_li383_1.4m.nc", path_bnorm="../data/li383/bnorm.li383_1.4m")
