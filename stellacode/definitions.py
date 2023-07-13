from pydantic import BaseModel, Extra
import typing as tp


class PlasmaConfig(BaseModel):
    path_plasma: str 
    path_bnorm: tp.Optional[str] = None


class CoilConfig(BaseModel):
    path_coil: str