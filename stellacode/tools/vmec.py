import numpy as np
from scipy.io import netcdf


class VMECIO:
    def __init__(self, vmec_file: str):
        self.file = netcdf.netcdf_file(vmec_file)

    def get_val(self, label: str, theta, zeta, radius_label):
        mnc = self.file.variables[f"{label}mnc"][()]
        
        xm = self.file.variables["xm_nyq"][()] 
        xn = self.file.variables["xm_nyq"][()] 

        angle = 2 * np.pi * (theta * xm + zeta * xn)
        val= np.tensordot(mnc, np.cos(angle), 1)
        if f"{label}mns" in self.file.variables.keys():
            mns = self.file.variables[f"{label}mns"][()]
            val+=np.tensordot(mns, np.sin(angle), 1)
        return val