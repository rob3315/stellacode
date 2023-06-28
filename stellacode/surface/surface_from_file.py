from .pwc_surfaces.surface_pwc_ell_tri import Surface_PWC_Ell_Tri
from .pwc_surfaces.surface_pwc_ell_tri_3 import Surface_PWC_Ell_Tri_3
from .pwc_surfaces.surface_pwc_fourier import Surface_PWC_Fourier
from .pwc_surfaces.surface_pwc_fourier_3 import Surface_PWC_Fourier_3
from .fourier import FourierSurface


def surface_from_file(path_surf, n_fp, n_pol, n_tor):
    """This function reads a file and creates the corresponding Surface object.
    The following files are supported :
    - wout files from VMEC (ending with .nc)
    - nescin file (starting with nescin.)
    - text files
    - json files

    The following surfaces are supported :
    - Surface_Fourier
    - Surface_PWC_Fourier
    - Surface_PWC_Ell_Tri
    - Surface_PWC_Fourier_3
    - Surface_PWC_Ell_Tri_3

    Regarding the format of the text files, please have a look at the examples
    located in data/cws. Same for json files.

    :param path_surf: path to the file
    :type path_surf: string

    :param n_fp: number of field periods
    :type n_fp: int

    :param n_pol: number of points in the poloidal direction
    :type n_pol: int

    :param n_tor: number of points in the toroidal direction
    :type n_tor: int

    :return: the corresponding surface object
    :rtype: Surface
    """
    from os import sep

    file_extension = path_surf.split(".")[-1]

    if file_extension == "nc":
        return FourierSurface.from_file(path_surf, n_fp, n_pol, n_tor)

    elif path_surf.rpartition(sep)[-1][:6:] == "nescin":
        return FourierSurface.from_file(path_surf, n_fp, n_pol, n_tor)

    elif file_extension == "txt":
        with open(path_surf, "r") as f:
            first_line = next(f).strip()
        if first_line == "fourier":
            return FourierSurface.from_file(path_surf, n_fp, n_pol, n_tor)
        elif first_line == "pwc fourier":
            return Surface_PWC_Fourier.from_file(path_surf, n_fp, n_pol, n_tor)
        elif first_line == "pwc ellipticity triangularity":
            return Surface_PWC_Ell_Tri.from_file(path_surf, n_fp, n_pol, n_tor)
        elif first_line == "pwc fourier 3":
            return Surface_PWC_Fourier_3.from_file(path_surf, n_fp, n_pol, n_tor)
        elif first_line == "pwc ellipticity triangularity 3":
            return Surface_PWC_Ell_Tri_3.from_file(path_surf, n_fp, n_pol, n_tor)
        else:
            raise NotImplementedError

    elif file_extension == "json":
        import json

        with open(path_surf, "r") as f:
            data = json.load(f)
        n_cyl = data["surface"]["n_cyl"]
        parametrization = data["surface"]["parametrization"]

        if parametrization == "fourier":
            if n_cyl == 3:
                return Surface_PWC_Fourier_3.from_file(path_surf, n_fp, n_pol, n_tor)
            else:
                return Surface_PWC_Fourier.from_file(path_surf, n_fp, n_pol, n_tor)
        elif parametrization == "ell_tri":
            if n_cyl == 3:
                return Surface_PWC_Ell_Tri_3.from_file(path_surf, n_fp, n_pol, n_tor)
            else:
                return Surface_PWC_Ell_Tri.from_file(path_surf, n_fp, n_pol, n_tor)
        else:
            raise NotImplementedError(f"Parametrization: {parametrization} is not supported.")

    else:
        raise NotImplementedError(f"Parametrization: {parametrization} is not supported.")
