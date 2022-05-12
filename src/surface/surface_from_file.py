from msilib.schema import Error
from surface_Fourier import Surface_Fourier
from surface_pwc import Surface_PWC


def surface_from_file(pathfile):
    with open(pathfile, 'r') as f:
        first_line = next(f)
    if first_line == "fourier":
        return Surface_Fourier.load_file(pathfile)
    elif first_line == "pwc":
        return Surface_PWC.load_file(pathfile)
    else:
        raise ValueError(
            "The first line of your file does not correspond to any known surfaces.")
