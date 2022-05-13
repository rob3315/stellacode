from src.surface.surface_Fourier import Surface_Fourier
from src.surface.surface_pwc import Surface_PWC


def surface_from_file(pathfile):
    with open(pathfile, 'r') as f:
        first_line = next(f).strip()
    if first_line == "fourier":
        return Surface_Fourier.load_file(pathfile)
    elif first_line == "pwc":
        return Surface_PWC.load_file(pathfile)
    else:
        raise ValueError(
            "The first line of your file does not correspond to any known surfaces.")
