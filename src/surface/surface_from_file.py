from src.surface.surface_Fourier import Surface_Fourier
from src.surface.surface_pwc_fourier import Surface_PWC_Fourier
from src.surface.surface_pwc_spline import Surface_PWC_Spline


def surface_from_file(pathfile):
    with open(pathfile, 'r') as f:
        first_line = next(f).strip()
    if first_line == "fourier":
        return Surface_Fourier.load_file(pathfile)
    elif first_line == "pwc fourier":
        return Surface_PWC_Fourier.load_file(pathfile)
    elif first_line == "pwc spline":
        return Surface_PWC_Spline.load_file(pathfile)
    else:
        raise ValueError(
            "The first line of your file does not correspond to any known surfaces.")
