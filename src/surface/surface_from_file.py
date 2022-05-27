from src.surface.surface_Fourier import Surface_Fourier
from src.surface.surface_pwc_fourier import Surface_PWC_Fourier


def surface_from_file(path_surf, n_fp, n_pol, n_tor):
    if path_surf[-3::] == ".nc":
        return Surface_Fourier.load_file(path_surf, n_fp, n_pol, n_tor)
    else:
        with open(path_surf, 'r') as f:
            first_line = next(f).strip()
        if first_line == "fourier":
            return Surface_Fourier.load_file(path_surf, n_fp, n_pol, n_tor)
        elif first_line == "pwc fourier":
            return Surface_PWC_Fourier.load_file(path_surf, n_fp, n_pol, n_tor)
        else:
            raise ValueError(
                "The first line of your file does not correspond to any known surfaces.")
