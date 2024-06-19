import jax.numpy as np
from scipy.constants import mu_0
from os.path import dirname, join, realpath

mu_0_fac = mu_0 / (4 * np.pi)


PROJECT_PATH = join(f"{dirname(dirname(dirname(realpath(__file__))))}")
DATA_PATH = join(PROJECT_PATH, "data")
HTS_DATABASE_PATH = join(DATA_PATH, "hts_params_jeroen.xlsx")
