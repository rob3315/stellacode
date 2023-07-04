from scipy.spatial.distance import cdist
from stellacode import np


def get_min_dist(S1, S2):
    return cdist(np.reshape(S1, (-1, 3)), np.reshape(S2, (-1, 3))).min()

    # slower but differentiable
    # return np.linalg.norm(S1.P[...,None,None,:]-S2.P[None,None,...], axis=-1).min()
