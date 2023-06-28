""" contained non linear functions (and there derivative) to impose constraints
"""
import logging
from functools import partial

from jax import jit

from stellacode import np


@partial(jit, static_argnums=(0, 1, 2))
def f_non_linear(d_min_hard, d_min_soft, d_min_penalization, x):
    """blow up before d_min_hard and zero after d_min_soft

    :param d_min_hard: cost is inf for x smaller than this value
    :type d_min_hard: float
    :param d_min_soft: cost is 0 for x bigger than this value
    :type d_min_soft: float
    :param d_min_penalization: a multiplicative constant in the cost
    :type d_min_penalization: float
    :param x: the argument
    :type x: float
    :rtype: float
    """
    # if x < d_min_hard:
    #     logging.info('minimal distance overeached')
    #     return np.inf
    #     #raise Exception('minimal distance overeached')
    # else:

    return (
        d_min_penalization
        * np.maximum(d_min_soft - x, 0) ** 2
        / (1 - np.maximum(d_min_soft - x, 0) / (d_min_soft - d_min_hard))
    )


@partial(jit, static_argnums=(0, 1))
def f_e(c0, c1, x):
    """blow up after c0 and zero before c1

    :param c0: cost is 0 for x smaller than this value
    :type c0: float
    :param c1: cost is inf for x bigger than this value
    :type c1: float
    :param x: the argument
    :type x: float
    :rtype: float
    """

    # if 0 <= x and x <= c1:
    return np.maximum(x - c0, 0) ** 2 / (1 - np.maximum(x - c0, 0) / (c1 - c0))
    # else:
    #     raise ValueError
